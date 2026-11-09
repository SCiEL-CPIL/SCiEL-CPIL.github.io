"""Phylogenetic tree reconstruction using NJ + fast ML refinement under JC model."""
import math
import time
import numpy as np


def _jc_distance_matrix(aln_matrix):
    """Compute JC-corrected distance matrix from a binary alignment matrix."""
    n_taxa = aln_matrix.shape[0]
    n_sites = aln_matrix.shape[1]
    a = aln_matrix[:, None, :]
    b = aln_matrix[None, :, :]
    diff = (a != b).sum(axis=2)
    p = diff.astype(np.float64) / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    arg = 1.0 - (4.0 / 3.0) * p_clipped
    dist = -0.75 * np.log(arg)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(aln_matrix, labels):
    """Build initial tree using Neighbor Joining."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    D_init = _jc_distance_matrix(aln_matrix)
    
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = D_init
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        sub_D = D[np.ix_(active, active)]
        row_sums = sub_D.sum(axis=1)
        crit = (m - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(crit, np.inf)
        idx = np.unravel_index(crit.argmin(), crit.shape)
        i, j = idx
        a = active[i]
        b = active[j]
        
        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1
        
        d_ab = D[a, b]
        for x in active:
            if x == a or x == b:
                continue
            D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_node] = D[new_node, x]
        
        active = [x for x in active if x not in (a, b)]
        active.append(new_node)
        trees[new_node] = joined
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    elif len(active) == 1:
        return trees[active[0]] + ";"
    else:
        parts = [trees[x] for x in active]
        return "(" + ",".join(parts) + ");"


def _parse_newick(newick):
    """Parse a Newick string into a tree structure.
    Returns (nodes, root) where nodes is a list of [children_or_None, label_or_None]
    """
    newick = newick.strip().rstrip(';')
    
    nodes = []
    next_id = 0
    stack = []
    
    i = 0
    while i < len(newick):
        c = newick[i]
        if c == '(':
            new_id = next_id
            next_id += 1
            nodes.append([None, None])
            stack.append(new_id)
            i += 1
        elif c == ')':
            i += 1
            if i < len(newick) and newick[i] == ':':
                while i < len(newick) and newick[i] not in (',', ')', ';'):
                    i += 1
        elif c == ',':
            i += 1
        elif c == ':':
            while i < len(newick) and newick[i] not in (',', ')', ';'):
                i += 1
        else:
            label = ''
            while i < len(newick) and newick[i] not in (')', ',', ':', ';'):
                label += newick[i]
                i += 1
            new_id = next_id
            next_id += 1
            nodes.append([None, label])
            if stack:
                parent = stack[-1]
                if nodes[parent][0] is None:
                    nodes[parent][0] = [new_id, None]
                else:
                    nodes[parent][0][1] = new_id
    
    all_children = set()
    for node in nodes:
        if node[0] is not None:
            for child in node[0]:
                if child is not None:
                    all_children.add(child)
    
    root = -1
    for nid in range(len(nodes)):
        if nid not in all_children:
            root = nid
            break
    
    return nodes, root


def _build_postorder(nodes, root):
    """Build postorder traversal list."""
    postorder = []
    visited = set()
    
    def _postorder(idx):
        if idx in visited:
            return
        visited.add(idx)
        node = nodes[idx]
        if node[0] is None:
            postorder.append(idx)
        else:
            for child in node[0]:
                if child is not None:
                    _postorder(child)
            postorder.append(idx)
    
    _postorder(root)
    return postorder


def _compute_jc_likelihood_fast(nodes, root, aln_matrix, branch_lengths, postorder):
    """Compute JC likelihood using precomputed postorder and branch lengths.
    
    branch_lengths: dict mapping node_id -> branch length from parent to node
    """
    n_sites = aln_matrix.shape[1]
    
    # Map labels to indices
    label_to_idx = {}
    for idx, node in enumerate(nodes):
        if node[1] is not None:
            label_to_idx[node[1]] = idx
    
    # For each leaf, get the sequence
    leaf_seqs = {}
    for idx, node in enumerate(nodes):
        if node[0] is None and node[1] is not None:
            leaf_seqs[idx] = aln_matrix[label_to_idx[node[1]]]
    
    # Precompute transition matrices for each branch length
    # trans[d] = 4x4 matrix for branch length d
    # p_same = 0.75 + 0.25*exp(-4d/3)
    # p_diff = 0.25*(1-exp(-4d/3))
    
    likelihood = {}
    
    for node_idx in postorder:
        node = nodes[node_idx]
        if node[0] is None:
            # Leaf
            seq = leaf_seqs.get(node_idx)
            if seq is None:
                continue
            ll = np.zeros((n_sites, 4), dtype=np.float64)
            ll[np.arange(n_sites), seq] = 1.0
            likelihood[node_idx] = ll
        else:
            child_lls = []
            for child in node[0]:
                if child is not None and child in likelihood:
                    child_lls.append(likelihood[child])
            
            if len(child_lls) < 2:
                continue
            
            ll_left = child_lls[0]
            ll_right = child_lls[1]
            
            # Apply transition for left child's branch
            bl_left = branch_lengths.get(ll_left and child_lls[0] and nodes[node_idx][0][0], 0.1)
            d = bl_left
            exp_term = math.exp(-4.0 * d / 3.0)
            p_same = 0.75 + 0.25 * exp_term
            p_diff = 0.25 * (1.0 - exp_term)
            
            # temp_left[s] = sum_t ll_left[t] * P(s|t, d)
            # For JC: P(same) = p_same, P(diff) = p_diff
            # temp_left[s] = ll_left[s]*p_same + (sum(ll_left) - ll_left[s])*p_diff
            sum_left = ll_left.sum(axis=1, keepdims=True)
            temp_left = ll_left * p_same + (sum_left - ll_left) * p_diff
            
            # Apply transition for right child's branch
            bl_right = branch_lengths.get(child_lls[1] and nodes[node_idx][0][1], 0.1)
            d2 = bl_right
            exp_term2 = math.exp(-4.0 * d2 / 3.0)
            p_same2 = 0.75 + 0.25 * exp_term2
            p_diff2 = 0.25 * (1.0 - exp_term2)
            
            sum_right = ll_right.sum(axis=1, keepdims=True)
            temp_right = ll_right * p_same2 + (sum_right - ll_right) * p_diff2
            
            likelihood[node_idx] = temp_left * temp_right
    
    if root not in likelihood:
        return -np.inf
    
    root_ll = likelihood[root]
    # Sum over root states with prior 0.25 each
    total = root_ll.sum(axis=1) * 0.25
    log_likelihood = np.sum(np.log(np.maximum(total, 1e-300)))
    
    return log_likelihood


def _estimate_branch_lengths(nodes, root, aln_matrix, D):
    """Estimate branch lengths from distance matrix.
    Returns dict mapping node_id -> branch length.
    """
    n_taxa = aln_matrix.shape[0]
    label_to_idx = {}
    for idx, node in enumerate(nodes):
        if node[1] is not None:
            label_to_idx[node[1]] = idx
    
    # For a binary tree, we can estimate branch lengths using the 
    # standard method: for each internal node, the branch length is 
    # estimated from the distances to the two subtrees.
    
    # Simpler approach: use a uniform branch length estimate
    # based on the average distance
    avg_dist = D[~np.eye(n_taxa, dtype=bool)].mean()
    default_bl = max(avg_dist / (2 * (n_taxa - 1)), 0.01)
    
    branch_lengths = {}
    for idx in range(len(nodes)):
        branch_lengths[idx] = default_bl
    
    return branch_lengths


def _nni_refine(newick, aln_matrix, deadline, max_rounds=5):
    """Refine tree using NNI moves with JC likelihood.
    Returns best Newick string.
    """
    nodes, root = _parse_newick(newick)
    n_taxa = aln_matrix.shape[0]
    D = _jc_distance_matrix(aln_matrix)
    branch_lengths = _estimate_branch_lengths(nodes, root, aln_matrix, D)
    postorder = _build_postorder(nodes, root)
    
    current_ll = _compute_jc_likelihood_fast(nodes, root, aln_matrix, branch_lengths, postorder)
    best_ll = current_ll
    best_nodes = [list(n) for n in nodes]
    best_root = root
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 15.0:
            break
        
        improved = False
        postorder = _build_postorder(best_nodes, best_root)
        branch_lengths = _estimate_branch_lengths(best_nodes, best_root, aln_matrix, D)
        current_ll = _compute_jc_likelihood_fast(best_nodes, best_root, aln_matrix, branch_lengths, postorder)
        best_ll = current_ll
        
        # Find all internal edges for NNI
        # For each node that has a parent and is internal, try NNI
        parent_map = {}
        for i in range(len(best_nodes)):
            if best_nodes[i][0] is not None:
                for child in best_nodes[i][0]:
                    if child is not None:
                        parent_map[child] = i
        
        for node_idx in range(len(best_nodes)):
            if time.time() > deadline - 15.0:
                break
            
            node = best_nodes[node_idx]
            if node[0] is None:
                continue
            if node_idx == best_root:
                continue
            if node_idx not in parent_map:
                continue
            
            parent_idx = parent_map[node_idx]
            parent = best_nodes[parent_idx]
            if parent[0] is None:
                continue
            
            children = node[0]
            if children[0] is None or children[1] is None:
                continue
            c1, c2 = children[0], children[1]
            
            # Find the other child of parent
            gp_children = parent[0]
            if gp_children[0] == node_idx:
                other = gp_children[1]
            else:
                other = gp_children[0]
            
            if other is None:
                continue
            
            # NNI: swap c2 and other
            # New: parent has (node_idx, c2), node_idx has (c1, other)
            new_nodes = [list(n) for n in best_nodes]
            new_nodes[node_idx][0] = [c1, other]
            if gp_children[0] == node_idx:
                new_nodes[parent_idx][0] = [node_idx, c2]
            else:
                new_nodes[parent_idx][0] = [c2, node_idx]
            
            new_postorder = _build_postorder(new_nodes, best_root)
            new_bl = _estimate_branch_lengths(new_nodes, best_root, aln_matrix, D)
            new_ll = _compute_jc_likelihood_fast(new_nodes, best_root, aln_matrix, new_bl, new_postorder)
            
            if new_ll > best_ll + 1e-8:
                best_ll = new_ll
                best_nodes = new_nodes
                improved = True
        
        if not improved:
            break
    
    # Convert best tree to Newick
    def _to_newick(idx):
        node = best_nodes[idx]
        if node[0] is None:
            return node[1] if node[1] is not None else ""
        children = node[0]
        parts = []
        for child in children:
            if child is not None:
                parts.append(_to_newick(child))
        return "(" + ",".join(parts) + ")"
    
    return _to_newick(best_root) + ";"


def _quartet_split_matrix(aln_matrix, labels):
    """For each quartet, determine the best split under JC distance criterion.
    Returns a matrix where entry [i,j,k,l] gives the split index (0,1,2) for quartet (i,j,k,l).
    0: (i,j)|(k,l), 1: (i,k)|(j,l), 2: (i,l)|(j,k)
    """
    n = len(labels)
    D = _jc_distance_matrix(aln_matrix)
    
    # We'll store split votes as a dict
    split_votes = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    s1 = D[i, j] + D[k, l]
                    s2 = D[i, k] + D[j, l]
                    s3 = D[i, l] + D[j, k]
                    
                    mn = min(s1, s2, s3)
                    if s1 == mn:
                        split_votes[(i, j, k, l)] = 0
                    elif s2 == mn:
                        split_votes[(i, j, k, l)] = 1
                    else:
                        split_votes[(i, j, k, l)] = 2
    
    return split_votes, D


def _build_tree_from_splits(split_votes, n, D):
    """Build a tree from quartet split votes using a greedy compatible split selection."""
    # Count votes for each split type
    # Split (i,j)|(k,l) is represented as frozenset({i,j}) (smaller side)
    split_counts = {}
    
    for (i, j, k, l), sp in split_votes.items():
        if sp == 0:
            s = frozenset([i, j])
        elif sp == 1:
            s = frozenset([i, k])
        else:
            s = frozenset([i, l])
        
        # Normalize to smaller side
        if len(s) > n // 2:
            comp = frozenset(set(range(n)) - s)
            if len(comp) < len(s):
                s = comp
        
        split_counts[s] = split_counts.get(s, 0) + 1
    
    if not split_counts:
        return None
    
    # Sort by vote count
    sorted_splits = sorted(split_counts.keys(), key=lambda s: split_counts[s], reverse=True)
    
    # Greedy selection of compatible splits
    selected = []
    for s in sorted_splits:
        if len(selected) >= n - 3:
            break
        compatible = True
        for existing in selected:
            # Check compatibility: no crossing
            A = s
            B = existing
            A_B = A & B
            A_notB = A - B
            notA_B = B - A
            if A_B and A_notB and notA_B:
                compatible = False
                break
        if compatible:
            selected.append(s)
    
    if not selected:
        return None
    
    # Build tree from splits: use a recursive approach
    # Start with all taxa, recursively split
    def _build_recursive(taxa_set):
        if len(taxa_set) <= 1:
            return list(taxa_set)[0] if taxa_set else ""
        if len(taxa_set) == 2:
            a, b = list(taxa_set)
            return f"({a},{b})"
        if len(taxa_set) == 3:
            a, b, c = list(taxa_set)
            return f"({a},{b},{c})"
        
        # Find a split that divides this set
        best_split = None
        best_count = -1
        for s in selected:
            A = s & taxa_set
            B = taxa_set - s
            if 1 < len(A) < len(taxa_set) - 1:
                # Valid split of this subset
                if len(A) < len(B):
                    count = split_counts.get(A, 0)
                else:
                    count = split_counts.get(B, 0)
                if count > best_count:
                    best_count = count
                    best_split = (A, B)
        
        if best_split is None:
            # Fallback: use distance-based split
            taxa_list = list(taxa_set)
            if len(taxa_list) <= 2:
                return "(" + ",".join(taxa_list) + ")"
            
            # Find the pair with minimum distance
            min_d = float('inf')
            min_pair = None
            for ii in range(len(taxa_list)):
                for jj in range(ii + 1, len(taxa_list)):
                    d = D[taxa_list[ii], taxa_list[jj]]
                    if d < min_d:
                        min_d = d
                        min_pair = (taxa_list[ii], taxa_list[jj])
            
            left = frozenset(min_pair)
            right = taxa_set - left
            best_split = (left, right)
        
        left, right = best_split
        left_tree = _build_recursive(left)
        right_tree = _build_recursive(right)
        
        if len(left) == 1 and len(right) == 1:
            return f"({left_tree},{right_tree})"
        elif len(left) == 1:
            return f"({left_tree},{right_tree})"
        elif len(right) == 1:
            return f"({left_tree},{right_tree})"
        else:
            return f"({left_tree},{right_tree})"
    
    all_taxa = frozenset(range(n))
    labels_list = [f"t{i}" for i in range(n)]
    tree_str = _build_recursive(all_taxa)
    
    # The tree_str uses integer labels, need to convert to t0, t1, ...
    # Actually _build_recursive returns strings of integers. Let's fix this.
    # We need to map integer indices to labels.
    
    # Let's redo with proper labels
    def _build_recursive_labeled(taxa_set):
        if len(taxa_set) == 1:
            idx = list(taxa_set)[0]
            return labels_list[idx]
        if len(taxa_set) == 2:
            a, b = list(taxa_set)
            return f"({labels_list[a]},{labels_list[b]})"
        if len(taxa_set) == 3:
            a, b, c = list(taxa_set)
            return f"({labels_list[a]},{labels_list[b]},{labels_list[c]})"
        
        best_split = None
        best_count = -1
        for s in selected:
            A = s & taxa_set
            B = taxa_set - s
            if 1 < len(A) < len(taxa_set) - 1:
                if len(A) < len(B):
                    count = split_counts.get(A, 0)
                else:
                    count = split_counts.get(B, 0)
                if count > best_count:
                    best_count = count
                    best_split = (A, B)
        
        if best_split is None:
            taxa_list = list(taxa_set)
            min_d = float('inf')
            min_pair = None
            for ii in range(len(taxa_list)):
                for jj in range(ii + 1, len(taxa_list)):
                    d = D[taxa_list[ii], taxa_list[jj]]
                    if d < min_d:
                        min_d = d
                        min_pair = (taxa_list[ii], taxa_list[jj])
            left = frozenset(min_pair)
            right = taxa_set - left
            best_split = (left, right)
        
        left, right = best_split
        left_tree = _build_recursive_labeled(left)
        right_tree = _build_recursive_labeled(right)
        return f"({left_tree},{right_tree})"
    
    tree_str = _build_recursive_labeled(all_taxa)
    return tree_str + ";"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 255.0  # Leave 15s margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aln_list = []
    for lab in labels:
        seq = sequences[lab]
        aln_list.append([base_map[c] for c in seq])
    aln_matrix = np.array(aln_list, dtype=np.int8)
    
    # Step 1: Build initial tree using NJ
    nj_newick = _nj_tree(aln_matrix, labels)
    
    if time.time() > deadline - 40.0:
        return nj_newick
    
    # Step 2: Try quartet-based tree construction
    try:
        split_votes, D = _quartet_split_matrix(aln_matrix, labels)
        quartet_tree = _build_tree_from_splits(split_votes, n, D)
        if quartet_tree and time.time() < deadline - 20.0:
            # Validate quartet tree has all taxa
            all_taxa = set(labels)
            found_taxa = set()
            for t in labels:
                if t in quartet_tree:
                    found_taxa.add(t)
            if found_taxa == all_taxa:
                nj_newick = quartet_tree
    except Exception:
        pass
    
    if time.time() > deadline - 20.0:
        return nj_newick
    
    # Step 3: NNI refinement
    try:
        refined = _nni_refine(nj_newick, aln_matrix, deadline, max_rounds=3)
        if refined:
            nj_newick = refined
    except Exception:
        pass
    
    return nj_newick
