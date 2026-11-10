import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert to integer matrix: A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            mat[i, j] = base_map[c]
    
    # Compute pairwise Hamming distances vectorized
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Vectorized: count mismatches for taxon i vs all others
        diff = (mat[i:i+1, :] != mat[1:, :]).sum(axis=1)
        p = diff / L
        # JC correction
        p = np.clip(p, 0, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        dist_mat[i, i+1:] = d
        dist_mat[i+1:, i] = d
    
    return dist_mat

def _nj_build_tree(dist_mat, labels):
    n = len(labels)
    if n <= 2:
        return None, None
    
    # D is the full distance matrix, we'll maintain it as we merge
    D = dist_mat.copy()
    
    # Active nodes: initially 0..n-1
    active = list(range(n))
    next_id = n
    
    # tree_edges: node -> (child1, child2, len1, len2)
    tree_edges = {}
    
    while len(active) > 3:
        k = len(active)
        
        # Row sums: r[i] = sum of D[active[i], active[j]] for j != i
        # Vectorized computation
        active_arr = np.array(active)
        # Get submatrix
        sub_D = D[np.ix_(active_arr, active_arr)]
        r = sub_D.sum(axis=1)
        
        # Q matrix: q[i,j] = (k-2)*D[i,j] - r[i] - r[j]
        # Find minimum Q
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i+1, k):
                q = (k - 2) * sub_D[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Calculate branch lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (sub_D[best_i, best_j] + delta)
        len_b = 0.5 * (sub_D[best_i, best_j] - delta)
        
        # Ensure non-negative lengths
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        new_node = next_id
        next_id += 1
        
        tree_edges[new_node] = (node_a, node_b, len_a, len_b)
        
        # Update distances to new node
        # d(new, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        for idx, x in enumerate(active):
            if x != node_a and x != node_b:
                val = 0.5 * (D[node_a, x] + D[node_b, x] - D[node_a, node_b])
                if val < 0:
                    val = 0.0
                D[new_node, x] = val
                D[x, new_node] = val
        
        # Remove a and b, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
    
    # Handle the final 3 nodes
    if len(active) == 3:
        u, v, w = active
        len_u = 0.5 * (D[u, v] + D[u, w] - D[v, w])
        len_v = 0.5 * (D[u, v] + D[v, w] - D[u, w])
        len_w = 0.5 * (D[u, w] + D[v, w] - D[u, v])
        
        # Clamp to non-negative
        len_u = max(0, len_u)
        len_v = max(0, len_v)
        len_w = max(0, len_w)
        
        new_node_1 = next_id
        next_id += 1
        tree_edges[new_node_1] = (u, v, len_u, len_v)
        
        new_node_2 = next_id
        next_id += 1
        tree_edges[new_node_2] = (new_node_1, w, 0.0, len_w)
        
        root = new_node_2
    else:
        root = None
    
    return tree_edges, root

def _build_newick(tree_edges, root, labels, n):
    if root is None:
        return ",".join(labels)
    
    # Iterative Newick builder to avoid recursion limits
    # Use a stack-based approach
    result = []
    stack = [(root, False)]
    
    while stack:
        node, visited = stack.pop()
        
        if node < n:
            # Leaf node
            result.append(labels[node])
            continue
        
        if node not in tree_edges:
            # Fallback
            result.append(f"n{node}")
            continue
        
        child_a, child_b, len_a, len_b = tree_edges[node]
        
        if not visited:
            # First time visiting this internal node
            # Push back with visited=True, then push children
            stack.append((node, True))
            stack.append((child_b, False))
            stack.append((child_a, False))
        else:
            # Children have been processed
            # The last two items in result should be the subtrees for child_a and child_b
            sub_b = result.pop()
            sub_a = result.pop()
            result.append(f"({sub_a}:{len_a:.4f},{sub_b}:{len_b:.4f})")
    
    return result[0] if result else ""

def _get_splits(tree_edges, root, labels, n):
    """Get all non-trivial splits from the tree."""
    if root is None:
        return set()
    
    splits = set()
    
    # For each internal node, compute the split it defines
    # A split is defined by the set of leaves in one subtree vs the rest
    
    # First, compute for each node the set of leaves in its subtree
    leaf_sets = {}
    
    def compute_leaf_set(node):
        if node in leaf_sets:
            return leaf_sets[node]
        
        if node < n:
            leaf_sets[node] = frozenset([node])
            return leaf_sets[node]
        
        if node not in tree_edges:
            leaf_sets[node] = frozenset()
            return leaf_sets[node]
        
        child_a, child_b, _, _ = tree_edges[node]
        set_a = compute_leaf_set(child_a)
        set_b = compute_leaf_set(child_b)
        leaf_sets[node] = set_a | set_b
        return leaf_sets[node]
    
    compute_leaf_set(root)
    
    # For each internal node, the split is (leaf_set, complement)
    all_leaves = frozenset(range(n))
    
    for node, (child_a, child_b, _, _) in tree_edges.items():
        if node < n:
            continue
        
        # Get the leaf set for this node's subtree
        # But we need to consider which side of the parent
        # Actually, for an unrooted tree, each internal edge defines a split
        # Let's compute splits differently: for each internal node, 
        # the split is the set of leaves in the subtree rooted at that node
        
        # Actually, in an unrooted binary tree, each internal edge corresponds to a split.
        # Let's compute it by considering each child of an internal node.
        
        pass
    
    # Better approach: for each internal node, compute the set of leaves in its subtree
    # The split is (subtree_leaves, all_leaves - subtree_leaves)
    # But we need to avoid trivial splits (size 1 or n-1)
    
    for node in tree_edges:
        if node < n:
            continue
        
        ls = leaf_sets.get(node, frozenset())
        if 1 < len(ls) < n - 1:
            # Canonical form: smaller set first
            comp = all_leaves - ls
            if len(ls) <= len(comp):
                split = (ls, comp)
            else:
                split = (comp, ls)
            splits.add(split)
    
    return splits

def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance."""
    if not splits1 and not splits2:
        return 0.0
    
    s1 = set(splits1)
    s2 = set(splits2)
    
    common = s1 & s2
    diff = (s1 - s2) | (s2 - s1)
    
    total = len(s1 | s2)
    if total == 0:
        return 0.0
    
    return len(diff) / total

def _spr_refine(tree_edges, root, dist_mat, labels, n, time_limit):
    """Simple SPR (Subtree Prune and Regraft) refinement."""
    if root is None or n < 4:
        return tree_edges, root
    
    start_time = time.time()
    
    # Build a list of all internal nodes
    internal_nodes = [node for node in tree_edges if node >= n]
    
    if not internal_nodes:
        return tree_edges, root
    
    # For each internal node, try pruning and regrafting
    # This is a simplified version - just try a few moves
    
    improved = False
    max_iterations = 10  # Limit to avoid timeout
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        found_improvement = False
        
        # Try moving each internal node's subtree
        for node in list(internal_nodes):
            if node not in tree_edges:
                continue
            
            child_a, child_b, len_a, len_b = tree_edges[node]
            
            # Try pruning child_a and regrafting elsewhere
            # This is complex; for now, just do a simple swap
            # Swap children at this node
            tree_edges[node] = (child_b, child_a, len_b, len_a)
            
            # Check if this helps (simplified: just accept if it doesn't break)
            found_improvement = True
            break
        
        if not found_improvement:
            break
    
    return tree_edges, root

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0  # 5s margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        if n == 1:
            return f"{labels[0]};"
        return ""
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # Time check
    if time.time() - start_time > time_limit * 0.3:
        # Fallback: star tree
        inner = ",".join(labels)
        return f"({inner})"
    
    # 2. Neighbor Joining
    tree_edges, root = _nj_build_tree(dist_mat, labels)
    
    if root is None:
        # Fallback
        inner = ",".join(labels)
        return f"({inner})"
    
    # Time check
    if time.time() - start_time > time_limit * 0.6:
        newick = _build_newick(tree_edges, root, labels, n)
        return newick
    
    # 3. Build Newick
    newick = _build_newick(tree_edges, root, labels, n)
    
    # Final time check
    if time.time() - start_time > time_limit * 0.9:
        return newick
    
    return newick
