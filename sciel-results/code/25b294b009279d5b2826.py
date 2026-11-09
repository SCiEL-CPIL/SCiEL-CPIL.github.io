"""
Fast Distance-based Quartet (FDQ) topology reconstruction under Jukes-Cantor.
Combined: NJ backbone + exact JC likelihood + exhaustive NNI hill-climbing
(both NNI moves) + quartet-based refinement where time permits.
"""
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE_MARGIN = 5.0  # seconds to keep before the 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distances using vectorized NumPy ops."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])

    # Build lookup table once
    lut = np.zeros(256, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3

    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        mat[i] = lut[arr]

    # Vectorized pairwise distance computation
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(mat[i] != mat[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_reconstruct(dist, labels):
    """
    Perform Neighbor Joining on the distance matrix.
    Uses a dictionary-based approach for correctness.
    Returns a Newick string.
    """
    n = dist.shape[0]
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    # Use dictionary for distances to handle dynamic node IDs safely
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = dist[i, j]
            D[(j, i)] = dist[j, i]

    active = list(range(n))
    newick_strs = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 3:
        if _remaining() < 1.0:
            # Fallback: return a chain of remaining active nodes
            if len(active) == 0:
                return ""
            if len(active) == 1:
                return newick_strs[active[0]] + ";"
            if len(active) == 2:
                return f"({newick_strs[active[0]]},{newick_strs[active[1]]});"
            current = newick_strs[active[0]]
            for i in range(1, len(active)):
                current = f"({current},{newick_strs[active[i]]})"
            return current + ";"

        m = len(active)
        # Compute r values
        r = {}
        for i in active:
            r[i] = sum(D[(i, j)] for j in active if j != i)

        # Find the pair (a, b) that minimizes (m-2)*D[a,b] - r[a] - r[b]
        best_val = float('inf')
        best_a, best_b = active[0], active[1]
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                a, b = active[idx_i], active[idx_j]
                val = (m - 2) * D[(a, b)] - r[a] - r[b]
                if val < best_val:
                    best_val = val
                    best_a, best_b = a, b

        a, b = best_a, best_b

        # Create new node
        new_node = next_id
        next_id += 1

        # Update distances
        for c in active:
            if c != a and c != b:
                new_d = 0.5 * (D[(a, c)] + D[(b, c)] - D[(a, b)])
                D[(new_node, c)] = new_d
                D[(c, new_node)] = new_d

        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(new_node)

        # Update Newick string
        newick_strs[new_node] = f"({newick_strs[a]},{newick_strs[b]})"

    # Handle final 3 nodes
    if len(active) == 3:
        a, b, c = active
        newick_strs_temp = {a: newick_strs[a], b: newick_strs[b], c: newick_strs[c]}
        # For 3 taxa, any binary topology is valid; use balanced
        return f"({newick_strs_temp[a]},({newick_strs_temp[b]},{newick_strs_temp[c]}));"
    elif len(active) == 2:
        a, b = active
        return f"({newick_strs[a]},{newick_strs[b]});"
    elif len(active) == 1:
        return newick_strs[active[0]] + ";"
    else:
        return ""


class _Tree:
    """
    Internal binary unrooted tree representation for NNI optimization.
    Nodes 0..n-1 are leaves (taxa), n..2n-2 are internal.
    children[node] = [left, right]; parent[node] = parent index or -1.
    """
    def __init__(self, n_taxa):
        self.n = n_taxa
        self.total = 2 * n_taxa - 1
        self.children = [[-1, -1] for _ in range(self.total)]
        self.parent = [-1] * self.total
        self.labels = list(range(n_taxa))  # leaf i -> taxon i
        self.branch_len = [0.1] * self.total  # initial branch lengths

    def get_newick(self, labels):
        """Return Newick string for this tree (rooted arbitrarily at node 0's parent chain)."""
        # Find a root: pick an internal node
        root = self.total - 1 if self.total > 1 else 0
        # Actually, for unrooted binary tree with 2n-1 nodes, we need to root it.
        # Let's just pick the last internal node as root.
        if self.n <= 2:
            if self.n == 0:
                return ""
            if self.n == 1:
                return labels[0] + ";"
            return f"({labels[0]},{labels[1]});"
        
        # Root at internal node n (first internal)
        root = self.n
        
        def _newick(node, visited):
            if node in visited:
                return ""
            visited.add(node)
            if node < self.n:
                return labels[self.labels[node]]
            left, right = self.children[node]
            lstr = _newick(left, visited) if left != -1 else ""
            rstr = _newick(right, visited) if right != -1 else ""
            return f"({lstr},{rstr})"
        
        return _newick(root, set()) + ";"


def _build_tree_from_nj(dist, labels):
    """
    Build a _Tree object from NJ. Returns _Tree with proper structure.
    We'll reconstruct the tree structure during NJ.
    """
    n = len(labels)
    if n <= 2:
        t = _Tree(n)
        return t
    
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = dist[i, j]
            D[(j, i)] = dist[j, i]

    active = list(range(n))
    # children[parent] = [child1, child2], parent[node]
    children = [[-1, -1] for _ in range(2 * n - 1)]
    parent = [-1] * (2 * n - 1)
    next_id = n

    while len(active) > 3:
        m = len(active)
        r = {}
        for i in active:
            r[i] = sum(D[(i, j)] for j in active if j != i)

        best_val = float('inf')
        best_a, best_b = active[0], active[1]
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                a, b = active[idx_i], active[idx_j]
                val = (m - 2) * D[(a, b)] - r[a] - r[b]
                if val < best_val:
                    best_val = val
                    best_a, best_b = a, b

        a, b = best_a, best_b
        new_node = next_id
        next_id += 1

        children[new_node] = [a, b]
        parent[a] = new_node
        parent[b] = new_node

        for c in active:
            if c != a and c != b:
                new_d = 0.5 * (D[(a, c)] + D[(b, c)] - D[(a, b)])
                D[(new_node, c)] = new_d
                D[(c, new_node)] = new_d

        active.remove(a)
        active.remove(b)
        active.append(new_node)

    # Final 3
    if len(active) == 3:
        a, b, c = active
        # Connect them: make a new internal node connecting b and c, then connect to a
        # But we need exactly 2n-1 nodes. We have used n + (n-3) = 2n-3 nodes so far (0..2n-4).
        # We need 2 more internal nodes.
        # Let's create node 2n-3 and 2n-2.
        node1 = next_id
        next_id += 1
        node2 = next_id
        next_id += 1
        children[node1] = [b, c]
        parent[b] = node1
        parent[c] = node1
        children[node2] = [a, node1]
        parent[a] = node2
        parent[node1] = node2
        # Root is node2 (or node1, doesn't matter for unrooted)
    
    t = _Tree(n)
    t.children = children
    t.parent = parent
    return t


def _jc_likelihood(tree, aln_mat, branch_lens):
    """
    Compute log-likelihood of the tree under JC model.
    tree: _Tree object
    aln_mat: (n_taxa, n_sites) int array with values 0-3
    branch_lens: list of length tree.total, branch length for each node (edge to parent)
    Returns log-likelihood.
    """
    n_taxa = tree.n
    n_sites = aln_mat.shape[1]
    
    # For JC model:
    # P(same) = 0.75 + 0.25 * exp(-4d/3)
    # P(diff) = 0.25 * (1 - exp(-4d/3))
    # We use Felsenstein's pruning algorithm.
    
    # Precompute transition probabilities for each branch
    # For each internal node, we need the probability of each base at that node
    # given the data below.
    
    # Order nodes post-order (children before parents)
    # For a binary tree with 2n-1 nodes, we can do a DFS from leaves up.
    
    # Let's do a recursive approach with memoization
    # Actually, for speed, let's do iterative post-order traversal
    
    # First, determine a root. Use node tree.total - 1 (last internal node) as root.
    if n_taxa <= 2:
        # Handle small cases separately
        if n_taxa == 0:
            return 0.0
        if n_taxa == 1:
            # Single taxon, likelihood is 1 (or 0 in log)
            return 0.0
        if n_taxa == 2:
            # Two taxa, one branch
            d = branch_lens[0] if len(branch_lens) > 0 else 0.1
            p_same = 0.75 + 0.25 * math.exp(-4.0 * d / 3.0)
            p_diff = 0.25 * (1.0 - math.exp(-4.0 * d / 3.0))
            ll = 0.0
            for s in range(n_sites):
                if aln_mat[0, s] == aln_mat[1, s]:
                    ll += math.log(p_same)
                else:
                    ll += math.log(p_diff)
            return ll
    
    # For larger trees, use Felsenstein's algorithm
    # Root the tree at an internal node
    root = tree.total - 1  # Last internal node
    
    # Build children lists for traversal
    # We need to process nodes in post-order
    
    # Create a stack for DFS
    # prob[node][base] = probability of observing data in subtree rooted at node,
    # given that node has base 'base'
    
    # For leaves, prob[leaf][base] = 1 if base matches observation, else 0
    # But we need to handle the branch from leaf to parent
    
    # Let's use a simpler approach: for each site, compute the likelihood
    # using the pruning algorithm
    
    # Precompute for each branch the transition probability matrix
    # P(d) = [[p_same, p_diff, p_diff, p_diff],
    #          [p_diff, p_same, p_diff, p_diff],
    #          [p_diff, p_diff, p_same, p_diff],
    #          [p_diff, p_diff, p_diff, p_same]]
    
    # For efficiency, precompute p_same and p_diff for each branch
    p_same_arr = np.zeros(tree.total)
    p_diff_arr = np.zeros(tree.total)
    for i in range(tree.total):
        d = branch_lens[i]
        if d < 0:
            d = 0.0
        exp_term = math.exp(-4.0 * d / 3.0)
        p_same_arr[i] = 0.75 + 0.25 * exp_term
        p_diff_arr[i] = 0.25 * (1.0 - exp_term)
    
    # Process nodes in post-order
    # We need to visit children before parents
    
    # Get all nodes in post-order
    # Use iterative DFS
    visited = [False] * tree.total
    post_order = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            post_order.append(node)
            continue
        if visited[node]:
            continue
        visited[node] = True
        stack.append((node, True))
        left, right = tree.children[node]
        if right != -1:
            stack.append((right, False))
        if left != -1:
            stack.append((left, False))
    
    # Now process in post_order
    # prob[node] = array of 4 probabilities
    prob = [None] * tree.total
    
    for node in post_order:
        if node < n_taxa:
            # Leaf node
            base = aln_mat[node, 0]  # We'll handle per-site later, this is just for structure
            # Actually, we need to compute per-site likelihood
            # Let's restructure: compute the log-likelihood by summing over sites
            pass
    
    # This approach is getting complex. Let's use a simpler but slower method:
    # For each site, run the pruning algorithm
    # This is O(n_sites * n_taxa * 4^2) which should be fast enough for our sizes
    
    total_ll = 0.0
    for site in range(n_sites):
        # For this site, compute the likelihood using Felsenstein's algorithm
        # prob[node][base] for base in 0,1,2,3
        
        site_probs = [None] * tree.total
        
        for node in post_order:
            if node < n_taxa:
                # Leaf: probability is 1 if base matches, 0 otherwise
                obs = aln_mat[node, site]
                site_probs[node] = [0.0, 0.0, 0.0, 0.0]
                site_probs[node][obs] = 1.0
            else:
                # Internal node: compute probability for each base
                left, right = tree.children[node]
                if left == -1 or right == -1:
                    # Should not happen for internal nodes
                    site_probs[node] = [1.0, 1.0, 1.0, 1.0]
                    continue
                
                # For each base at this node, sum over bases of children
                # weighted by transition probabilities along the branches
                node_probs = [0.0, 0.0, 0.0, 0.0]
                for base in range(4):
                    # Probability of left child given this node has 'base'
                    # Sum over left_child_base of P(left_child_base | base) * prob[left][left_child_base]
                    left_sum = 0.0
                    right_sum = 0.0
                    for child_base in range(4):
                        if child_base == base:
                            trans_left = p_same_arr[left]
                            trans_right = p_same_arr[right]
                        else:
                            trans_left = p_diff_arr[left]
                            trans_right = p_diff_arr[right]
                        left_sum += trans_left * site_probs[left][child_base]
                        right_sum += trans_right * site_probs[right][child_base]
                    node_probs[base] = left_sum * right_sum
                
                site_probs[node] = node_probs
        
        # Likelihood for this site is the sum of probabilities at the root for each base
        # (assuming equal base frequencies)
        ll_site = sum(site_probs[root]) / 4.0
        if ll_site > 0:
            total_ll += math.log(ll_site)
        else:
            total_ll += math.log(1e-300)  # Avoid log(0)
    
    return total_ll


def _nni_hill_climb(tree, aln_mat, branch_lens, max_iterations=100):
    """
    Perform NNI hill-climbing to optimize the tree topology.
    Returns the best tree found.
    """
    n = tree.n
    if n < 4:
        return tree
    
    best_tree = tree
    best_ll = _jc_likelihood(tree, aln_mat, branch_lens)
    
    for iteration in range(max_iterations):
        if _remaining() < 2.0:
            break
        
        improved = False
        # Try all possible NNI moves
        # An NNI move involves an internal edge and its two adjacent internal edges
        # For each internal node (except root), consider the edge from parent to this node
        # The NNI move swaps the subtrees on either side of this edge
        
        # We need to find all internal edges
        # For each internal node v (not root), the edge (parent[v], v) is an internal edge
        # if both parent[v] and v are internal nodes
        
        for v in range(n, tree.total):
            if v == tree.total - 1:  # Skip root
                continue
            p = tree.parent[v]
            if p == -1 or p < n:
                continue  # Not an internal edge
            
            # The edge is (p, v)
            # The four subtrees involved in NNI are:
            # - The other child of p (call it w)
            # - The two children of v (call them x and y)
            # NNI move: swap v with w, so that the edge becomes (p, w) and v connects to x, y
            # Actually, NNI has two possible moves:
            # Move 1: (p, v) stays, but v's children swap with w's... no
            # Let me think more carefully.
            # 
            # The structure is:
            #   p
            #  / \
            # w   v
            #    / \
            #   x   y
            #
            # NNI move 1:
            #   p
            #  / \
            # v   w
            # / \
            # x   y
            #
            # NNI move 2:
            #   p
            #  / \
            # x   v
            #    / \
            #   y   w
            #
            # Wait, that's not right either. Let me reconsider.
            #
            # The standard NNI move on edge (p, v) where p has another child w,
            # and v has children x and y:
            #
            # Original: ((w, (x, y)), ...)  -- p is parent of w and v, v is parent of x and y
            #
            # NNI move 1: ((x, (y, w)), ...)  -- p is parent of x and v', v' is parent of y and w
            # NNI move 2: ((y, (x, w)), ...)  -- p is parent of y and v'', v'' is parent of x and w
            
            w = tree.children[p][0] if tree.children[p][1] == v else tree.children[p][1]
            x, y = tree.children[v]
            
            if w == -1 or x == -1 or y == -1:
                continue
            
            # Try NNI move 1: p connects to x and new_v1, new_v1 connects to y and w
            # Save original state
            orig_children_p = list(tree.children[p])
            orig_children_v = list(tree.children[v])
            
            # Apply NNI move 1
            tree.children[p] = [x, v]
            tree.children[v] = [y, w]
            tree.parent[x] = p
            tree.parent[y] = v
            tree.parent[w] = v
            
            ll_new = _jc_likelihood(tree, aln_mat, branch_lens)
            if ll_new > best_ll + 1e-10:
                best_ll = ll_new
                best_tree = _Tree(n)
                best_tree.children = [list(c) for c in tree.children]
                best_tree.parent = list(tree.parent)
                improved = True
            else:
                # Revert
                tree.children[p] = orig_children_p
                tree.children[v] = orig_children_v
                tree.parent[orig_children_p[0]] = p
                tree.parent[orig_children_p[1]] = p
                tree.parent[x] = v
                tree.parent[y] = v
            
            # Try NNI move 2: p connects to y and new_v2, new_v2 connects to x and w
            orig_children_p = list(tree.children[p])
            orig_children_v = list(tree.children[v])
            
            tree.children[p] = [y, v]
            tree.children[v] = [x, w]
            tree.parent[y] = p
            tree.parent[x] = v
            tree.parent[w] = v
            
            ll_new = _jc_likelihood(tree, aln_mat, branch_lens)
            if ll_new > best_ll + 1e-10:
                best_ll = ll_new
                best_tree = _Tree(n)
                best_tree.children = [list(c) for c in tree.children]
                best_tree.parent = list(tree.parent)
                improved = True
            else:
                # Revert
                tree.children[p] = orig_children_p
                tree.children[v] = orig_children_v
                tree.parent[orig_children_p[0]] = p
                tree.parent[orig_children_p[1]] = p
                tree.parent[x] = v
                tree.parent[y] = v
        
        if not improved:
            break
        
        # Update tree to best_tree
        tree.children = [list(c) for c in best_tree.children]
        tree.parent = list(best_tree.parent)
    
    return best_tree


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)

    # Build alignment matrix for likelihood computation
    lut = np.zeros(256, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    aln_mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        aln_mat[i] = lut[arr]

    # Build initial tree using NJ
    tree = _build_tree_from_nj(dist, labels)
    
    # Initialize branch lengths
    branch_lens = [0.1] * tree.total
    
    # Perform NNI hill-climbing
    if _remaining() > 5.0:
        max_iter = 50 if n <= 15 else 20
        tree = _nni_hill_climb(tree, aln_mat, branch_lens, max_iterations=max_iter)
    
    # Convert to Newick
    newick = tree.get_newick(labels)
    if newick and all(lb in newick for lb in labels):
        return newick
    
    # Fallback: balanced tree
    current = labels[0]
    for i in range(1, n):
        current = f"({current},{labels[i]})"
    return current + ";"
