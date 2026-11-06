"""Phylogenetic topology recovery: MLE for small n, NJ for large n."""
import math
import time
import itertools
import numpy as np

# Deadline guard: 270s wall-clock
_DEADLINE = 265.0
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
    n = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            si = sequences[labels[i]]
            sj = sequences[labels[j]]
            diff = sum(1 for x, y in zip(si, sj) if x != y)
            p = diff / len(si)
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _nj_reconstruct(sequences):
    """Neighbour joining on Jukes-Cantor corrected distances."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return f"{labels[0]};"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    matrix = _compute_distance_matrix(sequences, labels)
    n = n_taxa
    trees = [labels[i] for i in range(n)]
    active = list(range(n))
    
    while n > 3:
        # Compute total distances
        total = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    total[i] += matrix[i, j]
        
        best = None
        best_pair = (-1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                criterion = (n - 2) * matrix[i, j] - total[i] - total[j]
                if best is None or criterion < best:
                    best = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined = f"({trees[i]},{trees[j]})"
        new_idx = n - 1
        
        # Update matrix
        rest = [x for x in range(n) if x != i and x != j]
        new_matrix = np.zeros((n - 1, n - 1))
        
        for ni, oi in enumerate(rest):
            for nj, oj in enumerate(rest):
                if ni != nj:
                    new_matrix[ni, nj] = matrix[oi, oj]
        
        for ni, oi in enumerate(rest):
            d_new = 0.5 * (matrix[oi, i] + matrix[oi, j] - matrix[i, j])
            new_matrix[ni, new_idx] = d_new
            new_matrix[new_idx, ni] = d_new
        
        trees[new_idx] = joined
        active = rest + [new_idx]
        n = n - 1
        matrix = new_matrix
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def _jc_transition_matrix(t):
    """Return the 4x4 JC transition matrix for branch length t."""
    exp_term = math.exp(-4.0 * t / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 * (1.0 - exp_term)
    mat = np.zeros((4, 4))
    for k in range(4):
        for j in range(4):
            mat[k, j] = p_same if k == j else p_diff
    return mat


def _felsenstein_loglik(tree_splits, sequences, labels, branch_lengths):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm.
    tree_splits: list of splits (frozensets) defining the topology
    branch_lengths: dict mapping edge identifiers to lengths
    """
    n = len(labels)
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build site data
    site_data = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        for pos, c in enumerate(sequences[label]):
            site_data[i, pos] = char_map[c]
    
    n_sites = site_data.shape[1]
    
    # Build tree structure from splits
    # This is complex - for now, use a simpler approach
    # We'll represent the tree as a rooted binary tree
    
    # For MLE, we need to know the tree structure (parent-child relationships)
    # This requires building the tree from the splits, which is non-trivial
    
    # Alternative: use a pre-built tree structure
    # For small n, we can enumerate topologies as explicit tree structures
    
    return 0.0  # Placeholder


def _generate_topologies(n):
    """
    Generate all unrooted binary tree topologies for n taxa.
    Returns a list of tree structures, where each tree is represented
    as a list of edges (pairs of node indices).
    Leaves are 0..n-1, internal nodes are n..(2n-4).
    """
    if n == 3:
        # One topology: triangle
        return [([(0, 3), (1, 3), (2, 3)])]
    
    if n == 4:
        # Three topologies
        topologies = []
        # (0,1|2,3): edges (0,4),(1,4),(4,5),(2,5),(3,5)
        topologies.append([(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)])
        # (0,2|1,3)
        topologies.append([(0, 4), (2, 4), (4, 5), (1, 5), (3, 5)])
        # (0,3|1,2)
        topologies.append([(0, 4), (3, 4), (4, 5), (1, 5), (2, 5)])
        return topologies
    
    if n == 5:
        # 15 topologies
        # Generate by taking each 4-leaf topology and inserting leaf 4 into each edge
        four_leaf = _generate_topologies(4)
        topologies = []
        seen = set()
        
        for tree in four_leaf:
            # tree is a list of 5 edges for a 4-leaf tree
            # Internal nodes are 4 and 5
            # Insert leaf 4 (which is now the 5th leaf, index 4) into each edge
            # Wait, we're going from 4 leaves to 5 leaves
            # The 4-leaf tree has leaves 0,1,2,3 and internal nodes 4,5
            # We want to add leaf 4 and create a new internal node 6
            # Total nodes: 5 leaves + 3 internal = 8 nodes
            
            for edge in tree:
                u, v = edge
                # Subdivide edge (u,v) with new internal node 6
                # New edges: (u,6), (6,v), (6,4)
                new_tree = []
                for e in tree:
                    if e == edge:
                        continue
                    else:
                        new_tree.append(e)
                new_tree.append((u, 6))
                new_tree.append((6, v))
                new_tree.append((6, 4))
                
                # Canonical form: sort edges and nodes
                # For deduplication, use a canonical representation
                # Sort all edges
                canonical = tuple(sorted(tuple(sorted(e)) for e in new_tree))
                if canonical not in seen:
                    seen.add(canonical)
                    topologies.append(new_tree)
        
        return topologies
    
    # For n >= 6, return empty (too many to enumerate efficiently)
    return []


def _root_tree(edges, n_leaves):
    """
    Root a tree given by its edges.
    Returns a rooted tree structure with parent pointers.
    """
    n_nodes = 2 * n_leaves - 2
    # Build adjacency
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a leaf to use as root reference
    # Root at a leaf (say, leaf 0)
    root = 0
    
    # BFS to find parent pointers
    parent = {root: -1}
    visited = {root}
    queue = [root]
    children = {i: [] for i in range(n_nodes)}
    
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)
    
    return parent, children, root


def _felsenstein_mle(edges, sequences, labels):
    """
    Compute MLE log-likelihood for a tree topology.
    edges: list of (u,v) pairs defining the unrooted tree
    Returns (log_likelihood, branch_lengths_dict) or None.
    """
    n_leaves = len(labels)
    n_nodes = 2 * n_leaves - 2
    
    parent, children, root = _root_tree(edges, n_leaves)
    
    # Build post-order traversal
    post_order = []
    visited = set()
    
    def _dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in children[node]:
            _dfs(child)
        post_order.append(node)
    
    _dfs(root)
    
    # Build edge list with parent->child orientation
    oriented_edges = []
    for node in range(n_nodes):
        for child in children[node]:
            oriented_edges.append((node, child))
    
    n_edges = len(oriented_edges)
    
    # Initialize branch lengths
    bl = [0.1] * n_edges
    
    # Site data
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    site_data = np.zeros((n_leaves, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        for pos, c in enumerate(sequences[label]):
            site_data[i, pos] = char_map[c]
    
    n_sites = site_data.shape[1]
    
    def _compute_loglik(bl_vals):
        """Compute log-likelihood for given branch lengths."""
        total_loglik = 0.0
        
        for site in range(n_sites):
            # alpha[node][state] for each node
            alpha = np.zeros((n_nodes, 4))
            
            for node in post_order:
                if not children[node]:
                    # Leaf
                    # Find which leaf this is
                    leaf_label = labels[node] if node < n_leaves else None
                    if leaf_label is None:
                        return -np.inf
                    obs = site_data[node, site]
                    alpha[node, obs] = 1.0
                else:
                    # Internal node
                    prod_vec = np.ones(4)
                    for child in children[node]:
                        # Find branch length
                        bl_idx = None
                        for e, (p, c) in enumerate(oriented_edges):
                            if p == node and c == child:
                                bl_idx = e
                                break
                        if bl_idx is None:
                            return -np.inf
                        
                        t = bl_vals[bl_idx]
                        exp_term = math.exp(-4.0 * t / 3.0)
                        p_same = 0.25 + 0.75 * exp_term
                        p_diff = 0.25 * (1.0 - exp_term)
                        
                        sum_alpha = alpha[child].sum()
                        for k in range(4):
                            val = p_diff * sum_alpha + 0.75 * exp_term * alpha[child, k]
                            if val > 0:
                                prod_vec[k] *= val
                            else:
                                prod_vec[k] = 0.0
                    
                    alpha[node] = prod_vec
            
            site_lik = alpha[root].sum()
            if site_lik <= 0:
                return -np.inf
            total_loglik += math.log(site_lik)
        
        return total_loglik
    
    # Optimize branch lengths with coordinate descent
    best_lik = _compute_loglik(bl)
    if np.isinf(best_lik):
        return None
    
    grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    for _ in range(3):
        for e in range(n_edges):
            best_local = bl[e]
            best_local_lik = best_lik
            for t in grid:
                bl[e] = t
                lik = _compute_loglik(bl)
                if not np.isinf(lik) and lik > best_local_lik:
                    best_local_lik = lik
                    best_local = t
            bl[e] = best_local
            if best_local_lik > best_lik:
                best_lik = best_local_lik
    
    if np.isinf(best_lik):
        return None
    
    return best_lik


def _topology_to_newick(edges, n_leaves):
    """Convert a tree topology (list of edges) to Newick string."""
    parent, children, root = _root_tree(edges, n_leaves)
    
    def _to_newick(node):
        if not children[node]:
            return str(node)
        child_strs = [_to_newick(c) for c in children[node]]
        return "(" + ",".join(child_strs) + ")"
    
    newick = _to_newick(root)
    return newick + ";"


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return f"{labels[0]};"
    elif n == 2:
        return f"({labels[0]},{labels[1]});"
    elif n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # For n <= 5, try exhaustive MLE search
    if n <= 5:
        topologies = _generate_topologies(n)
        
        if topologies:
            best_newick = None
            best_lik = -np.inf
            
            for edges in topologies:
                lik = _felsenstein_mle(edges, sequences, labels)
                if lik is not None and lik > best_lik:
                    best_lik = lik
                    best_newick = _topology_to_newick(edges, n)
            
            if best_newick:
                # Map numeric labels to actual labels
                # The Newick uses 0..n-1, need to replace with t0..t(n-1)
                for i in range(n - 1, -1, -1):
                    best_newick = best_newick.replace(f",{i}", f",{labels[i]}")
                    best_newick = best_newick.replace(f"({i},", f"({labels[i]},")
                    best_newick = best_newick.replace(f"({i})", f"({labels[i]})")
                return best_newick
    
    # For n > 5 or if MLE failed, use NJ
    return _nj_reconstruct(sequences)


if __name__ == "__main__":
    test_sequences = {
        "t0": "ACGTACGTACGT",
        "t1": "ACGTACGTACGT",
        "t2": "ACGTACGTACGT",
        "t3": "ACGTACGTACGT",
    }
    result = reconstruct(test_sequences)
    print(result)

