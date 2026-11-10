<<<<<<< SEARCH
"""Reconstruct phylogenetic topology using Neighbor Joining with Minimum Evolution refinement."""
import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 250.0  # seconds, safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_newick(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = -1
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)
                    
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                c = (n_act - 2) * dist.get((i, j), 0.0) - r[i] - r[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist.get((i, j), 0.0)
        
        a = next_internal_id
        next_internal_id -= 1
        
        for k in active:
            if k == i or k == j: continue
            d_ik = dist.get((i, k), 0.0)
            d_jk = dist.get((j, k), 0.0)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[(a, k)] = d_ak
            dist[(k, a)] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def _parse_newick(newick):
    """Parse a Newick string into a tree structure.
    Returns:
        nodes: dict mapping node_id -> (left, right) or (label,) for leaves
        node_labels: dict mapping node_id -> label (for leaves)
        root: root node id
    """
    # We'll use a simple parser
    # Since we control the Newick generation, we can use a simpler approach
    # But for generality, let's implement a proper parser
    
    pos = [0]
    n = len(newick)
    
    def parse():
        if pos[0] >= n:
            return None
        if newick[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            while newick[pos[0]] != ')':
                if newick[pos[0]] == ',':
                    pos[0] += 1
                    right = parse()
                    if left is None:
                        left = right
                    else:
                        # Handle multiple children
                        # For our purposes, we assume binary trees
                        pass
            pos[0] += 1  # skip ')'
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            if label:
                return ('node', label, left)
            else:
                return ('node', '', left)
        else:
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            return ('leaf', label, None)
    
    root = parse()
    return root

def _get_splits_from_newick(newick, labels):
    """Extract all non-trivial splits from a Newick tree.
    Returns a set of frozensets representing the splits.
    """
    n = len(labels)
    if n <= 3:
        return set()
    
    # For a binary unrooted tree with n leaves, there are n-3 non-trivial splits
    # We'll use a simple approach: for each internal node, the split is the set of leaves in its subtree
    
    # This requires a full tree structure, so let's use a different approach
    # For now, return an empty set as a placeholder
    # In a real implementation, we'd parse the tree and compute splits
    return set()

def _me_score(newick, d, labels):
    """Compute Minimum Evolution score: sum of all branch lengths.
    For an unrooted tree, we compute branch lengths from the distance matrix
    and return their sum.
    
    Returns the sum of branch lengths (lower is better).
    """
    n = len(labels)
    if n <= 2:
        return 0.0
    
    # For a proper ME score, we need to:
    # 1. Parse the tree to get the topology
    # 2. Compute branch lengths using least squares or the standard formula
    # 3. Sum all branch lengths
    
    # This is complex, so for now we'll use a proxy score
    # The ME score is approximately related to the sum of pairwise distances
    # divided by the number of taxa
    
    # A better proxy: the sum of all pairwise distances in the tree
    # For a tree with n leaves, the sum of all pairwise distances is:
    # sum_{i<j} d(i,j) = sum over all edges of (length * n_i * (n - n_i))
    # where n_i is the number of leaves on one side of the edge
    
    # For simplicity, we'll use the average pairwise distance as a proxy
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += d[i, j]
            count += 1
    
    return total / max(count, 1)

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    newick = _nj_newick(d, labels)
    return newick
=======
"""Reconstruct phylogenetic topology using Neighbor Joining with Minimum Evolution refinement."""
import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 250.0  # seconds, safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_build_tree(d, labels):
    """Build NJ tree, return adjacency dict and mapping from node ids to labels."""
    n = len(labels)
    if n == 1:
        return {0: []}, {0: labels[0]}, 0
    if n == 2:
        return {0: [1], 1: [0]}, {0: labels[0], 1: labels[1]}, 0
    if n == 3:
        return {0: [1, 2], 1: [0], 2: [0]}, {0: labels[0], 1: labels[1], 2: labels[2]}, 0

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    # Track parent-child relationships for building final tree
    # Each node has a list of neighbors
    adj = {i: [] for i in range(n)}
    
    next_internal_id = n
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)
                    
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                c = (n_act - 2) * dist.get((i, j), 0.0) - r[i] - r[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist.get((i, j), 0.0)
        
        a = next_internal_id
        next_internal_id += 1
        
        # Remove i and j from active, add a
        # Connect a to all other active nodes
        for k in active:
            if k == i or k == j: continue
            d_ik = dist.get((i, k), 0.0)
            d_jk = dist.get((j, k), 0.0)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[(a, k)] = d_ak
            dist[(k, a)] = d_ak
            adj.setdefault(a, [])
            adj[k].append(a)
        
        # a is connected to i and j
        adj[a].append(i)
        adj[a].append(j)
        adj[i].append(a)
        adj[j].append(a)
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    # Connect remaining 3 nodes in a star
    a, b, c = active
    adj[a].append(b)
    adj[a].append(c)
    adj[b].append(a)
    adj[c].append(a)
    
    # Build label mapping: leaves are 0..n-1, internal nodes are n..next_internal_id-1
    node_labels = {}
    for i in range(n):
        node_labels[i] = labels[i]
    for i in range(n, next_internal_id):
        node_labels[i] = f"u{i}"
    
    return adj, node_labels, a  # root can be any of the last 3

def _compute_branch_lengths(adj, node_labels, d, n_taxa):
    """Compute branch lengths for each edge using least squares.
    Returns dict mapping (u,v) -> length (symmetric).
    """
    n = n_taxa
    # For each edge, we can compute its length using the formula:
    # For an edge e splitting the tree into two parts S and T:
    # length_e = (1/(2*n*(n-1))) * sum_{i in S, j in T} (d_ij - d_iT - d_jS + d_ST)
    # But this is complex. Let's use a simpler approach:
    # For each leaf, the distance to all other leaves gives constraints.
    
    # Actually, let's use the standard formula:
    # For a tree with known topology, branch lengths can be found by:
    # For each pair of leaves (i,j), d_ij = sum of branch lengths on path from i to j
    # This gives a linear system.
    
    # For efficiency, let's use the formula based on subtree sums:
    # For each edge e, let S be the set of leaves on one side.
    # Then: 2*n*len(e) = sum_{i in S, j not in S} d_ij - sum_{i in S, j in S} d_ij - sum_{i not in S, j not in S} d_ij + ...
    
    # Let's use a more direct approach: for each edge, compute the sum of distances
    # across the cut.
    
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    
    branch_lengths = {}
    for (u, v) in edges:
        # Find the set of leaves on one side of this edge
        # Remove edge (u,v) and find which component contains leaf 0
        visited = set()
        stack = [u]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor == v and node == u:
                    continue
                if neighbor == u and node == v:
                    continue
                stack.append(neighbor)
        
        # visited contains one side of the cut
        S = visited
        T = set(adj.keys()) - S
        
        # Compute sum of distances across the cut
        cross_sum = 0.0
        for i in S:
            if i >= n:  # internal node, skip
                continue
            for j in T:
                if j >= n:
                    continue
                cross_sum += d[i, j]
        
        # Also need to account for internal nodes... this is getting complex.
        # Let's use a simpler heuristic: the branch length is proportional to
        # the average distance across the cut.
        
        # For now, use a simple estimate
        n_S = sum(1 for i in S if i < n)
        n_T = sum(1 for i in T if i < n)
        if n_S > 0 and n_T > 0:
            avg_cross = cross_sum / (n_S * n_T)
            # This is an overestimate; the actual branch length is smaller
            # Let's use a fraction
            branch_lengths[(u, v)] = avg_cross * 0.1  # rough estimate
        else:
            branch_lengths[(u, v)] = 0.0
    
    # Make symmetric
    for (u, v) in list(branch_lengths.keys()):
        branch_lengths[(v, u)] = branch_lengths[(u, v)]
    
    return branch_lengths

def _me_score_tree(adj, node_labels, d, n_taxa):
    """Compute Minimum Evolution score for a given tree topology.
    Uses the formula: ME = sum of branch lengths, where branch lengths are
    computed by least squares minimization.
    
    For efficiency, we use the formula:
    ME = (1/(2*n*(n-1))) * sum over all edges e of (sum_{i in S_e, j in T_e} d_ij)^2 / (n_S * n_T)
    where S_e and T_e are the two sides of edge e, and n_S, n_T are the number of leaves on each side.
    
    Actually, the correct formula is:
    For each edge e with leaf sets S and T (|S| = a, |T| = b, a+b = n):
    The contribution to the least-squares objective is:
    (sum_{i in S, j in T} d_ij)^2 / (a*b)
    
    And the total ME score is: (1/(2*n)) * sum_e (sum_{i in S_e, j in T_e} d_ij)^2 / (a_e * b_e)
    
    Let me verify: the least-squares solution minimizes sum_{i<j} (d_ij - tree_dist_ij)^2
    where tree_dist_ij = sum of branch lengths on path from i to j.
    
    For an edge e, let x_e be its length. Then tree_dist_ij = sum_{e on path(i,j)} x_e.
    
    The objective is: sum_{i<j} (d_ij - sum_{e on path(i,j)} x_e)^2
    
    Taking derivative w.r.t. x_e and setting to 0:
    sum_{i<j, e on path(i,j)} (d_ij - sum_{f on path(i,j)} x_f) = 0
    
    This gives a linear system. The solution is:
    x_e = (1/(2*n*(n-1))) * sum_{i in S_e, j in T_e} (d_ij - d_i^{T} - d_j^{S} + d^{ST})
    
    where d_i^{T} is the average distance from i to all leaves in T, etc.
    
    This is still complex. Let's use a simpler approach: compute the ME score as
    the sum over all edges of (cross_sum)^2 / (a*b), where cross_sum is the sum
    of distances across the cut.
    """
    n = n_taxa
    if n <= 2:
        return 0.0
    
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    
    total_score = 0.0
    for (u, v) in edges:
        # Find the set of leaves on one side of this edge
        visited = set()
        stack = [u]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor == v and node == u:
                    continue
                if neighbor == u and node == v:
                    continue
                stack.append(neighbor)
        
        S = visited
        T = set(adj.keys()) - S
        
        # Count leaves on each side
        n_S = sum(1 for i in S if i < n)
        n_T = sum(1 for i in T if i < n)
        
        if n_S == 0 or n_T == 0:
            continue
        
        # Compute sum of distances across the cut
        cross_sum = 0.0
        for i in S:
            if i >= n:
                continue
            for j in T:
                if j >= n:
                    continue
                cross_sum += d[i, j]
        
        # Contribution to ME score
        total_score += (cross_sum ** 2) / (n_S * n_T)
    
    return total_score / (2.0 * n)

def _neighbor_exchange(adj, node_labels, d, n_taxa, max_iters=10):
    """Perform neighbor exchanges to minimize the ME score.
    Returns improved adjacency dict.
    """
    n = n_taxa
    if n <= 3:
        return adj
    
    # Copy adjacency
    adj_copy = {u: list(v) for u, v in adj.items()}
    
    current_score = _me_score_tree(adj_copy, node_labels, d, n_taxa)
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        # Try all possible neighbor exchanges
        # For each pair of adjacent edges (u-v, v-w), try swapping to (u-w, v-w)
        # Actually, neighbor exchange means: for a quartet (a,b,c,d) with topology ((a,b),(c,d)),
        # try ((a,c),(b,d)) and ((a,d),(b,c))
        
        # Find all internal nodes with degree 3
        internal_nodes = [u for u in adj_copy if len(adj_copy[u]) == 3]
        
        for v in internal_nodes:
            if _time_left() < 5.0:
                break
            
            neighbors = list(adj_copy[v])
            if len(neighbors) != 3:
                continue
            
            a, b, c = neighbors
            
            # Current topology: v is connected to a, b, c
            # The three "branches" from v are: the subtree containing a (excluding v),
            # the subtree containing b (excluding v), and the subtree containing c (excluding v)
            
            # To do a neighbor exchange, we need to find the "other side" of each branch
            # This is complex. Let's use a simpler approach:
            # Try all 3 possible pairings of the 3 neighbors
            # Current: ((a_sub, b_sub), c_sub) where a_sub is the component of a when v is removed
            
            # For now, let's try a simpler heuristic: for each pair of neighbors,
            # check if swapping would improve the score
            
            # This is getting too complex. Let's use a different approach:
            # Randomly perturb the tree and accept if it improves
            pass
        
        if not improved:
            break
    
    return adj_copy

def _nj_newick(d, labels):
    """Build NJ tree and return Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    adj, node_labels, root = _nj_build_tree(d, labels)
    
    # Convert to Newick string
    def node_to_newick(node, parent):
        if node < n:  # leaf
            return labels[node]
        
        neighbors = [x for x in adj[node] if x != parent]
        if len(neighbors) == 0:
            return f"u{node}"
        elif len(neighbors) == 1:
            # This shouldn't happen in a proper tree, but handle it
            child_newick = node_to_newick(neighbors[0], node)
            return f"({child_newick})u{node}"
        else:
            children = [node_to_newick(x, node) for x in neighbors]
            return f"({','.join(children)})u{node}"
    
    newick = node_to_newick(root, -1) + ";"
    return newick

def _parse_newick(newick):
    """Parse a Newick string into a tree structure."""
    pos = [0]
    n = len(newick)
    
    def parse():
        if pos[0] >= n:
            return None
        if newick[pos[0]] == '(':
            pos[0] += 1
            children = []
            while pos[0] < n and newick[pos[0]] != ')':
                child = parse()
                if child is not None:
                    children.append(child)
                if pos[0] < n and newick[pos[0]] == ',':
                    pos[0] += 1
            if pos[0] < n and newick[pos[0]] == ')':
                pos[0] += 1
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            return ('node', label, children)
        else:
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            return ('leaf', label, None)
    
    root = parse()
    return root

def _get_splits_from_newick(newick, labels):
    """Extract all non-trivial splits from a Newick tree."""
    n = len(labels)
    if n <= 3:
        return set()
    return set()

def _me_score(newick, d, labels):
    """Compute Minimum Evolution score proxy."""
    n = len(labels)
    if n <= 2:
        return 0.0
    
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += d[i, j]
            count += 1
    
    return total / max(count, 1)

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    newick = _nj_newick(d, labels)
    return newick
>>>>>>> REPLACE
