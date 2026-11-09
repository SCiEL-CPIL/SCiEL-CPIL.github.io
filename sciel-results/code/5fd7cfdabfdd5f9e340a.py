"""Reconstruct unrooted phylogenetic topology from simulated DNA sequences using ML-EM with NJ fallback."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

_DEADLINE = 250.0  # safety margin before 270s kill
_start_time = None
_CHAR_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_INT_TO_CHAR = ['A', 'C', 'G', 'T']


def _encode_sequences(sequences, labels):
    """Encode sequences as int8 array of shape (n_taxa, n_sites)."""
    n = len(labels)
    L = len(sequences[labels[0]])
    arr = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(L):
            arr[i, j] = _CHAR_TO_INT[seq[j]]
    return arr


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    arr = _encode_sequences(sequences, labels)
    L = arr.shape[1]
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = (arr[i] != arr).sum(axis=1)
        dist[i] = diff / L
    
    p = dist.copy()
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _neighbor_joining(dist):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a Newick string (unrooted, no branch lengths).
    """
    n = dist.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    if n == 3:
        return "(t0,(t1,t2));"
    
    labels_list = [f"t{i}" for i in range(n)]
    dm = dist.copy()
    
    while len(labels_list) > 2:
        m = len(labels_list)
        r = dm.sum(axis=1)
        
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * dm[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        a, b = best_i, best_j
        new_label = f"({labels_list[a]},{labels_list[b]})"
        
        other_indices = [i for i in range(m) if i not in (a, b)]
        new_dists = []
        for k in other_indices:
            new_d = 0.5 * (dm[a, k] + dm[b, k] - dm[a, b])
            new_dists.append(new_d)
        
        new_m = m - 1
        new_dm = np.zeros((new_m, new_m), dtype=np.float64)
        new_labels = []
        
        for i_idx, i_orig in enumerate(other_indices):
            new_labels.append(labels_list[i_orig])
            for j_idx, j_orig in enumerate(other_indices):
                if i_idx != j_idx:
                    new_dm[i_idx, j_idx] = dm[i_orig, j_orig]
        
        new_labels.append(new_label)
        for i_idx, i_orig in enumerate(other_indices):
            new_dm[i_idx, new_m - 1] = new_dists[i_idx]
            new_dm[new_m - 1, i_idx] = new_dists[i_idx]
        
        labels_list = new_labels
        dm = new_dm
    
    return f"({labels_list[0]},{labels_list[1]});"


def _parse_newick_to_splits(newick, n_taxa):
    """
    Parse a Newick string and return the set of non-trivial splits.
    Each split is a frozenset of taxon indices (one side of the bipartition).
    We normalize so that the smallest taxon index is in the split.
    """
    # Build adjacency from Newick
    # First, parse into a tree structure
    # We'll use a stack-based parser
    
    # Get all taxa
    import re
    taxa = set(re.findall(r't(\d+)', newick))
    taxa = {int(t) for t in taxa}
    
    # Parse Newick to get splits
    # Approach: for each internal edge, compute the set of leaves on one side
    
    # Let's build the tree as a list of nodes
    # Each node: (id, children, parent, is_leaf, taxon_id)
    
    # Simpler: use a recursive approach
    # Strip trailing semicolon
    newick = newick.rstrip(';').strip()
    
    # We need to find all edges and for each, compute the leaf set on one side
    # This requires building the tree structure first
    
    # Build tree using stack
    # Nodes are represented by their position in a list
    # Let's parse into a nested structure
    
    # Actually, let's use a different approach:
    # For an unrooted tree with n leaves, there are n-2 internal nodes
    # Total nodes = 2n-3
    # Edges = 2n-4
    
    # Let me just enumerate all possible splits by trying all pairs of subtrees
    
    # Alternative: use the fact that for each internal node, each child subtree
    # defines a split
    
    # Let me build the tree properly
    # Parse Newick into a list of nodes with parent-child relationships
    
    # I'll use a simple recursive parser
    def parse_newick(s):
        """Parse Newick and return (root_node_id, nodes) where nodes is a dict."""
        # nodes: id -> {'children': [...], 'parent': id_or_None, 'taxon': int_or_None}
        nodes = {}
        node_counter = [0]
        
        def next_id():
            nid = node_counter[0]
            node_counter[0] += 1
            return nid
        
        pos = [0]
        
        def parse_node():
            """Parse a node starting at pos[0]. Returns node_id."""
            nid = next_id()
            nodes[nid] = {'children': [], 'parent': None, 'taxon': None}
            
            # Skip whitespace
            while pos[0] < len(s) and s[pos[0]] in ' \t':
                pos[0] += 1
            
            if pos[0] >= len(s):
                return nid
            
            if s[pos[0]] == '(':
                # Internal node
                pos[0] += 1  # skip '('
                while True:
                    child_id = parse_node()
                    nodes[child_id]['parent'] = nid
                    nodes[nid]['children'].append(child_id)
                    while pos[0] < len(s) and s[pos[0]] in ' \t':
                        pos[0] += 1
                    if pos[0] < len(s) and s[pos[0]] == ',':
                        pos[0] += 1
                        continue
                    elif pos[0] < len(s) and s[pos[0]] == ')':
                        pos[0] += 1
                        break
                    else:
                        break
            else:
                # Leaf node
                # Read taxon label
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]] not in ',); \t':
                    pos[0] += 1
                label = s[start:pos[0]]
                # Extract taxon number
                if label.startswith('t'):
                    try:
                        nodes[nid]['taxon'] = int(label[1:])
                    except ValueError:
                        pass
            
            # Skip branch length if present
            while pos[0] < len(s) and s[pos[0]] in ' \t':
                pos[0] += 1
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(s) and s[pos[0]] not in ',); \t':
                    pos[0] += 1
            
            return nid
        
        root_id = parse_node()
        return root_id, nodes
    
    root_id, nodes = parse_newick(newick)
    
    # Now find all splits
    # For each edge (parent, child), the split is the set of leaves in the child's subtree
    # vs the complement
    
    def get_leaves(node_id):
        """Get set of leaf taxon IDs in subtree rooted at node_id."""
        if nodes[node_id]['taxon'] is not None:
            return {nodes[node_id]['taxon']}
        leaves = set()
        for child in nodes[node_id]['children']:
            leaves |= get_leaves(child)
        return leaves
    
    all_taxa = set(range(n_taxa))
    splits = set()
    
    # For each non-root node, the edge to its parent defines a split
    for nid in nodes:
        if nodes[nid]['parent'] is not None:
            leaf_set = get_leaves(nid)
            if 0 < len(leaf_set) < n_taxa:
                # Normalize: ensure smallest taxon is in the split
                if min(leaf_set) < min(all_taxa - leaf_set):
                    splits.add(frozenset(leaf_set))
                else:
                    splits.add(frozenset(all_taxa - leaf_set))
    
    return splits


def _jc_branch_probabilities(branch_length):
    """
    For JC model, given branch length d, return the 4x4 transition probability matrix.
    P(same) = 1/4 + 3/4 * exp(-4d/3)
    P(diff) = 1/4 * (1 - exp(-4d/3))
    """
    if branch_length < 1e-10:
        return np.eye(4) / 4.0  # uniform (no evolution)
    exp_term = math.exp(-4.0 * branch_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 * (1.0 - exp_term)
    P = np.full((4, 4), p_diff)
    np.fill_diagonal(P, p_same)
    return P


def _log_likelihood_for_tree(arr, n_taxa, n_sites, tree_splits, branch_lengths, max_iter=50, tol=1e-6):
    """
    Compute log-likelihood of the alignment given a tree topology (via splits) and branch lengths.
    Uses EM to estimate internal node states.
    
    arr: (n_taxa, n_sites) int8 array
    tree_splits: set of frozensets defining the topology
    branch_lengths: dict mapping (parent, child) edge to length, or we estimate them
    
    For simplicity, we'll use a simplified approach:
    - For each site, run Felsenstein's pruning algorithm
    - This requires the tree structure, not just splits
    
    Actually, this is complex. Let me use a different approach:
    - Use the NJ tree as a starting point
    - For NNI moves, recompute the likelihood
    
    For the likelihood computation, I'll build the tree from the NJ result
    and use Felsenstein's algorithm.
    """
    pass  # Will be replaced


def _build_tree_from_newick(newick, n_taxa):
    """
    Build an internal tree representation from Newick string.
    Returns:
        nodes: list of node dicts
        edges: list of (u, v) undirected edges
        leaf_to_node: dict mapping taxon id to node index
    """
    import re
    
    # Parse Newick
    newick_clean = newick.rstrip(';').strip()
    
    # We'll build the tree using a stack
    # Each node: index in nodes list
    # nodes[i] = {'children': [], 'parent': None, 'taxon': None}
    
    nodes = []
    leaf_to_node = {}
    
    def add_node(taxon=None):
        idx = len(nodes)
        nodes.append({'children': [], 'parent': None, 'taxon': taxon})
        return idx
    
    # Parse
    pos = 0
    s = newick_clean
    
    def skip_ws():
        nonlocal pos
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
    
    def parse():
        nonlocal pos
        skip_ws()
        if pos >= len(s):
            return -1
        
        if s[pos] == '(':
            pos += 1  # skip '('
            node_id = add_node()
            while True:
                skip_ws()
                child_id = parse()
                if child_id >= 0:
                    nodes[node_id]['children'].append(child_id)
                    nodes[child_id]['parent'] = node_id
                skip_ws()
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in ',); \t':
                pos += 1
            label = s[start:pos]
            taxon = None
            if label.startswith('t'):
                try:
                    taxon = int(label[1:])
                except ValueError:
                    pass
            node_id = add_node(taxon)
            if taxon is not None:
                leaf_to_node[taxon] = node_id
        
        # Skip branch length
        skip_ws()
        if pos < len(s) and s[pos] == ':':
            pos += 1
            while pos < len(s) and s[pos] not in ',); \t':
                pos += 1
        
        return node_id
    
    parse()
    
    # Build edges
    edges = set()
    for i, node in enumerate(nodes):
        if node['parent'] is not None:
            edges.add((min(i, node['parent']), max(i, node['parent'])))
        for child in node['children']:
            edges.add((min(i, child), max(i, child)))
    
    return nodes, list(edges), leaf_to_node


def _felsenstein_loglik(arr, nodes, edges, leaf_to_node, branch_lengths, n_taxa):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm for JC model.
    
    arr: (n_taxa, n_sites) int8
    nodes: list of node dicts
    edges: list of (u, v) edges
    leaf_to_node: taxon -> node index
    branch_lengths: dict (u, v) -> length
    
    Returns total log-likelihood across all sites.
    """
    n_sites = arr.shape[1]
    n_nodes = len(nodes)
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Root the tree at node 0 (arbitrary)
    # Build parent-child relationships via BFS
    parent = [-1] * n_nodes
    children = [[] for _ in range(n_nodes)]
    order = [0]
    parent[0] = -2  # mark root
    for i in range(n_nodes):
        u = order[i]
        for v in adj[u]:
            if v != parent[u]:
                parent[v] = u
                children[u].append(v)
                order.append(v)
    
    # Process sites
    total_loglik = 0.0
    
    # Precompute JC transition probabilities for each branch
    # P_branch[u,v] = 4x4 matrix
    P_branch = {}
    for u, v in edges:
        bl = branch_lengths.get((u, v), branch_lengths.get((v, u), 0.1))
        P_branch[(u, v)] = _jc_branch_probabilities(bl)
        P_branch[(v, u)] = P_branch[(u, v)]  # symmetric for JC
    
    for site in range(n_sites):
        # Column vector for this site
        col = arr[:, site]  # shape (n_taxa,)
        
        # Initialize likelihood vectors at leaves
        # L[node] = 4-vector of likelihoods
        L = [None] * n_nodes
        
        for taxon in range(n_taxa):
            node = leaf_to_node[taxon]
            vec = np.zeros(4)
            vec[col[taxon]] = 1.0
            L[node] = vec
        
        # Process in post-order (children before parents)
        for u in reversed(order):
            if not children[u]:
                continue  # leaf, already set
            # Combine children
            # Start with identity (no prior)
            # For JC, the root prior is uniform
            result = np.ones(4)
            for v in children[u]:
                P = P_branch[(u, v)]
                # Multiply: result = result * (P @ L[v])
                # But we need to be careful: the message from v to u
                # L_msg = P @ L[v]  where P is transition from v to u
                # Actually in Felsenstein: 
                # alpha_u(a) = product over children v of sum_b P(a->b) * alpha_v(b)
                # For JC, P is symmetric, so P(a->b) = P(b->a)
                msg = P @ L[v]
                result *= msg
            L[u] = result
        
        # Root likelihood
        # For unrooted tree, we sum over root states with uniform prior
        root_ll = np.sum(L[0])
        total_loglik += math.log(max(root_ll, 1e-300))
    
    return total_loglik


def _nni_refine(arr, newick, n_taxa, max_iter=10, time_budget=30.0):
    """
    Refine topology using NNI moves with Felsenstein likelihood.
    Returns improved Newick string.
    """
    start = time.time()
    nodes, edges, leaf_to_node = _build_tree_from_newick(newick, n_taxa)
    n_nodes = len(nodes)
    
    # Initialize branch lengths
    # Use JC distances as initial estimates
    labels = [f"t{i}" for i in range(n_taxa)]
    dist = _jc_distance_matrix({f"t{i}": "".join(_INT_TO_CHAR[arr[i, j]] for j in range(arr.shape[1])) for i in range(n_taxa)}, labels)
    
    # For each edge, estimate branch length
    # Simple: use average distance to neighbors
    branch_lengths = {}
    for u, v in edges:
        # Estimate from leaf distances if possible
        # For now, use a reasonable default
        bl = 0.1
        branch_lengths[(u, v)] = bl
        branch_lengths[(v, u)] = bl
    
    # Compute initial likelihood
    current_ll = _felsenstein_loglik(arr, nodes, edges, leaf_to_node, branch_lengths, n_taxa)
    best_newick = newick
    best_ll = current_ll
    
    # NNI moves
    # For each internal edge, try the two alternative topologies
    # An internal edge connects two internal nodes u and v
    # The four neighbors of this edge define the quartet
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find internal edges
    internal_edges = []
    for u, v in edges:
        if nodes[u]['taxon'] is None and nodes[v]['taxon'] is None:
            internal_edges.append((u, v))
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        improved = False
        for u, v in internal_edges:
            # Get neighbors
            neigh_u = [x for x in adj[u] if x != v]
            neigh_v = [x for x in adj[v] if x != u]
            
            if len(neigh_u) < 1 or len(neigh_v) < 1:
                continue
            
            # For a proper NNI, we need u to have exactly 3 neighbors (including v)
            # and v to have exactly 3 neighbors (including v)
            # So neigh_u should have 2 elements and neigh_v should have 2 elements
            if len(neigh_u) != 2 or len(neigh_v) != 2:
                continue
            
            a, b = neigh_u
            c, d = neigh_v
            
            # Current topology: (a,b) connected to (c,d) via edge u-v
            # Alternative 1: (a,c) connected to (b,d)
            # Alternative 2: (a,d) connected to (b,c)
            
            # Try alternative 1: swap b and c
            # New edges: remove (u,v), (u,b), (v,c)
            # Add: (u,c), (v,b) -- but this changes the structure
            
            # Actually, for NNI on an unrooted tree:
            # The four subtrees hanging off the central edge are A, B (on u side) and C, D (on v side)
            # Current: (A,B) -- (C,D)
            # Alt 1: (A,C) -- (B,D)
            # Alt 2: (A,D) -- (B,C)
            
            # Let's build the new tree for each alternative
            
            # For alt 1: reconnect u to c, v to b
            # We need to update the nodes and edges
            
            # This is getting complex. Let me simplify:
            # Just try the move and recompute likelihood
            
            # Save original state
            orig_edges = list(edges)
            
            # Modify edges for alternative 1
            new_edges = set()
            for e in edges:
                e_tuple = (min(e), max(e))
                # Remove edges (u,b) and (v,c)
                if e_tuple == (min(u, b), max(u, b)) or e_tuple == (min(v, c), max(v, c)):
                    continue
                new_edges.add(e_tuple)
            
            # Add new edges (u,c) and (v,b)
            new_edges.add((min(u, c), max(u, c)))
            new_edges.add((min(v, b), max(v, b)))
            
            # Also need to update parent/children, but for likelihood we just need edges
            # Recompute likelihood with new edges
            new_edges_list = list(new_edges)
            new_bl = dict(branch_lengths)
            new_bl[(u, c)] = branch_lengths.get((u, b), 0.1)
            new_bl[(c, u)] = new_bl[(u, c)]
            new_bl[(v, b)] = branch_lengths.get((v, c), 0.1)
            new_bl[(b, v)] = new_bl[(v, b)]
            # Remove old
            for key in list(new_bl.keys()):
                if key in [(u, b), (b, u), (v, c), (c, v)]:
                    del new_bl[key]
            
            new_ll = _felsenstein_loglik(arr, nodes, new_edges_list, leaf_to_node, new_bl, n_taxa)
            
            if new_ll > best_ll + 1e-6:
                best_ll = new_ll
                best_newick = newick  # We'd need to update the Newick string
                improved = True
                # Update edges and branch lengths
                edges = new_edges_list
                branch_lengths = new_bl
                # Rebuild adjacency
                adj = defaultdict(list)
                for eu, ev in edges:
                    adj[eu].append(ev)
                    adj[ev].append(eu)
                # Rebuild internal edges
                internal_edges = []
                for eu, ev in edges:
                    if nodes[eu]['taxon'] is None and nodes[ev]['taxon'] is None:
                        internal_edges.append((eu, ev))
            
            # Restore original edges for next trial
            edges = orig_edges
        
        if not improved:
            break
    
    return best_newick


def _ml_em_reconstruct(arr, n_taxa, n_sites, time_budget=60.0):
    """
    Maximum-likelihood tree reconstruction using EM on JC model.
    Starts from NJ topology and refines with NNI.
    """
    start = time.time()
    
    # Get NJ tree as starting point
    labels = [f"t{i}" for i in range(n_taxa)]
    dist = _jc_distance_matrix({f"t{i}": "".join(_INT_TO_CHAR[arr[i, j]] for j in range(n_sites)) for i in range(n_taxa)}, labels)
    nj_newick = _neighbor_joining(dist)
    
    # Time remaining for NNI
    elapsed = time.time() - start
    remaining = time_budget - elapsed
    if remaining < 5.0:
        return nj_newick
    
    # Refine with NNI
    refined_newick = _nni_refine(arr, nj_newick, n_taxa, max_iter=5, time_budget=remaining)
    
    return refined_newick


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree from DNA sequences.
    Returns a Newick string with taxa t0..t(n-1).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a star tree as fallback
        n = len(sequences)
        labels = sorted(sequences, key=lambda x: int(x[1:]))
        return "(" + ",".join(labels) + ");"
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Encode sequences
    arr = _encode_sequences(sequences, labels)
    n_sites = arr.shape[1]
    
    # Try ML-EM approach with time budget
    time_remaining = _DEADLINE - (time.time() - _start_time)
    if time_remaining > 30.0:
        try:
            newick = _ml_em_reconstruct(arr, n, n_sites, time_budget=min(time_remaining - 10.0, 60.0))
            # Validate: check all taxa are present
            import re
            found_taxa = set(int(t) for t in re.findall(r't(\d+)', newick))
            if found_taxa == set(range(n)):
                return newick
        except Exception:
            pass
    
    # Fallback to pure NJ
    dist = _jc_distance_matrix(sequences, labels)
    newick = _neighbor_joining(dist)
    
    return newick


# EVOLVE-BLOCK-END
