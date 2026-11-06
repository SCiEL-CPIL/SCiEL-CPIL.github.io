"""Phylogenetic topology recovery via NJ with NNI refinement under JC-ML."""
import math
import time

# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        arr = np.zeros(len(s), dtype=np.int8)
        for i, c in enumerate(s):
            if c == 'A':
                arr[i] = 0
            elif c == 'C':
                arr[i] = 1
            elif c == 'G':
                arr[i] = 2
            else:
                arr[i] = 3
        seq_arrays.append(arr)

    L = len(sequences[labels[0]])
    mat = np.stack(seq_arrays)  # shape (n, L)

    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 32
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2)
        p = counts.astype(np.float64) / L
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc

    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick_simple(dist, labels):
    """
    Simpler NJ implementation using explicit node tracking.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]

    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        row_sum = {}
        for i in nodes:
            s = 0.0
            for j in nodes:
                if i != j:
                    s += D.get((i, j), 0.0)
            row_sum[i] = s

        best_crit = float('inf')
        best_a = None
        best_b = None
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                d_ij = D.get((i, j), 0.0)
                crit = (m - 2) * d_ij - row_sum[i] - row_sum[j]
                if crit < best_crit:
                    best_crit = crit
                    best_a = i
                    best_b = j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(tree[x] for x in nodes) + ")"

        new_id = next_id
        next_id += 1

        tree[new_id] = f"({tree[a]},{tree[b]})"

        for x in nodes:
            if x == a or x == b:
                continue
            d_ax = D.get((a, x), 0.0)
            d_bx = D.get((b, x), 0.0)
            d_ab = D.get((a, b), 0.0)
            new_d = 0.5 * (d_ax + d_bx - d_ab)
            D[(new_id, x)] = new_d
            D[(x, new_id)] = new_d

        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    a, b, c = nodes
    return f"({tree[a]},{tree[b]},{tree[c]})"


def _parse_newick(newick, labels):
    """
    Parse a Newick string into an adjacency-based tree representation.
    Returns:
        adj: dict mapping node_id -> list of (neighbor_id, branch_length)
        node_to_label: dict mapping node_id -> label (for leaves) or None (for internal)
        next_id: next available node id
    """
    # Strip trailing semicolon
    s = newick.rstrip(';').strip()
    
    adj = {}
    node_to_label = {}
    next_id = 0
    leaf_counter = 0
    
    # Tokenize
    i = 0
    stack = []  # stack of (node_id, state) where state tracks parsing
    
    def create_node(label=None):
        nonlocal next_id, leaf_counter
        nid = next_id
        next_id += 1
        adj[nid] = []
        if label is not None:
            node_to_label[nid] = label
            leaf_counter += 1
        else:
            node_to_label[nid] = None
        return nid
    
    # Manual recursive descent parser
    def parse(pos):
        """Parse starting at position pos. Returns (node_id, new_pos)."""
        nonlocal i, next_id, leaf_counter
        
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            # Internal node
            pos += 1
            node_id = create_node(None)
            children = []
            while pos < len(s) and s[pos] != ')':
                child_id, pos = parse(pos)
                if child_id is not None:
                    children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            # Optional branch length after closing paren
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            # Connect children
            for child in children:
                adj[node_id].append((child, 0.0))
                adj[child].append((node_id, 0.0))
            return node_id, pos
        else:
            # Leaf: read label
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ':', ';'):
                pos += 1
            label = s[start:pos]
            node_id = create_node(label)
            # Optional branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            return node_id, pos
    
    root_id, _ = parse(0)
    return adj, node_to_label, next_id, root_id


def _jc_likelihood(adj, node_to_label, sequences, labels):
    """
    Compute the JC maximum likelihood score (log-likelihood) for the given topology.
    Uses Felsenstein's pruning algorithm.
    adj: dict node -> list of (neighbor, branch_len)
    node_to_label: dict node -> label or None
    sequences: dict label -> DNA string
    labels: list of labels
    """
    # Build leaf data
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Get all nodes
    all_nodes = list(adj.keys())
    node_set = set(all_nodes)
    
    # Build parent-child relationships via DFS from an arbitrary root
    # For unrooted tree, pick an arbitrary node as root
    root = all_nodes[0]
    
    # Build rooted tree: parent map, children map
    parent = {}
    children = {}
    for node in all_nodes:
        children[node] = []
    
    # DFS
    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor, _ in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)
    
    # Process nodes in post-order
    # Get post-order traversal
    post_order = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            post_order.append(node)
        else:
            stack.append((node, True))
            for child in children[node]:
                stack.append((child, False))
    
    # Felsenstein's algorithm: compute likelihoods at each node
    # For each node and each nucleotide state, compute the probability of observing
    # the data in the subtree below that node given that node has that state.
    
    # Transition probability under JC: P(t) for time t
    # P(same) = 0.25 + 0.75 * exp(-4t/3)
    # P(diff) = 0.25 * (1 - exp(-4t/3))
    
    # For each node, store likelihood vector [P(A), P(C), P(G), P(T)]
    # indexed 0,1,2,3 for A,C,G,T
    
    def get_state_encoding(c):
        if c == 'A':
            return 0
        elif c == 'C':
            return 1
        elif c == 'G':
            return 2
        else:
            return 3
    
    # Precompute transition matrices for unique branch lengths
    branch_lengths = set()
    for node in all_nodes:
        for neighbor, blen in adj[node]:
            branch_lengths.add(blen)
    
    trans_cache = {}
    for blen in branch_lengths:
        if blen < 1e-10:
            trans_cache[blen] = np.array([[1.0, 0, 0, 0],
                                          [0, 1.0, 0, 0],
                                          [0, 0, 1.0, 0],
                                          [0, 0, 0, 1.0]])
        else:
            p_same = 0.25 + 0.75 * math.exp(-4.0 * blen / 3.0)
            p_diff = (1.0 - p_same) / 3.0
            trans_cache[blen] = np.array([
                [p_same, p_diff, p_diff, p_diff],
                [p_diff, p_same, p_diff, p_diff],
                [p_diff, p_diff, p_same, p_diff],
                [p_diff, p_diff, p_diff, p_same]
            ])
    
    # Likelihood vectors
    liks = {}
    
    for node in post_order:
        if node in node_to_label and node_to_label[node] is not None:
            # Leaf node
            label = node_to_label[node]
            seq = sequences[label]
            # For each site, the likelihood is 1 for the observed state, 0 for others
            # We'll accumulate log-likelihoods
            # Actually, let's work with log-likelihoods to avoid underflow
            # For a leaf, the "likelihood vector" at each site is a one-hot
            # But we need to handle multiple sites. Let's store per-site or use log-sum.
            # Simpler: store the log-likelihood contribution directly.
            # For a leaf with observed state s at site k, the likelihood is 1 if state=s, 0 otherwise.
            # In log space: log(1) = 0 for the observed state, -inf for others.
            # This is awkward. Let's use a different approach.
            
            # Alternative: store likelihood vectors as products over sites.
            # For a leaf, at each site, the vector is one-hot. The product over sites
            # gives a vector where each component is the product of the one-hot entries.
            # Only the component matching all observed states would be non-zero, but since
            # different sites may have different observed states, this doesn't work simply.
            
            # Correct approach: for each site independently, compute the likelihood,
            # then sum log-likelihoods over sites.
            # Let's restructure: compute per-site log-likelihoods.
            
            pass
        
        # We'll handle this differently - compute per-site log-likelihoods
    
    # Let me redo this with per-site computation
    # For efficiency, we'll compute the log-likelihood as sum over sites of log-likelihood at each site
    
    # For each site, run Felsenstein's algorithm
    # This is O(n * L * 4^2) which is fine for our sizes
    
    total_loglik = 0.0
    
    for site in range(L):
        # For this site, compute the likelihood at the root
        # Using Felsenstein's pruning
        
        # Post-order processing
        site_liks = {}  # node -> array of 4 floats (likelihoods for A,C,G,T)
        
        for node in post_order:
            if node in node_to_label and node_to_label[node] is not None:
                # Leaf
                label = node_to_label[node]
                observed = get_state_encoding(sequences[label][site])
                vec = np.zeros(4)
                vec[observed] = 1.0
                site_liks[node] = vec
            else:
                # Internal node
                vec = np.zeros(4)
                for state in range(4):
                    prod = 1.0
                    for child in children[node]:
                        # Get branch length to child
                        blen = 0.0
                        for neighbor, b in adj[node]:
                            if neighbor == child:
                                blen = b
                                break
                        T = trans_cache[blen]
                        # P(state at child | state at node) * L(child)
                        child_lik = site_liks[child]
                        for child_state in range(4):
                            prod *= T[state, child_state] * child_lik[child_state]
                    vec[state] = prod
                site_liks[node] = vec
        
        # Likelihood at root: sum over states (uniform prior)
        root_lik = np.sum(site_liks[root]) / 4.0
        if root_lik > 0:
            total_loglik += math.log(root_lik)
        else:
            total_loglik += -1e10  # Very low likelihood
    
    return total_loglik


def _nni_refine(newick, sequences, labels, deadline):
    """
    Perform NNI (Nearest Neighbor Interchange) refinement on the tree.
    Returns the improved Newick string.
    """
    adj, node_to_label, next_id, root = _parse_newick(newick, labels)
    
    # Compute initial likelihood
    current_ll = _jc_likelihood(adj, node_to_label, sequences, labels)
    
    # Build a set of edges for quick lookup
    # For NNI, we consider each internal edge and try the two alternative rearrangements
    
    # Identify internal edges: edges where both endpoints are internal nodes
    # In an unrooted tree with n leaves, there are n-3 internal edges
    
    # Get list of internal nodes
    internal_nodes = [n for n in adj.keys() if node_to_label[n] is None]
    
    # For each internal edge, find the NNI moves
    # An internal edge connects two internal nodes u and v
    # Each of u and v has two other neighbors (besides each other)
    # NNI: swap the connections so that u connects to one of v's other neighbors, etc.
    
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Re-parse to get fresh adjacency (in case we modified it)
        # Actually, let's work with the current adj
        # Find all internal edges
        internal_edge_set = set()
        for u in internal_nodes:
            for v, _ in adj[u]:
                if node_to_label[v] is None and (v, u) not in internal_edge_set and (u, v) not in internal_edge_set:
                    if u < v:
                        internal_edge_set.add((u, v))
        
        best_gain = 0.0
        best_move = None
        
        for (u, v) in internal_edge_set:
            # Get other neighbors of u and v
            u_neighbors = [x for x, _ in adj[u] if x != v]
            v_neighbors = [x for x, _ in adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors  # Two neighbors of u (besides v)
            c, d = v_neighbors  # Two neighbors of v (besides u)
            
            # Current configuration: u-v, with u connected to a,b and v connected to c,d
            # NNI alternative 1: remove u-v, add u-c and v-b (or u-d and v-a)
            # Actually, NNI on edge (u,v) where u has neighbors {a,b,v} and v has neighbors {c,d,u}:
            # The two alternatives are:
            # 1. Replace edge (u,v) with edges (u,c) and (v,b) - wait, that's not right
            # 
            # Let me think more carefully. In an unrooted tree, an internal edge (u,v) 
            # divides the tree into two parts. u is connected to a, b, and v. 
            # v is connected to c, d, and u.
            # 
            # NNI: remove edge (u,v) and add one of:
            # - (u,c) and (v,b) -> but this changes connectivity
            # 
            # Actually, the standard NNI: for internal edge between nodes u and v,
            # where u's other neighbors are a and b, and v's other neighbors are c and d,
            # the two NNI alternatives are:
            # 1. Replace (u,v) with (u,c) and (v,a)  [swap a and c]
            # 2. Replace (u,v) with (u,d) and (v,b)  [swap b and d]
            
            # Alternative 1: remove (u,v), add (u,c) and (v,a)
            # But we need to be careful: we're changing the topology
            # Let's create a modified copy
            
            # Save state
            adj_backup = {k: list(v2) for k, v2 in adj.items()}
            
            # Remove edge (u,v)
            adj[u] = [(x, bl) for x, bl in adj[u] if x != v]
            adj[v] = [(x, bl) for x, bl in adj[v] if x != u]
            
            # Add edges (u,c) and (v,a)
            adj[u].append((c, 0.1))  # temporary branch length
            adj[c].append((u, 0.1))
            adj[v].append((a, 0.1))
            adj[a].append((v, 0.1))
            
            ll1 = _jc_likelihood(adj, node_to_label, sequences, labels)
            gain1 = ll1 - current_ll
            
            # Restore
            adj.clear()
            adj.update(adj_backup)
            
            # Alternative 2: remove (u,v), add (u,d) and (v,b)
            adj_backup = {k: list(v2) for k, v2 in adj.items()}
            
            adj[u] = [(x, bl) for x, bl in adj[u] if x != v]
            adj[v] = [(x, bl) for x, bl in adj[v] if x != u]
            
            adj[u].append((d, 0.1))
            adj[d].append((u, 0.1))
            adj[v].append((b, 0.1))
            adj[b].append((v, 0.1))
            
            ll2 = _jc_likelihood(adj, node_to_label, sequences, labels)
            gain2 = ll2 - current_ll
            
            # Restore
            adj.clear()
            adj.update(adj_backup)
            
            if gain1 > best_gain:
                best_gain = gain1
                best_move = (u, v, a, b, c, d, 1)
            if gain2 > best_gain:
                best_gain = gain2
                best_move = (u, v, a, b, c, d, 2)
        
        if best_move is not None and best_gain > 1e-10:
            # Apply the best move
            u, v, a, b, c, d, move_type = best_move
            
            # Remove edge (u,v)
            adj[u] = [(x, bl) for x, bl in adj[u] if x != v]
            adj[v] = [(x, bl) for x, bl in adj[v] if x != u]
            
            if move_type == 1:
                # Add (u,c) and (v,a)
                adj[u].append((c, 0.1))
                adj[c].append((u, 0.1))
                adj[v].append((a, 0.1))
                adj[a].append((v, 0.1))
            else:
                # Add (u,d) and (v,b)
                adj[u].append((d, 0.1))
                adj[d].append((u, 0.1))
                adj[v].append((b, 0.1))
                adj[b].append((v, 0.1))
            
            current_ll += best_gain
            improved = True
    
    # Convert back to Newick
    return _adj_to_newick(adj, node_to_label, root)


def _adj_to_newick(adj, node_to_label, root):
    """Convert adjacency representation back to Newick string."""
    if len(adj) == 1:
        return node_to_label[root] or "t0"
    
    # Build parent-child via DFS
    parent = {}
    children = {}
    for node in adj:
        children[node] = []
    
    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor, _ in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)
    
    def build(node):
        if node in node_to_label and node_to_label[node] is not None:
            return node_to_label[node]
        if not children[node]:
            return node_to_label[node] or "t?"
        parts = [build(child) for child in children[node]]
        return "(" + ",".join(parts) + ")"
    
    result = build(root)
    # Ensure all leaves are present
    return result


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250  # Safety margin from 270s limit

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)

    # Check deadline
    if time.time() > _DEADLINE - 10:
        return "(" + ",".join(labels) + ")"

    # Use NJ for initial topology
    result = _nj_newick_simple(dist, labels)

    # NNI refinement if time permits
    if n <= 20 and time.time() < _DEADLINE - 30:
        try:
            refined = _nni_refine(result, sequences, labels, _DEADLINE - 5)
            if refined:
                result = refined
        except Exception:
            pass  # Keep NJ result if NNI fails

    return result

# EVOLVE-BLOCK-END
