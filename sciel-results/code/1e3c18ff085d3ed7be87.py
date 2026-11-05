import numpy as np
import time
import sys
import math

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    
    # Efficient pairwise mismatch count
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return "", []
    if n == 1:
        return labels[0] + ";", []
    if n == 2:
        return f"({labels[0]},{labels[1]});", []
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    # Store edges for later? No, we just need the topology.
    # We will build the tree structure in a way that allows NNI.
    # For now, just get the Newick and parse it back into a graph.
    
    while len(active) > 3:
        m = len(active)
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
        r = d_active.sum(axis=1)
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        newick = f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        newick = "(" + ",".join(trees[x] for x in active) + ");"
        
    return newick, []

def _parse_newick_to_graph(newick):
    """
    Parses Newick string into a graph representation.
    Returns:
    - nodes: list of nodes. Each node is a dict: {'type': 'leaf'/'internal', 'label': str, 'neighbors': [ids]}
    - n_nodes: number of nodes
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
        
    nodes = []
    
    def parse(s):
        s = s.strip()
        if not s:
            return None
        
        if s.startswith('('):
            # Internal node
            my_id = len(nodes)
            nodes.append({'type': 'internal', 'label': None, 'neighbors': []})
            
            # Find matching close paren
            depth = 0
            i = 0
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = s[1:i]
            # Split inner by commas at depth 0
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            for c in children:
                child_id = parse(c)
                if child_id is not None:
                    nodes[my_id]['neighbors'].append(child_id)
                    nodes[child_id]['neighbors'].append(my_id)
            
            return my_id
        else:
            # Leaf
            label = s.split(':')[0].strip()
            my_id = len(nodes)
            nodes.append({'type': 'leaf', 'label': label, 'neighbors': []})
            return my_id
            
    parse(newick)
    return nodes

def _graph_to_newick(nodes):
    """
    Converts graph representation back to Newick string.
    """
    n = len(nodes)
    if n == 0:
        return ""
    
    # Find a leaf to root at
    root = None
    for i, node in enumerate(nodes):
        if node['type'] == 'leaf':
            root = i
            break
    
    if root is None:
        root = 0
        
    def to_newick(node_id, parent_id):
        node = nodes[node_id]
        if node['type'] == 'leaf':
            return node['label']
        
        children = [c for c in node['neighbors'] if c != parent_id]
        if not children:
            return "" # Should not happen for internal node in a tree
        
        child_strs = []
        for c in children:
            child_str = to_newick(c, node_id)
            if child_str:
                child_strs.append(child_str)
        
        if len(child_strs) == 0:
            return ""
        elif len(child_strs) == 1:
            return "(" + child_strs[0] + ")"
        else:
            return "(" + ",".join(child_strs) + ")"
            
    res = to_newick(root, -1)
    return res + ";"

def _calculate_likelihood(nodes, sequences, branch_lengths):
    """
    Calculates the likelihood of the sequences given the tree and branch lengths.
    branch_lengths: dict mapping edge (u, v) to length. Since it's undirected, we need a consistent key.
    We can use sorted tuple (min(u,v), max(u,v)).
    """
    n_taxa = len(sequences)
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    n_sites = seqs.shape[1]
    
    # Map labels to node indices
    label_to_node = {}
    for i, node in enumerate(nodes):
        if node['type'] == 'leaf':
            label_to_node[node['label']] = i
            
    # Map node indices to sequence indices
    node_to_seq_idx = {}
    for i, label in enumerate(labels):
        node_to_seq_idx[label_to_node[label]] = i
        
    # Precompute transition matrices for each branch length
    # P(t) = [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]
    # a = 0.75 + 0.25 * exp(-4t/3)
    # b = 0.25 * (1 - exp(-4t/3))
    
    # We need to root the tree for the pruning algorithm.
    # Let's root at node 0 (or any internal node).
    # Find an internal node
    root = 0
    for i, node in enumerate(nodes):
        if node['type'] == 'internal':
            root = i
            break
            
    # Do a DFS to get parent-child relationships and order
    parent = [-1] * len(nodes)
    children = [[] for _ in range(len(nodes))]
    order = []
    stack = [root]
    parent[root] = -2 # Mark root
    
    while stack:
        u = stack.pop()
        order.append(u)
        for v in nodes[u]['neighbors']:
            if v != parent[u]:
                parent[v] = u
                children[u].append(v)
                stack.append(v)
                
    # Process nodes in post-order
    # L[u] is a vector of 4 likelihoods for states A, C, G, T
    L = [None] * len(nodes)
    
    # Precompute P matrices for each edge
    # Edge (u, v) where u is parent of v
    P_matrices = {}
    for u in range(len(nodes)):
        for v in children[u]:
            # Get branch length
            key = (min(u, v), max(u, v))
            t = branch_lengths.get(key, 0.1) # Default length
            exp_term = np.exp(-4.0 * t / 3.0)
            a = 0.75 + 0.25 * exp_term
            b = 0.25 * (1.0 - exp_term)
            P = np.array([
                [a, b, b, b],
                [b, a, b, b],
                [b, b, a, b],
                [b, b, b, a]
            ])
            P_matrices[(u, v)] = P
            
    # Initialize leaves
    for u in order:
        if nodes[u]['type'] == 'leaf':
            seq_idx = node_to_seq_idx[u]
            L[u] = np.zeros(4)
            # Map ACGT to indices 0,1,2,3
            for s in range(n_sites):
                # This is inefficient. We should vectorize over sites.
                pass
                
    # Vectorized Likelihood Calculation
    # L[u] will be a matrix of shape (n_sites, 4)
    L = [None] * len(nodes)
    
    for u in reversed(order):
        if nodes[u]['type'] == 'leaf':
            seq_idx = node_to_seq_idx[u]
            seq = seqs[seq_idx]
            # One-hot encode
            L[u] = np.zeros((n_sites, 4))
            L[u][np.arange(n_sites), [0,1,2,3][np.where((seq == 'A') | (seq == 'C') | (seq == 'G') | (seq == 'T'))]] 
            # This is wrong. We need to map characters to indices.
            char_map = {'A':0, 'C':1, 'G':2, 'T':3}
            indices = np.array([char_map[c] for c in seq])
            L[u][np.arange(n_sites), indices] = 1.0
        else:
            # Internal node
            # L[u] = product over children of (P_child @ L_child)
            # P_child is (4, 4), L_child is (n_sites, 4)
            # Result is (n_sites, 4)
            
            # Start with ones
            L_u = np.ones((n_sites, 4))
            
            for v in children[u]:
                P = P_matrices[(u, v)]
                # L_u = L_u * (P @ L[v])
                # P @ L[v] is (4, 4) @ (n_sites, 4) -> (4, 4) @ (4, n_sites) ? No.
                # We want for each site s and state s_u: sum_{s_v} P[s_u, s_v] * L[v][s, s_v]
                # This is (P @ L[v].T).T
                L_v = L[v] # (n_sites, 4)
                # P is (4, 4)
                # We want (n_sites, 4)
                # L_contrib = L_v @ P.T  -> (n_sites, 4) @ (4, 4) = (n_sites, 4)
                # Check: L_contrib[s, s_u] = sum_{s_v} L_v[s, s_v] * P.T[s_v, s_u] = sum_{s_v} L_v[s, s_v] * P[s_u, s_v]
                # Correct.
                L_contrib = L_v @ P.T
                L_u *= L_contrib
                
            L[u] = L_u
            
    # Total likelihood
    # Root is 'root'
    # L[root] is (n_sites, 4)
    # Likelihood for each site is sum over root states
    # L_site = L[root].sum(axis=1)
    # Total Likelihood = prod(L_site)
    # Log Likelihood = sum(log(L_site))
    
    L_site = L[root].sum(axis=1)
    # Avoid log(0)
    L_site = np.maximum(L_site, 1e-300)
    log_L = np.sum(np.log(L_site))
    
    return log_L

def _nni_refine(nodes, sequences, branch_lengths):
    """
    Performs NNI moves to refine the topology.
    """
    improved = True
    current_log_L = _calculate_likelihood(nodes, sequences, branch_lengths)
    
    while improved:
        improved = False
        # Find all internal edges
        internal_edges = []
        for u in range(len(nodes)):
            if nodes[u]['type'] == 'internal':
                for v in nodes[u]['neighbors']:
                    if v > u and nodes[v]['type'] == 'internal':
                        internal_edges.append((u, v))
                        
        for u, v in internal_edges:
            # Get neighbors of u and v
            neighbors_u = [n for n in nodes[u]['neighbors'] if n != v]
            neighbors_v = [n for n in nodes[v]['neighbors'] if n != u]
            
            if len(neighbors_u) != 2 or len(neighbors_v) != 2:
                continue
                
            a, b = neighbors_u
            c, d = neighbors_v
            
            # Current topology: ((A, B), (C, D))
            # NNI 1: ((A, C), (B, D))
            # NNI 2: ((A, D), (B, C))
            
            # We need to calculate likelihood for the new topologies.
            # Instead of modifying the graph and recalculating everything,
            # we can just create a new graph for each candidate and calculate likelihood.
            # This is O(n) per candidate, which is fine.
            
            candidates = [
                (a, c, b, d),
                (a, d, b, c)
            ]
            
            for a1, b1, c1, d1 in candidates:
                # Create a new graph
                new_nodes = [dict(node) for node in nodes]
                # Update neighbors
                # u was connected to a, b, v
                # v was connected to c, d, u
                # New:
                # u connected to a1, b1, v
                # v connected to c1, d1, u
                
                # Helper to update neighbors
                def update_neighbors(node_id, new_neighbors):
                    new_nodes[node_id]['neighbors'] = new_neighbors
                    
                update_neighbors(u, [a1, b1, v])
                update_neighbors(v, [c1, d1, u])
                
                # Update the neighbors of a1, b1, c1, d1 to point to u or v
                # a1's neighbor u is correct if a1 was u's neighbor
                # But we need to ensure the other side is updated.
                # a1 was connected to u. Now u is connected to a1.
                # But a1's neighbor list might still have u.
                # We just changed u's list. We need to change a1's list to point to u.
                # Actually, a1 is already connected to u in the original graph.
                # So a1's list contains u.
                # We just changed u's list to contain a1.
                # So the edge (u, a1) is consistent.
                # Same for b1, c1, d1.
                
                # Wait, if a1 == a, then u is already connected to a.
                # If a1 == c, then u was NOT connected to c.
                # So we need to update c's neighbor list to include u, and remove v.
                
                # This manual graph manipulation is error-prone.
                # Let's just use the likelihood function that takes a graph.
                # But we need to pass the new graph.
                
                # Let's try a different approach:
                # The likelihood calculation depends on the tree structure.
                # We can write a function that calculates likelihood given the "local" configuration.
                # But that's complex.
                
                # Let's stick to modifying the graph and calling _calculate_likelihood.
                # We need to be careful with the neighbor lists.
                
                # Reset neighbor lists for the 4 subtrees
                # a1's neighbor should be u
                # b1's neighbor should be u
                # c1's neighbor should be v
                # d1's neighbor should be v
                
                # In the original graph:
                # a's neighbor is u
                # b's neighbor is u
                # c's neighbor is v
                # d's neighbor is v
                
                # If a1 == a, no change needed for a.
                # If a1 == c, then c's neighbor must change from v to u.
                
                # Let's just rebuild the neighbor lists for u, v, a1, b1, c1, d1.
                
                # This is getting messy. Let's use a simpler representation for NNI.
                # Or just accept that it's a bit slow and do it carefully.
                
                # Let's try to just swap the neighbors in the `new_nodes` list.
                
                # Original:
                # u: [a, b, v]
                # v: [c, d, u]
                # a: [u, ...]
                # b: [u, ...]
                # c: [v, ...]
                # d: [v, ...]
                
                # New:
                # u: [a1, b1, v]
                # v: [c1, d1, u]
                # a1: [u, ...] (rest of a1's neighbors)
                # b1: [u, ...]
                # c1: [v, ...]
                # d1: [v, ...]
                
                # We need to update the neighbor lists of a1, b1, c1, d1 to replace the old parent with the new parent.
                
                # Let's do it.
                new_nodes[u]['neighbors'] = [a1, b1, v]
                new_nodes[v]['neighbors'] = [c1, d1, u]
                
                # Update a1
                old_parent_a1 = v if a1 in (c, d) else u
                new_nodes[a1]['neighbors'] = [n for n in new_nodes[a1]['neighbors'] if n != old_parent_a1]
                new_nodes[a1]['neighbors'].append(u)
                
                # Update b1
                old_parent_b1 = v if b1 in (c, d) else u
                new_nodes[b1]['neighbors'] = [n for n in new_nodes[b1]['neighbors'] if n != old_parent_b1]
                new_nodes[b1]['neighbors'].append(u)
                
                # Update c1
                old_parent_c1 = u if c1 in (a, b) else v
                new_nodes[c1]['neighbors'] = [n for n in new_nodes[c1]['neighbors'] if n != old_parent_c1]
                new_nodes[c1]['neighbors'].append(v)
                
                # Update d1
                old_parent_d1 = u if d1 in (a, b) else v
                new_nodes[d1]['neighbors'] = [n for n in new_nodes[d1]['neighbors'] if n != old_parent_d1]
                new_nodes[d1]['neighbors'].append(v)
                
                new_log_L = _calculate_likelihood(new_nodes, sequences, branch_lengths)
                
                if new_log_L > current_log_L:
                    current_log_L = new_log_L
                    nodes = new_nodes
                    improved = True
                    # If improved, we should restart the loop or continue?
                    # Continuing is fine, but the internal_edges list is now stale.
                    # Let's break and restart the while loop.
                    break
                    
            if improved:
                break
                
    return nodes, current_log_L

def _optimize_branch_lengths(nodes, sequences, max_iters=5):
    """
    Optimizes branch lengths for a fixed topology using coordinate descent.
    """
    # Initialize branch lengths with a small value
    branch_lengths = {}
    for u in range(len(nodes)):
        for v in nodes[u]['neighbors']:
            if u < v:
                branch_lengths[(u, v)] = 0.1
                
    current_log_L = _calculate_likelihood(nodes, sequences, branch_lengths)
    
    for _ in range(max_iters):
        improved = False
        for key in branch_lengths:
            u, v = key
            # Find the best length for this branch
            # Use ternary search on [0, 5]
            lo = 0.0
            hi = 5.0
            for _ in range(20):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                branch_lengths[key] = m1
                L1 = _calculate_likelihood(nodes, sequences, branch_lengths)
                branch_lengths[key] = m2
                L2 = _calculate_likelihood(nodes, sequences, branch_lengths)
                
                if L1 < L2:
                    lo = m1
                else:
                    hi = m2
                    
            best_len = (lo + hi) / 2.0
            branch_lengths[key] = best_len
            new_log_L = _calculate_likelihood(nodes, sequences, branch_lengths)
            
            if new_log_L > current_log_L:
                current_log_L = new_log_L
                improved = True
            else:
                # Revert? No, ternary search finds the max for that branch.
                # If it's worse, it means the previous length was better?
                # No, ternary search finds the local max.
                # So new_log_L should be >= current_log_L.
                pass
                
        if not improved:
            break
            
    return branch_lengths, current_log_L

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 5:
        return "(" + ",".join(labels) + ");"
        
    # NJ
    newick, _ = _neighbor_joining(d, labels)
    
    # Parse to graph
    nodes = _parse_newick_to_graph(newick)
    
    # Initialize branch lengths
    branch_lengths = {}
    for u in range(len(nodes)):
        for v in nodes[u]['neighbors']:
            if u < v:
                branch_lengths[(u, v)] = 0.1
                
    # NNI Refinement
    nodes, log_L = _nni_refine(nodes, sequences, branch_lengths)
    
    # Optimize Branch Lengths
    if time.time() - start_time < deadline - 10:
        branch_lengths, log_L = _optimize_branch_lengths(nodes, sequences)
        
    # Convert back to Newick
    final_newick = _graph_to_newick(nodes)
    
    return final_newick

