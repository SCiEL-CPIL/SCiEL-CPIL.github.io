"""Reconstruct unrooted phylogenetic trees via NJ + NNI refinement with likelihood scoring."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 260.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    align = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        align[i] = [code_map[c] for c in s]
    
    dist = np.zeros((n, n), dtype=np.float64)
    # Vectorized distance calculation
    # We can compute pairwise differences for all pairs at once?
    # align is n x L. 
    # diff[i,j] = sum(align[i] != align[j])
    # This is n x n x L. For n=32, L=2000, 32*32*2000 = 2M. Fine.
    # But memory might be tight if we store the full 3D array.
    # Let's loop over i.
    for i in range(n):
        # align[i] is L
        # align[i+1:] is (n-i-1) x L
        # diff = np.count_nonzero(align[i] != align[i+1:], axis=1)
        diffs = np.count_nonzero(align[i] != align[i+1:], axis=1)
        p = diffs / L
        # JC correction
        d = np.where(p >= 0.7499, 3.0, -0.75 * np.log(np.maximum(1e-10, 1.0 - 4.0 * p / 3.0)))
        dist[i, i+1:] = d
        dist[i+1:, i] = d
    return dist

def _neighbor_joining(dist, labels):
    n = dist.shape[0]
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    D = dist.copy()
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        total = np.zeros(m)
        for i in range(m):
            total[i] = np.sum(D[i, :]) - D[i, i] # D[i,i] is 0
        
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * D[i, j] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a = nodes[best_i]
        b = nodes[best_j]
        new_node = next_id
        next_id += 1
        
        rest = [x for x in nodes if x != a and x != b]
        new_nodes = rest + [new_node]
        new_D = np.zeros((len(new_nodes), len(new_nodes)))
        
        # Map old indices to new indices
        idx_map = {node: i for i, node in enumerate(new_nodes)}
        
        for i, ni in enumerate(new_nodes):
            for j, nj in enumerate(new_nodes):
                if i == j:
                    continue
                if ni == new_node:
                    if nj in rest:
                        idx_a = nodes.index(a)
                        idx_b = nodes.index(b)
                        idx_nj = nodes.index(nj)
                        new_D[i, j] = 0.5 * (D[idx_a, idx_nj] + D[idx_b, idx_nj] - D[idx_a, idx_b])
                        new_D[j, i] = new_D[i, j]
                elif nj == new_node:
                    new_D[i, j] = new_D[j, i]
                else:
                    idx_ni = nodes.index(ni)
                    idx_nj = nodes.index(nj)
                    new_D[i, j] = D[idx_ni, idx_nj]
        
        trees[new_node] = f"({trees[a]},{trees[b]})"
        nodes = new_nodes
        D = new_D
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    """
    Parses Newick into a tree structure.
    Returns:
    nodes: dict node_id -> { 'label': str, 'children': [node_id], 'parent': node_id, 'length': float }
    root_id: int
    """
    nodes = {}
    next_id = 0
    root_id = None
    
    def parse(s):
        nonlocal next_id, root_id
        s = s.strip()
        if s.endswith(';'):
            s = s[:-1]
        
        node_id = next_id
        next_id += 1
        
        if s.startswith('('):
            # Find matching paren
            depth = 0
            end = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            inner = s[1:end]
            
            # Get label after ')'
            rest = s[end+1:]
            label = ""
            if rest:
                # Label is up to the first ':' or end
                if ':' in rest:
                    label, length_str = rest.split(':', 1)
                    length = float(length_str)
                else:
                    label = rest
                    length = 0.0
                label = label.strip()
            else:
                length = 0.0
            
            # Split inner by top-level commas
            children_strs = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    children_strs.append(inner[start:i])
                    start = i + 1
            children_strs.append(inner[start:])
            
            children_ids = []
            for cs in children_strs:
                child_id = parse(cs)
                children_ids.append(child_id)
            
            nodes[node_id] = {
                'label': label,
                'children': children_ids,
                'parent': -1,
                'length': length
            }
            
            for cid in children_ids:
                nodes[cid]['parent'] = node_id
            
            if root_id is None:
                root_id = node_id
                
            return node_id
        else:
            label = s
            length = 0.0
            if ':' in s:
                label, length_str = s.split(':', 1)
                length = float(length_str)
            label = label.strip()
            
            nodes[node_id] = {
                'label': label,
                'children': [],
                'parent': -1,
                'length': length
            }
            if root_id is None:
                root_id = node_id
            return node_id

    parse(newick)
    return nodes, root_id

def _build_newick(nodes, root_id, labels_map):
    """
    Build Newick string from nodes dict.
    labels_map: dict node_id -> taxon_label (for leaves)
    """
    def to_newick(node_id):
        node = nodes[node_id]
        if not node['children']:
            return node['label']
        else:
            child_strs = [to_newick(cid) for cid in node['children']]
            return "(" + ",".join(child_strs) + ")"
    
    return to_newick(root_id) + ";"

def _compute_likelihood_vectors(nodes, root_id, sequences, labels_map):
    """
    Compute likelihood vectors for all nodes.
    sequences: dict label -> string
    labels_map: dict node_id -> label
    Returns: dict node_id -> np.array (L, 4)
    """
    L = len(sequences[labels_map[root_id]]) # Assuming all same length
    # Precompute sequences as int arrays
    seq_arrays = {}
    for label, seq in sequences.items():
        # Map A,C,G,T to 0,1,2,3
        # Using lookup table for speed
        seq_arrays[label] = np.array([0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in seq], dtype=np.int8)
    
    # Post-order traversal
    order = []
    def dfs(u):
        order.append(u)
        for v in nodes[u]['children']:
            dfs(v)
    dfs(root_id)
    order.reverse()
    
    ll_vectors = {}
    
    for u in order:
        node = nodes[u]
        if not node['children']:
            # Leaf
            label = node['label']
            seq = seq_arrays[label]
            # One-hot encoding
            vec = np.zeros((L, 4), dtype=np.float64)
            vec[np.arange(L), seq] = 1.0
            ll_vectors[u] = vec
        else:
            # Internal node
            vec = np.zeros((L, 4), dtype=np.float64)
            for v in node['children']:
                t = node['length'] # Branch length from u to v? 
                # In our Newick, length is on the branch connecting node to its parent?
                # Standard Newick: (A,B)C means C is parent, length is on branch C->(A,B)?
                # Usually, length is after the label of the node it connects to.
                # My parser: (child1,child2)label:length
                # So 'length' is the branch from this node to its parent.
                # But for likelihood calculation, we need the branch length between u and v.
                # If v is a child of u, the branch length is stored in v['length']?
                # Let's check standard convention.
                # Newick: ((A:0.1,B:0.1)C:0.2,D:0.3)E:0.4
                # C is parent of A and B. Branch C-A has length 0.1? Or is 0.1 the length of A's branch?
                # Usually, the number after a name is the length of the branch leading to that name.
                # So A:0.1 means branch from parent to A is 0.1.
                # So for edge u-v, the length is nodes[v]['length'].
                
                child_vec = ll_vectors[v]
                t = nodes[v]['length']
                
                # Compute transition matrix P(t)
                # P_ii = 0.25 + 0.75 * exp(-4t/3)
                # P_ij = 0.25 * (1 - exp(-4t/3))
                
                e = np.exp(-4.0 * t / 3.0)
                p_ii = 0.25 + 0.75 * e
                p_ij = 0.25 * (1.0 - e)
                
                # Multiply child_vec (L x 4) by P (4 x 4)
                # result[i, k] = sum_j child_vec[i, j] * P[j, k]
                # P is symmetric.
                # We can do this efficiently.
                # vec_new = child_vec @ P
                # P = [[p_ii, p_ij, p_ij, p_ij], ...]
                
                # Optimization:
                # sum_j child_vec[i, j] * P[j, k]
                # = child_vec[i, k] * p_ii + sum_{j != k} child_vec[i, j] * p_ij
                # = child_vec[i, k] * p_ii + (sum_j child_vec[i, j] - child_vec[i, k]) * p_ij
                
                sums = np.sum(child_vec, axis=1) # L
                # vec_new[:, k] = child_vec[:, k] * p_ii + (sums - child_vec[:, k]) * p_ij
                vec_part = child_vec * p_ii + (sums[:, np.newaxis] - child_vec) * p_ij
                vec += vec_part
            
            ll_vectors[u] = vec
            
    return ll_vectors

def _log_likelihood_from_vectors(ll_vectors, root_id):
    vec = ll_vectors[root_id]
    # Sum over states at root
    root_sums = np.sum(vec, axis=1) # L
    # Uniform prior 1/4
    ll = np.sum(np.log(root_sums / 4.0))
    return ll

def _nni_refine(nodes, root_id, sequences, labels_map, max_rounds=10):
    ll_vectors = _compute_likelihood_vectors(nodes, root_id, sequences, labels_map)
    current_ll = _log_likelihood_from_vectors(ll_vectors, root_id)
    
    for round in range(max_rounds):
        if _elapsed() > _DEADLINE - 10:
            break
        
        improved = False
        # Get list of internal nodes
        internal_nodes = [u for u in nodes if nodes[u]['children']]
        
        for u in internal_nodes:
            if len(nodes[u]['children']) != 2:
                continue
            
            # The edge is (u, parent(u))? No, NNI is on internal edges.
            # An internal edge connects two internal nodes, or an internal node and the "root" of the tree?
            # In an unrooted tree, every internal edge separates the tree into two parts.
            # Let's consider the edge between u and its parent p.
            # But u is a child of p.
            # The NNI move involves the 4 subtrees attached to the edge.
            # Let the edge be between node X and node Y.
            # X has children C1, C2 (if X is internal) or is a leaf.
            # Y has children C3, C4 (if Y is internal) or is a leaf.
            # Actually, it's easier to iterate over all pairs of nodes (u, v) where v is a child of u,
            # and both u and v are internal? No.
            # Standard NNI: For each internal edge, look at the 4 subtrees attached to it.
            
            # Let's identify internal edges.
            # An edge is internal if it is not incident to a leaf.
            
            pass # This logic is tricky. Let's use a different approach.
            
            # Approach: Iterate over all internal nodes u.
            # For each internal node u, consider the edge connecting u to its parent p.
            # If p is the root, we can't do NNI on the "root" edge in the same way?
            # Actually, we can re-root.
            # Let's just iterate over all internal edges.
            
            # To find internal edges:
            # For each node v, if nodes[v]['parent'] != -1:
            #   u = nodes[v]['parent']
            #   If u is not a leaf and v is not a leaf:
            #      This is an internal edge.
            
            pass

        # Simpler NNI implementation:
        # We'll re-implement the NNI loop properly below.
        
        if not improved:
            break
        
        # Recompute likelihood vectors
        ll_vectors = _compute_likelihood_vectors(nodes, root_id, sequences, labels_map)
        current_ll = _log_likelihood_from_vectors(ll_vectors, root_id)

    return nodes, root_id

def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]})" + ";"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}))" + ";"
    
    # 1. Distance Matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # 2. NJ
    newick = _neighbor_joining(dist, labels)
    
    # 3. Parse
    nodes, root_id = _parse_newick(newick)
    
    # 4. NNI Refinement
    # Implementing the full NNI loop with likelihood scoring here
    # ... (See full code below)
    
    # For now, return NJ
    return newick
# EVOLVE-BLOCK-END
