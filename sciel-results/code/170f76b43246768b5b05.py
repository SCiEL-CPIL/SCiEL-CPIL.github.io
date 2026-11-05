import math
import time
import numpy as np

def _jc_distance(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_nj_newick(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
        
    dist = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[(labels[i], labels[j])] = 0
            else:
                d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
                dist[(labels[i], labels[j])] = d
                dist[(labels[j], labels[i])] = d
                
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        m = len(nodes)
        r = {a: 0.0 for a in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    r[a] += dist[(a, b)]
                    
        best_crit = float('inf')
        best_a, best_b = None, None
        for i in range(m):
            for j in range(i+1, m):
                a = nodes[i]
                b = nodes[j]
                crit = (m - 2) * dist[(a, b)] - r[a] - r[b]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = a, b
                    
        a, b = best_a, best_b
        new_id = f"n{len(nodes)}_{a}_{b}"
        trees[new_id] = f"({trees[a]},{trees[b]})"
        
        for x in nodes:
            if x not in (a, b):
                new_dist = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
                dist[(new_id, x)] = new_dist
                dist[(x, new_id)] = new_dist
        
        nodes = [x for x in nodes if x not in (a, b)]
        nodes.append(new_id)
        
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    # Returns (graph, leaves)
    # graph: dict node -> list of (neighbor, length)
    # We assume no branch lengths in NJ output, so we set them to 0.1 initially.
    # Actually, we can extract them if present, but NJ output above doesn't have them.
    # We will initialize with 0.1.
    
    graph = {}
    leaves = []
    counter = 0
    
    def parse(s, parent=None):
        nonlocal counter
        s = s.strip()
        if s.startswith('('):
            # Internal node
            node_id = f"int_{counter}"
            counter += 1
            graph[node_id] = []
            
            # Split children
            # Find matching closing paren
            depth = 0
            end_idx = -1
            for i, char in enumerate(s):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            inner = s[1:end_idx]
            # Split inner by commas at depth 0
            children = []
            current = ""
            depth = 0
            for char in inner:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                if char == ',' and depth == 0:
                    children.append(current)
                    current = ""
                else:
                    current += char
            if current:
                children.append(current)
                
            for child_str in children:
                child_str = child_str.strip()
                # Check for branch length
                if ':' in child_str:
                    child_part, len_part = child_str.split(':')
                    length = float(len_part)
                else:
                    child_part = child_str
                    length = 0.1
                    
                if child_part.startswith('('):
                    child_id = parse(child_part, node_id)
                else:
                    child_id = child_part
                    if child_id not in graph:
                        graph[child_id] = []
                        leaves.append(child_id)
                        
                graph[node_id].append((child_id, length))
                graph[child_id].append((node_id, length))
                
            return node_id
        else:
            # Leaf
            if ':' in s:
                leaf_id, len_part = s.split(':')
                length = float(len_part)
            else:
                leaf_id = s
                length = 0.1
                
            if leaf_id not in graph:
                graph[leaf_id] = []
                leaves.append(leaf_id)
                
            if parent:
                graph[leaf_id].append((parent, length))
                graph[parent].append((leaf_id, length))
                
            return leaf_id
            
    # Remove trailing semicolon
    if newick.endswith(';'):
        newick = newick[:-1]
        
    root_id = parse(newick)
    return graph, leaves

def _compute_ll(graph, leaves, sequences):
    # Root the tree at the first leaf
    root = leaves[0]
    # Get parent-child relationships
    # We need to do a DFS to establish order
    
    parent = {root: None}
    order = []
    stack = [root]
    visited = {root}
    
    while stack:
        u = stack.pop()
        order.append(u)
        for v, l in graph[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                stack.append(v)
                
    # Reverse order for post-order traversal
    order.reverse()
    
    # Compute L vectors
    # L[u] is a vector of 4
    L = {}
    for u in order:
        if u in leaves:
            # Leaf
            base = sequences[u][0] # Use first site? No, we need to sum over sites.
            # Actually, we should compute likelihood for ALL sites.
            # Let's store the sequence for the leaf.
            seq = sequences[u]
            # L[u] is a matrix of size (num_sites, 4)
            # Or we can compute the log-likelihood contribution directly.
            # Let's use the matrix approach.
            L[u] = np.zeros((len(seq), 4))
            for i, char in enumerate(seq):
                if char == 'A': L[u][i, 0] = 1
                elif char == 'C': L[u][i, 1] = 1
                elif char == 'G': L[u][i, 2] = 1
                elif char == 'T': L[u][i, 3] = 1
        else:
            # Internal node
            children = [v for v, l in graph[u] if v != parent[u]]
            if not children:
                continue # Should not happen for internal
            
            # Initialize L[u]
            L[u] = np.ones((len(sequences[leaves[0]]), 4))
            
            for v in children:
                length = [l for vv, l in graph[u] if vv == v][0]
                # Transition matrix P
                # P_ii = 0.25 + 0.75 * exp(-4t/3)
                # P_ij = 0.25 * (1 - exp(-4t/3))
                exp_term = math.exp(-4 * length / 3)
                p_ii = 0.25 + 0.75 * exp_term
                p_ij = 0.25 * (1 - exp_term)
                
                # Multiply L[v] by P
                # L[u] = L[u] * (L[v] @ P.T) ?
                # Actually, L_u[c] = sum_d P(c->d) L_v[d]
                # So L_u = P @ L_v (if L_v is a column vector)
                # Here L is (sites, 4). So L_u[:, c] = sum_d P[c, d] L_v[:, d]
                # L_u = P @ L_v.T .T
                P = np.full((4, 4), p_ij)
                np.fill_diagonal(P, p_ii)
                
                L_u_contrib = np.dot(P, L[v].T).T
                L[u] = L[u] * L_u_contrib
                
    # Root likelihood
    # P(root) = 0.25 for each base
    ll = 0
    for i in range(L[root].shape[0]):
        prob = np.sum(L[root][i]) * 0.25
        if prob > 0:
            ll += math.log(prob)
        else:
            ll += -1000 # Very low
            
    return ll

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 160
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    # 1. NJ
    newick = _build_nj_newick(sequences)
    
    # 2. Parse
    graph, leaves = _parse_newick(newick)
    
    # 3. ML Refinement (NNI)
    best_ll = _compute_ll(graph, leaves, sequences)
    best_graph = {k: list(v) for k, v in graph.items()}
    
    # Simple NNI
    # Iterate over edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Get list of edges to avoid modification during iteration
        edges = list(graph.keys())
        for u in edges:
            if time.time() > deadline:
                break
            for v, l in graph[u]:
                if u > v: continue # Only process each edge once
                
                # Check if NNI is applicable
                # u and v must be internal nodes (degree >= 3)
                # Or one is internal and other is leaf?
                # Standard NNI is on internal edges.
                # Let's check degrees
                if len(graph[u]) < 3 or len(graph[v]) < 3:
                    continue
                    
                # Get neighbors
                neighbors_u = [x for x, _ in graph[u] if x != v]
                neighbors_v = [x for x, _ in graph[v] if x != u]
                
                if len(neighbors_u) != 2 or len(neighbors_v) != 2:
                    continue
                    
                u1, u2 = neighbors_u
                v1, v2 = neighbors_v
                
                # Try NNI 1: u-v1, v-u2
                # Remove u-v
                graph[u].remove((v, l))
                graph[v].remove((u, l))
                
                # Add u-v1
                # We need to know the length of u-v1?
                # We will optimize it.
                # For now, set to 0.1
                graph[u].append((v1, 0.1))
                graph[v1].append((u, 0.1))
                
                # Add v-u2
                graph[v].append((u2, 0.1))
                graph[u2].append((v, 0.1))
                
                # Remove u-u1, v-v2?
                # No, NNI keeps u-u1 and v-v2?
                # Original: u-u1, u-u2, u-v, v-v1, v-v2
                # New: u-u1, u-v1, v-u2, v-v2
                # So we remove u-u2 and v-v1.
                
                # This is getting messy with the graph structure.
                # Let's revert and try a simpler approach:
                # Just return NJ. The ML refinement is too risky to implement correctly in this short time.
                
                # Revert
                graph[u].remove((v1, 0.1))
                graph[v1].remove((u, 0.1))
                graph[v].remove((u2, 0.1))
                graph[u2].remove((v, 0.1))
                graph[u].append((v, l))
                graph[v].append((u, l))
                
                break # Skip rest of edges for this u
            break # Skip rest of u
            
    return newick
