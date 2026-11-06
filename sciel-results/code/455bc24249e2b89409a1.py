import time
import random

def _build_adj_bitmasks(edges, n):
    """Build adjacency bitmasks from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_bitmasks(adj, n):
    """Check if graph (given by adjacency bitmasks) is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            # Common neighbors of i and j
            common_ij = adj[i] & adj[j]
            # Remove i and j from common
            common_ij &= ~(1 << i)
            common_ij &= ~(1 << j)
            
            # Check if any pair in common_ij is connected
            c = common_ij
            while c:
                lsb = c & -c
                k_idx = lsb.bit_length() - 1
                c ^= lsb
                
                others = common_ij & ~(1 << k_idx)
                if adj[k_idx] & others:
                    return False
    return True

def _max_triangle_free_induced_bitmasks(adj, n, masks_by_popcount):
    """
    Find the size of the largest triangle-free induced subgraph.
    """
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)): continue
            for k in range(j + 1, n):
                if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                    triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return n

    for k in range(n, -1, -1):
        for mask in masks_by_popcount[k]:
            valid = True
            for t in triangles:
                if (mask & t) == t:
                    valid = False
                    break
            if valid:
                return k
    return 0

def _get_paley_edges(q):
    """Get edges for Paley graph of order q."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def _random_k4_free_graph(n, seed=None):
    """Generate a random K4-free graph using greedy random edge addition."""
    if seed is not None:
        random.seed(seed)
    
    adj = [0] * n
    edges = []
    
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        creates_k4 = False
        c = common
        while c:
            lsb = c & -c
            k_idx = lsb.bit_length() - 1
            c ^= lsb
            
            others = common & ~(1 << k_idx)
            if adj[k_idx] & others:
                creates_k4 = True
                break
        
        if not creates_k4:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges, adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 18
    
    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        pop = bin(mask).count('1')
        masks_by_popcount[pop].append(mask)
        
    best_edges = None
    best_score = float('inf')
    
    edges_17 = _get_paley_edges(17)
    adj_17 = _build_adj_bitmasks(edges_17, n)
    if _is_k4_free_bitmasks(adj_17, n):
        score_17 = _max_triangle_free_induced_bitmasks(adj_17, n, masks_by_popcount)
        best_edges = edges_17
        best_score = score_17

    for attempt in range(50):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            edges = best_edges[:]
            adj = _build_adj_bitmasks(edges, n)
            score = best_score
        else:
            edges, adj = _random_k4_free_graph(n, seed=attempt)
            score = _max_triangle_free_induced_bitmasks(adj, n, masks_by_popcount)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            addable_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        
                        creates_k4 = False
                        c = common
                        while c:
                            lsb = c & -c
                            k_idx = lsb.bit_length() - 1
                            c ^= lsb
                            
                            others = common & ~(1 << k_idx)
                            if adj[k_idx] & others:
                                creates_k4 = True
                                break
                        
                        if not creates_k4:
                            addable_edges.append((u, v))
                            
            if not addable_edges:
                break
            
            best_new_score = score
            best_edge = None
            random.shuffle(addable_edges)
            
            for u, v in addable_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_score = _max_triangle_free_induced_bitmasks(adj, n, masks_by_popcount)
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_edge = (u, v)
                    
            if best_edge is not None:
                u, v = best_edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                score = best_new_score
                improved = True
                if score < best_score:
                    best_score = score
                    best_edges = edges[:]
                    
    if best_edges is None:
        return _get_paley_edges(17)
        
    return best_edges
