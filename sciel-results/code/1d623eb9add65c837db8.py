import random
import time

def _popcount(x):
    return bin(x).count('1')

def _is_k4_free_bitmask(adj):
    """Check if graph with adjacency bitmasks is K4-free."""
    n = 18
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                m = common
                while m:
                    lsb = m & -m
                    a = lsb.bit_length() - 1
                    m ^= lsb
                    if adj[a] & common:
                        return False
    return True

def _max_triangle_free_bitmask(adj):
    """
    Find the maximum number of vertices inducing a triangle-free subgraph.
    Uses branch and bound with bitmasks.
    """
    n = 18
    best = [0]
    
    def branch(candidates, current_set, current_size):
        # Pruning
        if current_size + _popcount(candidates) <= best[0]:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick a vertex with high degree in the induced subgraph of candidates
        # This tends to prune faster
        v_idx = -1
        max_deg = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = _popcount(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = v
        
        v = v_idx
        v_bit = 1 << v
        rest_candidates = candidates & ~v_bit
        
        # Option 1: Exclude v
        branch(rest_candidates, current_set, current_size)
        
        # Option 2: Include v
        # Check if v can be added to current_set without forming a triangle
        N_v = adj[v] & current_set
        if N_v:
            # Check if N_v contains an edge
            m2 = N_v
            conflict = False
            while m2:
                lsb2 = m2 & -m2
                u = lsb2.bit_length() - 1
                m2 ^= lsb2
                if adj[u] & N_v:
                    conflict = True
                    break
            if conflict:
                return # Cannot add v
        
        # If no conflict, we can add v
        new_set = current_set | v_bit
        new_size = current_size + 1
        
        # Update candidates: remove any vertex w that would form a triangle with v and any u in current_set
        forbidden = 0
        m3 = N_v
        while m3:
            lsb3 = m3 & -m3
            u = lsb3.bit_length() - 1
            m3 ^= lsb3
            forbidden |= (adj[v] & adj[u])
        
        new_candidates = rest_candidates & ~forbidden
        
        branch(new_candidates, new_set, new_size)
        
    branch((1 << n) - 1, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Initial graph: Turan T(18,3)
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(n):
        for j in range(i + 1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_score = _max_triangle_free_bitmask(adj)
    best_adj = adj[:]
    
    # Local Search
    # Strategy: Add edges within parts to create more triangles, reducing max triangle-free subgraph.
    
    def get_edges_within_parts(adj):
        edges = []
        for p in range(3):
            for i in range(len(parts[p])):
                for j in range(i + 1, len(parts[p])):
                    u = parts[p][i]
                    v = parts[p][j]
                    if not (adj[u] & (1 << v)):
                        edges.append((u, v))
        return edges

    def get_edges_between_parts(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                part_i = 0 if i < 6 else (1 if i < 12 else 2)
                part_j = 0 if j < 6 else (1 if j < 12 else 2)
                if part_i != part_j and (adj[i] & (1 << j)):
                    edges.append((i, j))
        return edges

    attempts = 0
    while time.time() < deadline - 5:
        # Try adding edges within parts
        if random.random() < 0.7:
            # 70% chance to add an intra-part edge
            intra_edges = get_edges_within_parts(adj)
            if intra_edges:
                u, v = random.choice(intra_edges)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_bitmask(adj):
                    score = _max_triangle_free_bitmask(adj)
                    if score < best_score:
                        best_score = score
                        best_adj = adj[:]
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        else:
            # 30% chance to remove an inter-part edge or swap
            inter_edges = get_edges_between_parts(adj)
            if inter_edges:
                u, v = random.choice(inter_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                score = _max_triangle_free_bitmask(adj)
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Occasionally try a random restart with a dense random K4-free graph
        if attempts % 100 == 0 and random.random() < 0.1:
             # Generate random K4-free graph
             adj_new = [0] * n
             for _ in range(300):
                 u = random.randint(0, n - 1)
                 v = random.randint(0, n - 1)
                 if u == v: continue
                 if u > v: u, v = v, u
                 if not (adj_new[u] & (1 << v)):
                     adj_new[u] |= (1 << v)
                     adj_new[v] |= (1 << u)
                     if not _is_k4_free_bitmask(adj_new):
                         adj_new[u] &= ~(1 << v)
                         adj_new[v] &= ~(1 << u)
             score = _max_triangle_free_bitmask(adj_new)
             if score < best_score:
                 best_score = score
                 best_adj = adj_new[:]
                 adj = adj_new[:]
        
        attempts += 1

    # Convert best_adj back to list of edges
    final_edges = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                final_edges.append([i, j])
            m ^= lsb
            
    return final_edges
