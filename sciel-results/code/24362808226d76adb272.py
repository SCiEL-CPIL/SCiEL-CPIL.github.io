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
    Uses branch and bound with bitmasks and good ordering.
    """
    n = 18
    full_mask = (1 << n) - 1
    best = [0]
    
    # Precompute edge list for fast edge-in-mask checks
    edge_mask_pairs = []
    for i in range(n):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edge_mask_pairs.append((i, j, (1 << i) | (1 << j)))
            m ^= lsb
    
    def _has_edge_in_mask(mask):
        """Check if the subgraph induced by mask contains any edge."""
        for i, j, edge_mask in edge_mask_pairs:
            if mask & edge_mask == edge_mask:
                return True
        return False
    
    def branch(candidates, current_set, current_size):
        # Pruning
        if current_size + _popcount(candidates) <= best[0]:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidate subgraph (for better pruning)
        v_idx = -1
        best_score = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = _popcount(adj[v] & candidates)
            if deg > best_score:
                best_score = deg
                v_idx = v
        
        v = v_idx
        v_bit = 1 << v
        rest_candidates = candidates & ~v_bit
        
        # Option 1: Include v
        N_v = adj[v] & current_set
        can_add = True
        if N_v and _has_edge_in_mask(N_v):
            can_add = False
        
        if can_add:
            new_set = current_set | v_bit
            new_size = current_size + 1
            
            # Update candidates: remove vertices that would form triangle with v and some u in current_set
            forbidden = 0
            if N_v:
                m3 = N_v
                while m3:
                    lsb3 = m3 & -m3
                    u = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    forbidden |= (adj[v] & adj[u])
            
            new_candidates = rest_candidates & ~forbidden
            branch(new_candidates, new_set, new_size)
        
        # Option 2: Exclude v
        branch(rest_candidates, current_set, current_size)
    
    branch(full_mask, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Initial graph: Turan T(18,3)
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_score = _max_triangle_free_bitmask(adj)
    best_adj = adj[:]
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges
        
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if not (m & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    # Systematic local search: try all single edge flips
    def systematic_local_search(adj_current, score_current):
        adj_work = adj_current[:]
        score_work = score_current
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            edges = get_edges(adj_work)
            non_edges = get_non_edges(adj_work)
            
            # Try removing edges
            for u, v in edges:
                if time.time() > deadline - 10:
                    break
                adj_work[u] &= ~(1 << v)
                adj_work[v] &= ~(1 << u)
                s = _max_triangle_free_bitmask(adj_work)
                if s < score_work:
                    score_work = s
                    improved = True
                else:
                    adj_work[u] |= (1 << v)
                    adj_work[v] |= (1 << u)
            
            # Try adding edges
            for u, v in non_edges:
                if time.time() > deadline - 10:
                    break
                adj_work[u] |= (1 << v)
                adj_work[v] |= (1 << u)
                if _is_k4_free_bitmask(adj_work):
                    s = _max_triangle_free_bitmask(adj_work)
                    if s < score_work:
                        score_work = s
                        improved = True
                    else:
                        adj_work[u] &= ~(1 << v)
                        adj_work[v] &= ~(1 << u)
                else:
                    adj_work[u] &= ~(1 << v)
                    adj_work[v] &= ~(1 << u)
        
        return adj_work, score_work
    
    # First: systematic local search from Turan
    best_adj, best_score = systematic_local_search(best_adj, best_score)
    
    # Then: random restarts with local search
    attempts = 0
    while time.time() < deadline - 5:
        # Generate a random K4-free graph by starting from empty and adding edges
        adj_new = [0] * n
        rng = random.Random(attempts)
        for _ in range(300):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if not (adj_new[u] & (1 << v)):
                adj_new[u] |= (1 << v)
                adj_new[v] |= (1 << u)
                if not _is_k4_free_bitmask(adj_new):
                    adj_new[u] &= ~(1 << v)
                    adj_new[v] &= ~(1 << u)
        
        # Do a quick local search from this random graph
        adj_new, score_new = systematic_local_search(adj_new, _max_triangle_free_bitmask(adj_new))
        
        if score_new < best_score:
            best_score = score_new
            best_adj = adj_new[:]
        
        # Also try random swaps from current best
        if time.time() < deadline - 5:
            adj_try = best_adj[:]
            edges = get_edges(adj_try)
            non_edges = get_non_edges(adj_try)
            if edges and non_edges:
                for _ in range(min(20, len(edges))):
                    if time.time() > deadline - 5:
                        break
                    u, v = random.choice(edges)
                    x, y = random.choice(non_edges)
                    adj_try[u] &= ~(1 << v)
                    adj_try[v] &= ~(1 << u)
                    adj_try[x] |= (1 << y)
                    adj_try[y] |= (1 << x)
                    if _is_k4_free_bitmask(adj_try):
                        s = _max_triangle_free_bitmask(adj_try)
                        if s < best_score:
                            best_score = s
                            best_adj = adj_try[:]
                    # Revert
                    adj_try[u] |= (1 << v)
                    adj_try[v] |= (1 << u)
                    adj_try[x] &= ~(1 << y)
                    adj_try[y] &= ~(1 << x)
                    # Rebuild lists
                    edges = get_edges(adj_try)
                    non_edges = get_non_edges(adj_try)
        
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
