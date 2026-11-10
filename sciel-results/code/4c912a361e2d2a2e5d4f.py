import time
import random

_START = time.time()
_DEADLINE = 120.0
_N = 18
_MASK_ALL = (1 << _N) - 1

def _elapsed():
    return time.time() - _START

def _is_k4_free_addition(adj, u, v):
    if (adj[u] >> v) & 1:
        return True
    common = adj[u] & adj[v]
    c = common
    while c:
        lsb = c & -c
        x = lsb.bit_length() - 1
        c ^= lsb
        if adj[x] & common:
            return False
    return True

def _build_triangle_adj(adj):
    t_adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common:
                    t_adj[u] |= 1 << v
                    t_adj[v] |= 1 << u
    return t_adj

def _max_independent_set_size(t_adj):
    best = [0]
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        
        # Prune
        if size + mask.bit_count() <= best[0]:
            return
            
        # Heuristic: Pick vertex with highest degree in the induced subgraph
        max_deg = -1
        v = -1
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            deg = (t_adj[x] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = x
        
        if v == -1:
            return

        # Branch 1: Include v
        new_mask = mask & ~t_adj[v] & ~(1 << v)
        _solve(new_mask, size + 1)
        
        # Branch 2: Exclude v
        _solve(mask & ~(1 << v), size)

    _solve(_MASK_ALL, 0)
    return best[0]

def _generate_turan():
    """Generate Turan graph T(18, 3)."""
    adj = [0] * _N
    # Partition into 3 sets of 6
    set_a = list(range(0, 6))
    set_b = list(range(6, 12))
    set_c = list(range(12, 18))
    
    for u in set_a:
        for v in set_b:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        for v in set_c:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
    for u in set_b:
        for v in set_c:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
    return adj

def _generate_random_k4free():
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    random.seed(42)
    
    # Initialize with Turan graph
    adj = _generate_turan()
    t_adj = _build_triangle_adj(adj)
    score = _max_independent_set_size(t_adj)
    
    if score < best_score:
        best_score = score
        best_adj = adj[:]
        
    # Local Search Loop
    # We will alternate between:
    # 1. Greedy Add Edge (monotonic improvement)
    # 2. Stochastic Edge Swap (to escape local minima)
    
    while _elapsed() < deadline - 2:
        improved = False
        
        # Phase 1: Greedy Add Edge
        while improved and _elapsed() < deadline - 2:
            improved = False
            non_edges = []
            for u in range(_N):
                for v in range(u + 1, _N):
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
            
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if _elapsed() > deadline - 2:
                    break
                
                if _is_k4_free_addition(adj, u, v):
                    # Tentatively add
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    
                    if new_score < score:
                        score = new_score
                        t_adj = new_t_adj
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                            if best_score <= 4:
                                # Found very good solution
                                pass
                    else:
                        # Revert
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
            
        # Phase 2: Stochastic Edge Swap (Remove one, Add another)
        # This helps if we are stuck at a local minimum where no single add works
        # but a swap works.
        if _elapsed() < deadline - 2:
            edges_list = []
            non_edges = []
            for u in range(_N):
                for v in range(u + 1, _N):
                    if (adj[u] >> v) & 1:
                        edges_list.append((u, v))
                    else:
                        non_edges.append((u, v))
            
            if edges_list and non_edges:
                # Try a few random swaps
                swaps_tried = 0
                max_swaps = 20
                while swaps_tried < max_swaps and _elapsed() < deadline - 2:
                    e_rm = random.choice(edges_list)
                    e_add = random.choice(non_edges)
                    
                    u1, v1 = e_rm
                    u2, v2 = e_add
                    
                    # Remove e_rm
                    adj[u1] ^= 1 << v1
                    adj[v1] ^= 1 << u1
                    
                    # Check if e_add is valid K4-free addition
                    if _is_k4_free_addition(adj, u2, v2):
                        adj[u2] |= 1 << v2
                        adj[v2] |= 1 << u2
                        
                        new_t_adj = _build_triangle_adj(adj)
                        new_score = _max_independent_set_size(new_t_adj)
                        
                        # Accept if better or equal (to explore)
                        if new_score <= score:
                            score = new_score
                            t_adj = new_t_adj
                            if score < best_score:
                                best_score = score
                                best_adj = adj[:]
                        else:
                            # Revert e_add
                            adj[u2] ^= 1 << v2
                            adj[v2] ^= 1 << u2
                    # e_rm remains removed if we accepted the swap? 
                    # No, if we revert e_add, we should restore e_rm to keep state consistent for next trial?
                    # Actually, if we revert e_add, we are back to state with e_rm removed.
                    # We should restore e_rm to go back to original state.
                    adj[u1] |= 1 << v1
                    adj[v1] |= 1 << u1
                    
                    swaps_tried += 1
                    
                    # If we found a very good score, break
                    if best_score <= 4:
                        break

    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
