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

def _generate_random_k4free(seed_offset=0):
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.seed(42 + seed_offset)
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def _local_search_add_edges(adj, score, deadline, best_score, best_adj_ref):
    """Local search that only adds edges. Returns (improved, new_score, new_adj)."""
    improved = False
    while improved and _elapsed() < deadline - 3:
        improved = False
        non_edges = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if _elapsed() > deadline - 3:
                break
            
            if _is_k4_free_addition(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                
                new_t_adj = _build_triangle_adj(adj)
                new_score = _max_independent_set_size(new_t_adj)
                
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score[0]:
                        best_score[0] = score
                        best_adj_ref[0] = adj[:]
                        if best_score[0] <= 8:
                            return True, score, adj
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
    
    return improved, score, adj

def _local_search_swap_edges(adj, score, deadline, best_score, best_adj_ref):
    """Local search that tries edge swaps (remove one, add another)."""
    improved = False
    while improved and _elapsed() < deadline - 3:
        improved = False
        
        # Get edges and non-edges
        edges_list = []
        non_edges = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if (adj[u] >> v) & 1:
                    edges_list.append((u, v))
                else:
                    non_edges.append((u, v))
        
        if not edges_list or not non_edges:
            break
            
        random.shuffle(edges_list)
        random.shuffle(non_edges)
        
        for e_rm in edges_list:
            if _elapsed() > deadline - 3:
                break
            
            u1, v1 = e_rm
            # Temporarily remove edge
            adj[u1] ^= 1 << v1
            adj[v1] ^= 1 << u1
            
            for e_add in non_edges:
                if _elapsed() > deadline - 3:
                    break
                
                u2, v2 = e_add
                if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                    continue
                
                if _is_k4_free_addition(adj, u2, v2):
                    adj[u2] |= 1 << v2
                    adj[v2] |= 1 << u2
                    
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    
                    if new_score < score:
                        score = new_score
                        improved = True
                        if score < best_score[0]:
                            best_score[0] = score
                            best_adj_ref[0] = adj[:]
                            if best_score[0] <= 8:
                                return True, score, adj
                        break
                    else:
                        adj[u2] ^= 1 << v2
                        adj[v2] ^= 1 << u2
            
            # Restore removed edge
            adj[u1] |= 1 << v1
            adj[v1] |= 1 << u1
            
            if best_score[0] <= 8:
                break
    
    return improved, score, adj

def construct():
    deadline = _START + _DEADLINE
    best_score = [18]
    best_adj = [[0] * _N]
    
    random.seed(42)
    
    # Multiple random restarts with different seeds
    for seed_offset in range(0, 50):
        if _elapsed() > deadline - 10:
            break
        
        adj = _generate_random_k4free(seed_offset)
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        if score < best_score[0]:
            best_score[0] = score
            best_adj[0] = adj[:]
            if best_score[0] <= 8:
                break
        
        # Phase 1: Add edges
        improved, score, adj = _local_search_add_edges(
            adj, score, deadline, best_score, best_adj
        )
        if best_score[0] <= 8:
            break
        
        # Phase 2: Swap edges
        improved, score, adj = _local_search_swap_edges(
            adj, score, deadline, best_score, best_adj
        )
        if best_score[0] <= 8:
            break
        
        # Phase 3: Add edges again after swaps
        improved, score, adj = _local_search_add_edges(
            adj, score, deadline, best_score, best_adj
        )
        if best_score[0] <= 8:
            break

    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[0][u] >> v) & 1:
                edges.append([u, v])
                
    return edges
