import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 140.0
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

def _greedy_mis_lower_bound(t_adj, mask):
    count = 0
    m = mask
    while m:
        best_v = -1
        best_deg = _N + 1
        c = m
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            deg = (t_adj[v] & m).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        count += 1
        m &= ~t_adj[best_v] & ~(1 << best_v)
    return count

def _max_independent_set_size(t_adj):
    best = [0]
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        if size + mask.bit_count() <= best[0]:
            return
        lb = size + _greedy_mis_lower_bound(t_adj, mask)
        if lb > best[0]:
            best[0] = lb
            if size + mask.bit_count() <= best[0]:
                return
        
        v = -1
        max_deg = -1
        c = mask
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            deg = (t_adj[x] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = x
        
        new_mask = mask & ~t_adj[v] & ~(1 << v)
        _solve(new_mask, size + 1)
        _solve(mask & ~(1 << v), size)

    _solve(_MASK_ALL, 0)
    return best[0]

def _generate_random_k4free():
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def _fast_score(adj):
    """Fast score computation using precomputed triangle adjacency."""
    t_adj = _build_triangle_adj(adj)
    return _max_independent_set_size(t_adj)

def _simulated_annealing(adj, deadline, target_score=11):
    """SA with incremental triangle tracking for fast evaluation."""
    current_score = _fast_score(adj)
    best_adj = adj[:]
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    edge_set = set(all_edges)
    
    while _elapsed() < deadline - 5 and T > 0.1:
        # Pick a random edge to toggle
        u, v = random.choice(all_edges)
        has_edge = (adj[u] >> v) & 1
        
        # Try toggle
        if has_edge:
            # Remove edge
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            new_score = _fast_score(adj)
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    if best_score <= target_score:
                        break
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
        else:
            # Add edge (must be K4-free)
            if _is_k4_free_addition(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                new_score = _fast_score(adj)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                        if best_score <= target_score:
                            break
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
        
        T *= cooling
    
    return best_adj, best_score

def _local_search_polish(adj, current_score, deadline):
    """Polish with 1-move and 2-move neighborhood."""
    improved = True
    while improved and _elapsed() < deadline - 3:
        improved = False
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        # 1-move: try adding each missing edge
        for u in range(_N):
            for v in range(u + 1, _N):
                if not ((adj[u] >> v) & 1):
                    if _is_k4_free_addition(adj, u, v):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        new_t_adj = _build_triangle_adj(adj)
                        new_score = _max_independent_set_size(new_t_adj)
                        if new_score < score:
                            score = new_score
                            improved = True
                        else:
                            adj[u] ^= 1 << v
                            adj[v] ^= 1 << u
                        if _elapsed() > deadline - 3:
                            return adj, score
        
        # 1-move: try removing each existing edge
        for u in range(_N):
            for v in range(u + 1, _N):
                if (adj[u] >> v) & 1:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                    if _elapsed() > deadline - 3:
                        return adj, score
        
        # 2-move: try adding one edge and removing another
        if _elapsed() < deadline - 5:
            missing = [(u, v) for u in range(_N) for v in range(u + 1, _N) if not ((adj[u] >> v) & 1)]
            existing = [(u, v) for u in range(_N) for v in range(u + 1, _N) if (adj[u] >> v) & 1]
            
            random.shuffle(missing)
            random.shuffle(existing)
            
            for (u1, v1) in missing[:30]:
                if not _is_k4_free_addition(adj, u1, v1):
                    continue
                adj[u1] |= 1 << v1
                adj[v1] |= 1 << u1
                
                for (u2, v2) in existing[:30]:
                    adj[u2] ^= 1 << v2
                    adj[v2] ^= 1 << u2
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    if new_score < score:
                        score = new_score
                        improved = True
                        adj[u2] |= 1 << v2
                        adj[v2] |= 1 << u2
                        break
                    else:
                        adj[u2] |= 1 << v2
                        adj[v2] |= 1 << u2
                
                adj[u1] ^= 1 << v1
                adj[v1] ^= 1 << u1
                
                if _elapsed() > deadline - 5:
                    return adj, score
    
    return adj, score

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    random.seed(42)
    
    # Start with known good construction: complete 3-partite graph K6,6,6
    # This has score 12 (any 2 parts form a triangle-free induced subgraph of size 12)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * _N
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
    
    base_score = _fast_score(adj)
    best_score = base_score
    best_adj = adj[:]
    
    # Phase 1: Simulated Annealing from multiple random starts
    for start_iter in range(8):
        if _elapsed() > deadline - 40:
            break
        
        if start_iter == 0:
            sa_adj = best_adj[:]
        else:
            sa_adj = _generate_random_k4free()
        
        sa_result, sa_score = _simulated_annealing(sa_adj, deadline, target_score=11)
        if sa_score < best_score:
            best_score = sa_score
            best_adj = sa_result[:]
        
        if best_score <= 11:
            break
    
    # Phase 2: Local search polish on the best solution
    if best_score < 18 and _elapsed() < deadline - 10:
        polished_adj, polished_score = _local_search_polish(best_adj[:], best_score, deadline)
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj
    
    # Phase 3: More SA iterations if time remains
    if best_score > 11 and _elapsed() < deadline - 15:
        for extra_iter in range(5):
            if _elapsed() > deadline - 10:
                break
            sa_adj = _generate_random_k4free()
            sa_result, sa_score = _simulated_annealing(sa_adj, deadline, target_score=11)
            if sa_score < best_score:
                best_score = sa_score
                best_adj = sa_result[:]
            if best_score <= 11:
                break
    
    # Final polish
    if best_score < 18 and _elapsed() < deadline - 3:
        polished_adj, polished_score = _local_search_polish(best_adj[:], best_score, deadline)
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj
    
    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
