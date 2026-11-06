"""MILP-inspired constraint-satisfaction with bitmask scoring and targeted repair."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 135.0  # safety margin before 150s kill

def _score_from_mask(adj_mask, n=16):
    """Compute max triangle degree-sum using bitmasks. adj_mask[i] is int with bit j set if edge (i,j)."""
    deg = np.array([adj_mask[i].bit_count() for i in range(n)], dtype=np.int32)
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i] >> j & 1:
                common = adj_mask[i] & adj_mask[j]
                # Iterate over set bits in common
                k = 0
                while common:
                    if common & 1:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
                    common >>= 1
                    k += 1
    return best

def _build_mask(edge_set, n=16):
    """Build bitmask adjacency from edge set."""
    mask = [0] * n
    for u, v in edge_set:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _mask_to_edges(mask, n=16):
    """Convert bitmask adjacency to sorted edge list."""
    edges = set()
    for i in range(n):
        m = mask[i]
        j = 0
        while m:
            if m & 1 and j > i:
                edges.add((i, j))
            m >>= 1
            j += 1
    return edges

def _fast_local_search(adj_mask, n, edge_set, time_limit=5.0):
    """Fast local search using bitmask scoring. Returns (score, adj_mask, edge_set)."""
    start = time.time()
    current_score = _score_from_mask(adj_mask, n)
    best_score = current_score
    best_mask = list(adj_mask)
    best_edge_set = set(edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start < time_limit:
        if time.time() > _DEADLINE_GUARD:
            break
        
        current_edges = list(edge_set)
        non_edges = [e for e in all_possible if e not in edge_set]
        
        best_swap_score = current_score
        best_swap = None
        
        for e_rm in current_edges:
            u, v = e_rm
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edge_set.remove(e_rm)
            
            for e_add in non_edges:
                a, b = e_add
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                edge_set.add(e_add)
                
                score = _score_from_mask(adj_mask, n)
                
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                edge_set.remove(e_add)
                
                if score < best_swap_score:
                    best_swap_score = score
                    best_swap = (e_rm, e_add)
            
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(e_rm)
        
        if best_swap:
            e_rm, e_add = best_swap
            u, v = e_rm
            a, b = e_add
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edge_set.remove(e_rm)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            edge_set.add(e_add)
            current_score = best_swap_score
            if current_score < best_score:
                best_score = current_score
                best_mask = list(adj_mask)
                best_edge_set = set(edge_set)
        else:
            break
    
    return best_score, best_mask, best_edge_set

def _targeted_repair(adj_mask, n, edge_set, target_score=23, time_limit=10.0):
    """Targeted repair: focus on reducing triangles with degree sum > target_score."""
    start = time.time()
    current_score = _score_from_mask(adj_mask, n)
    if current_score <= target_score:
        return current_score, adj_mask, edge_set
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start < time_limit:
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Find the worst triangle
        deg = np.array([adj_mask[i].bit_count() for i in range(n)], dtype=np.int32)
        worst_tri = None
        worst_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] >> j & 1:
                    common = adj_mask[i] & adj_mask[j]
                    k = 0
                    while common:
                        if common & 1:
                            s = deg[i] + deg[j] + deg[k]
                            if s > worst_sum:
                                worst_sum = s
                                worst_tri = (i, j, k)
                        common >>= 1
                        k += 1
        
        if worst_tri is None or worst_sum <= target_score:
            break
        
        i, j, k = worst_tri
        # Try to remove one edge from the worst triangle and add a "safe" edge
        candidates_rm = [(i,j), (i,k), (j,k)]
        random.shuffle(candidates_rm)
        
        improved = False
        for e_rm in candidates_rm:
            u, v = e_rm
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edge_set.remove(e_rm)
            
            # Find a non-edge to add that doesn't create new high-sum triangles
            non_edges = [e for e in all_possible if e not in edge_set]
            random.shuffle(non_edges)
            
            best_add = None
            best_score_after = current_score
            
            for e_add in non_edges[:50]:  # Limit candidates for speed
                a, b = e_add
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                edge_set.add(e_add)
                
                score = _score_from_mask(adj_mask, n)
                
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                edge_set.remove(e_add)
                
                if score < best_score_after:
                    best_score_after = score
                    best_add = e_add
            
            # Revert removal
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(e_rm)
            
            if best_add and best_score_after < current_score:
                a, b = best_add
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                edge_set.remove(e_rm)
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                edge_set.add(best_add)
                current_score = best_score_after
                improved = True
                break
        
        if not improved:
            break
    
    return current_score, adj_mask, edge_set

def _milp_relaxation(n=16, target_edges=65, time_limit=20.0):
    """
    MILP-inspired approach: greedily construct a graph minimizing max triangle degree sum.
    Uses a smooth surrogate: minimize sum of exp(alpha * (deg_sum - T)) over triangles.
    """
    start = time.time()
    alpha = 2.0
    T = 23.0  # target threshold
    
    # Initialize with a random graph
    edge_set = set()
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    for e in all_possible[:target_edges]:
        edge_set.add(e)
    
    adj_mask = _build_mask(edge_set, n)
    current_score = _score_from_mask(adj_mask, n)
    best_score = current_score
    best_mask = list(adj_mask)
    best_es = set(edge_set)
    
    # Greedy edge selection: build up graph one edge at a time, choosing edge that minimizes worst-case impact
    # First, reset to empty and build up
    edge_set2 = set()
    adj_mask2 = [0] * n
    
    for _ in range(target_edges):
        if time.time() - start > time_limit:
            break
        
        deg = np.array([adj_mask2[i].bit_count() for i in range(n)], dtype=np.int32)
        
        best_e = None
        best_obj = float('inf')
        
        # Sample a subset of candidate edges for speed
        candidates = [e for e in all_possible if e not in edge_set2]
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        for e in candidates:
            u, v = e
            adj_mask2[u] |= (1 << v)
            adj_mask2[v] |= (1 << u)
            
            # Compute objective: sum of exp(alpha*(s-T)) over triangles involving u or v
            # For speed, just compute max triangle sum
            score = _score_from_mask(adj_mask2, n)
            
            adj_mask2[u] &= ~(1 << v)
            adj_mask2[v] &= ~(1 << u)
            
            # Objective: prefer edges that don't increase max triangle sum
            obj = score + 0.1 * (deg[u] + deg[v])  # slight penalty for high-degree edges
            
            if obj < best_obj:
                best_obj = obj
                best_e = e
        
        if best_e:
            u, v = best_e
            adj_mask2[u] |= (1 << v)
            adj_mask2[v] |= (1 << u)
            edge_set2.add(best_e)
    
    score2 = _score_from_mask(adj_mask2, n)
    if score2 < best_score:
        best_score = score2
        best_mask = list(adj_mask2)
        best_es = set(edge_set2)
    
    return best_score, best_mask, best_es

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Phase 1: Start with structured initialization (K_{6,10} + matching)
    edge_set = set()
    for u in range(6):
        for v in range(6, 16):
            edge_set.add((u, v))
    for a, b in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        edge_set.add((a, b))
    
    adj_mask = _build_mask(edge_set, n)
    best_score = _score_from_mask(adj_mask, n)
    best_mask = list(adj_mask)
    best_edge_set = set(edge_set)
    
    # Phase 2: Fast local search from structured start
    if time.time() < _DEADLINE_GUARD - 60:
        score, mask, es = _fast_local_search(list(adj_mask), n, set(edge_set), time_limit=8.0)
        if score < best_score:
            best_score = score
            best_mask = mask
            best_edge_set = es
    
    # Phase 3: MILP relaxation - build graph greedily
    if time.time() < _DEADLINE_GUARD - 40:
        score, mask, es = _milp_relaxation(n, target_edges, time_limit=15.0)
        if score < best_score:
            best_score = score
            best_mask = mask
            best_edge_set = es
    
    # Phase 4: Targeted repair on best solution
    if time.time() < _DEADLINE_GUARD - 15:
        score, mask, es = _targeted_repair(list(best_mask), n, set(best_edge_set), target_score=23, time_limit=12.0)
        if score < best_score:
            best_score = score
            best_mask = mask
            best_edge_set = es
    
    # Phase 5: Multiple random restarts with local search
    for trial in range(8):
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Create perturbed start
        adj_cur = list(best_mask)
        es_cur = set(best_edge_set)
        k = random.randint(5, 15)
        for _ in range(k):
            existing = list(es_cur)
            if not existing:
                break
            e_rm = random.choice(existing)
            u, v = e_rm
            adj_cur[u] &= ~(1 << v)
            adj_cur[v] &= ~(1 << u)
            es_cur.discard(e_rm)
            
            for _try in range(100):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) == (u, v) or (a, b) in es_cur:
                    continue
                adj_cur[a] |= (1 << b)
                adj_cur[b] |= (1 << a)
                es_cur.add((a, b))
                break
        
        score, mask, es = _fast_local_search(adj_cur, n, es_cur, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_mask = mask
            best_edge_set = es
        
        if best_score <= 23:
            break
    
    # Final polish: targeted repair on best
    if time.time() < _DEADLINE_GUARD - 5:
        score, mask, es = _targeted_repair(list(best_mask), n, set(best_edge_set), target_score=23, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_mask = mask
            best_edge_set = es
    
    # Final verification
    final_score = _score_from_mask(best_mask, n)
    if final_score != best_score:
        best_score = final_score
    
    edges = [[u, v] for u, v in sorted(best_edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
