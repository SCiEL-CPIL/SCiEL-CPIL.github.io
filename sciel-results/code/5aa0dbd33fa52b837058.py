"""Biclique partition: Maximal biclique enumeration + Greedy Set Cover + Conflict-based Local Search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
_EDGE_SET = set()
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))

def _star_partition():
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res

def _valid_partition(bics):
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET

def _biclique_edges(L, R):
    es = set()
    for u in L:
        for v in R:
            es.add((min(u, v), max(u, v)))
    return es

def _generate_candidates(deadline):
    """Generate a pool of candidate bicliques by enumerating small left sides."""
    candidates = []
    # We will enumerate subsets of vertices of size 1 to 4 for L
    # For each L, we find the maximal R such that L x R is a biclique.
    # To ensure we don't miss bicliques where the "small" side is actually R, 
    # we rely on the fact that we can swap L and R in the representation.
    # However, the problem requires specific L, R lists. 
    # We will generate candidates (L, R) where L is small.
    
    max_size = 4
    vertices = list(range(_N))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline:
            break
        for L_tuple in combinations(vertices, size):
            L = list(L_tuple)
            # Find maximal R: intersection of neighbors of all u in L
            # R must be disjoint from L
            R_set = set()
            for u in L:
                R_set = R_set.union(_ADJ[u])
            R_set.difference_update(L)
            
            if not R_set:
                continue
            
            # Check if L x R_set is a valid biclique (all edges exist)
            # Since R_set is intersection of neighbors, all u in L are connected to all v in R_set.
            # So it is a valid biclique.
            # But we might want to trim R to only include vertices that have edges in the remaining set?
            # No, candidates are static.
            
            # Optimization: Only keep if it covers at least 1 edge? Yes.
            # Also, we should avoid duplicates or dominated bicliques if possible, but for now just collect.
            # To keep pool size manageable, we might only keep if |L|*|R| is "good" or just all.
            # With N=30, C(30,1)+C(30,2)+C(30,3)+C(30,4) = 30 + 435 + 4060 + 27405 = ~32k candidates.
            # This is small enough to handle.
            
            R = list(R_set)
            if R:
                candidates.append((L, R))
                
    # Also add single edge bicliques as fallbacks? 
    # The greedy set cover will pick the best ones.
    return candidates

def _greedy_set_cover(candidates, deadline, seed=None):
    """Greedy set cover on the candidate pool."""
    rng = random.Random(seed)
    remaining = set(_EDGE_SET)
    chosen = []
    
    # Pre-compute edge sets for candidates for faster lookup?
    # Or just compute on the fly. With 32k candidates, pre-computing might be heavy in memory but fast in time.
    # Let's compute on the fly but optimize.
    
    # To speed up, we can sort candidates by size descending? 
    # Greedy picks max coverage.
    
    while remaining and time.time() < deadline:
        best_idx = -1
        best_count = 0
        best_L = None
        best_R = None
        
        # We can't iterate all 32k candidates every step if steps are many.
        # But N=30, max edges ~150. Max steps ~150 (if single edges).
        # 150 * 32k = 4.8M ops. This is fine in Python if optimized.
        
        # Optimization: Filter candidates that have ANY edge in remaining.
        # We can just check intersection.
        
        for i, (L, R) in enumerate(candidates):
            # Quick check: if no edge from this biclique is in remaining, skip
            # This is expensive to check naively.
            # Let's just compute coverage.
            count = 0
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                best_L = L
                best_R = R
            if time.time() > deadline:
                break
                
        if best_count == 0:
            # Fallback: pick any remaining edge
            e = next(iter(remaining))
            a, b = e
            best_L = [a]
            best_R = [b]
        else:
            best_L, best_R = candidates[best_idx]
            
        chosen.append((best_L, best_R))
        
        # Remove covered edges
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))
                
    # Handle any leftover edges (should be none if candidates cover all edges)
    for e in list(remaining):
        a, b = e
        chosen.append(([a], [b]))
        
    return chosen

def _try_merge(bics, i, j):
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2):
        return None
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics

def _local_search(bics, candidates, deadline):
    """Local search: Merge and Swap."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        
        # 1. Try Merges
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
        
        # 2. Try Swaps (Replace one biclique with a candidate that covers its edges + more)
        if not improved:
            for i in range(n):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                edges_to_cover = _biclique_edges(Li, Ri)
                
                # Find a candidate that covers all edges in edges_to_cover
                # and potentially allows removing other bicliques?
                # Simple swap: Replace biclique i with candidate k if k covers edges_to_cover
                # and k is "better" (e.g., covers more edges that are currently covered by others, 
                # allowing those others to be removed? No, that's complex).
                
                # Let's try: Can we replace biclique i with a candidate k such that 
                # the union of (bics without i) + (k) is valid and has fewer bicliques?
                # Or just: Does k cover edges_to_cover? If so, we can replace i with k.
                # This doesn't reduce count immediately, but might enable merges.
                
                # Better Swap: 
                # Can we remove biclique i if its edges are covered by the union of other bicliques?
                # No, partition requires disjoint coverage.
                
                # Let's try a different move: 
                # If we have biclique i, can we replace it with TWO bicliques from candidates 
                # that cover the same edges? This increases count, bad.
                
                # Let's stick to Merges and maybe "Split-Repack" if we had it.
                # With candidate pool, we can try:
                # For each biclique i, try to find a candidate k that covers edges(i) 
                # and is "larger" or "different" such that it allows a merge with another biclique j?
                
                # For now, just Merges.
                pass
                
    return bics

def construct():
    deadline = time.time() + 85.0
    
    # 1. Generate Candidates
    candidates = _generate_candidates(deadline - 5.0)
    
    # 2. Greedy Set Cover with multiple seeds
    best_bics = None
    best_count = float('inf')
    
    seeds = [None, 42, 123, 7, 999, 31415, 271828, 161803, 555, 888, 111, 222]
    
    for seed in seeds:
        if time.time() >= deadline - 10.0:
            break
            
        bics = _greedy_set_cover(candidates, deadline - 5.0, seed=seed)
        
        # 3. Local Search
        bics = _local_search(bics, candidates, deadline - 3.0)
        
        if not _valid_partition(bics):
            continue
            
        if len(bics) < best_count:
            best_count = len(bics)
            best_bics = bics
            
    if best_bics is None:
        best_bics = _star_partition()
        
    result = [[list(map(int, L)), list(map(int, R))] for L, R in best_bics]
    return result
# EVOLVE-BLOCK-END
