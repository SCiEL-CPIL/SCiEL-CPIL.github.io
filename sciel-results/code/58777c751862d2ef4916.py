import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()

# Precomputed popcount table
_POPCOUNT_TABLE = [0] * 65536
for _i in range(1, 65536):
    _POPCOUNT_TABLE[_i] = _POPCOUNT_TABLE[_i >> 1] + (_i & 1)

def _popcount_fast(x):
    return _POPCOUNT_TABLE[x & 0xFFFF] + _POPCOUNT_TABLE[(x >> 16) & 0xFFFF]

def _is_non_2colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation on bitmasks.
    """
    m = len(edges)
    if m == 0:
        return False
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    def propagate(mask0, mask1):
        changed = True
        while changed:
            changed = False
            for e_mask in edge_masks:
                c0 = _popcount_fast(e_mask & mask0)
                c1 = _popcount_fast(e_mask & mask1)
                c_un = 5 - c0 - c1
                if c0 == 5 or c1 == 5:
                    return 0, 0, False
                if c0 == 4 and c_un == 1:
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask1 = mask1 | unassigned_mask
                    if new_mask1 != mask1:
                        mask1 = new_mask1
                        changed = True
                elif c1 == 4 and c_un == 1:
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask0 = mask0 | unassigned_mask
                    if new_mask0 != mask0:
                        mask0 = new_mask0
                        changed = True
        return mask0, mask1, True

    def backtrack(mask0, mask1):
        best_v = -1
        best_score = -1
        unassigned_all = ~(mask0 | mask1) & ((1 << n) - 1)
        
        while unassigned_all:
            lsb = unassigned_all & -unassigned_all
            v = lsb.bit_length() - 1
            unassigned_all ^= lsb
            
            score = 0
            for ei in vert_edges[v]:
                e_mask = edge_masks[ei]
                c0 = _popcount_fast(e_mask & mask0)
                c1 = _popcount_fast(e_mask & mask1)
                c_un = 5 - c0 - c1
                if c_un == 1:
                    score += 10
                elif c_un == 2:
                    score += 5
                elif c_un == 3:
                    score += 2
                elif c_un == 4:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            for e_mask in edge_masks:
                c0 = _popcount_fast(e_mask & mask0)
                c1 = _popcount_fast(e_mask & mask1)
                if c0 == 5 or c1 == 5:
                    return True
            return False
        
        new_mask0 = mask0 | (1 << best_v)
        if not (new_mask0 & mask1):
            nm0, nm1, ok = propagate(new_mask0, mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        new_mask1 = mask1 | (1 << best_v)
        if not (new_mask1 & mask0):
            nm0, nm1, ok = propagate(mask0, new_mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        return True

    nm0, nm1, ok = propagate(0, 0)
    if not ok:
        return True
    return backtrack(nm0, nm1)


def _greedy_prune(n, edges, rng, timeout):
    """
    Greedy pruning: repeatedly try to remove edges, prioritizing those 
    whose removal is most likely to succeed (i.e., edges that are 'redundant').
    Uses set-cover-like heuristic: remove edges that cover the most 'easy' 
    (low-constraint) vertex combinations first.
    """
    start_time = time.time()
    edges = edges[:]
    n = len(range(n))
    
    # Compute vertex degrees
    def compute_degrees(edge_list):
        degs = [0] * n
        for e in edge_list:
            for v in e:
                degs[v] += 1
        return degs
    
    changed = True
    while changed and time.time() - start_time < timeout:
        changed = False
        degs = compute_degrees(edges)
        
        # Score each edge: prefer removing edges with lower total degree sum
        # (edges on low-degree vertices are more likely redundant)
        scored = []
        for i, e in enumerate(edges):
            score = sum(degs[v] for v in e) + rng.random() * 0.1
            scored.append((score, i))
        
        scored.sort()
        
        for score, idx in scored:
            if time.time() - start_time > timeout:
                break
            if idx >= len(edges):
                continue
            e = edges[idx]
            edges_without = [ed for j, ed in enumerate(edges) if j != idx]
            if _is_non_2colorable(n, edges_without):
                edges = edges_without
                changed = True
    
    return edges


def _local_search(n, edges, rng, timeout):
    """
    Local search: try removing one edge, adding one edge, or swapping.
    Focus on finding smaller non-2-colorable hypergraphs.
    """
    start_time = time.time()
    edges = edges[:]
    all_possible = list(combinations(range(n), 5))
    edge_set = set(tuple(e) for e in edges)
    
    # Try to remove edges
    removed = True
    while removed and time.time() - start_time < timeout:
        removed = False
        shuffled = edges[:]
        rng.shuffle(shuffled)
        for e in shuffled:
            if time.time() - start_time > timeout:
                break
            edges_without = [ed for ed in edges if ed != e]
            if _is_non_2colorable(n, edges_without):
                edges = edges_without
                edge_set.discard(tuple(e))
                removed = True
    
    # Try swapping: remove one, add another
    if time.time() - start_time < timeout:
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            if len(edges) < 2:
                break
            # Pick a random edge to remove
            idx = rng.randrange(len(edges))
            e_remove = edges[idx]
            edges_without = [ed for i, ed in enumerate(edges) if i != idx]
            
            # Try adding a random new edge
            e_add = list(rng.choice(all_possible))
            if tuple(e_add) in edge_set:
                continue
            
            candidate = edges_without + [e_add]
            if _is_non_2colorable(n, candidate):
                edges = candidate
                edge_set.discard(tuple(e_remove))
                edge_set.add(tuple(e_add))
    
    return edges


def _try_from_seed(seed, n, timeout):
    """
    Start from a random subset of edges, check if non-2-colorable,
    then prune.
    """
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    # Start with a moderate number of edges
    start_count = min(len(all_edges), rng.randint(30, 60))
    edges = [list(e) for e in all_edges[:start_count]]
    
    if not _is_non_2colorable(n, edges):
        return None
    
    # Prune
    edges = _greedy_prune(n, edges, rng, timeout)
    return edges


def _try_incremental(seed, n, timeout):
    """
    Build up edges incrementally until non-2-colorable, then prune.
    """
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    edges = []
    for e in all_edges:
        if time.time() - START > DEADLINE - 5:
            break
        edges.append(list(e))
        if len(edges) >= 25:
            if _is_non_2colorable(n, edges):
                # Found a non-2-colorable set, now prune
                edges = _greedy_prune(n, edges, rng, min(timeout, DEADLINE - time.time() - 5))
                return edges
        if len(edges) > 80:
            break
    
    return None


def _known_small_construction():
    """
    Return a known small non-2-colorable 5-uniform hypergraph.
    We'll use the complete hypergraph on 9 vertices as a safe fallback,
    but try to find smaller ones through search.
    """
    # Complete 5-uniform on 9 vertices: C(9,5) = 126 edges
    # This is definitely non-2-colorable
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}


def construct():
    best_edges = float('inf')
    best_result = None
    
    # Fallback
    fallback = _known_small_construction()
    best_result = fallback
    best_edges = len(fallback['edges'])
    
    # Try various strategies
    strategies = []
    
    # Strategy 1: Incremental build for different n values
    for n in [8, 9, 10, 11, 12]:
        strategies.append(('incremental', n))
    
    # Strategy 2: Random seed + prune for different n values
    for n in [9, 10, 11, 12, 13]:
        strategies.append(('seed', n))
    
    # Strategy 3: Greedy prune from complete graph (only for small n)
    for n in [8, 9]:
        strategies.append(('complete_prune', n))
    
    # Shuffle strategies for diversity
    rng_master = random.Random(42)
    rng_master.shuffle(strategies)
    
    seed = 0
    for strategy, n in strategies:
        if time.time() > DEADLINE - 10:
            break
        
        time_left = DEADLINE - time.time()
        if time_left < 8:
            break
        
        timeout = min(15.0, time_left * 0.3)
        
        try:
            if strategy == 'incremental':
                seed += 1
                edges = _try_incremental(seed, n, timeout)
            elif strategy == 'seed':
                seed += 1
                edges = _try_from_seed(seed, n, timeout)
            elif strategy == 'complete_prune':
                seed += 1
                rng = random.Random(seed)
                edges = [list(e) for e in combinations(range(n), 5)]
                if _is_non_2colorable(n, edges):
                    edges = _greedy_prune(n, edges, rng, timeout)
            else:
                continue
            
            if edges is not None and len(edges) < best_edges:
                best_edges = len(edges)
                best_result = {'n': n, 'edges': edges}
                
                # Local search to improve further
                if time.time() < DEADLINE - 15:
                    rng_ls = random.Random(seed + 100)
                    ls_timeout = min(10.0, DEADLINE - time.time() - 10)
                    improved = _local_search(n, best_result['edges'], rng_ls, ls_timeout)
                    if len(improved) < best_edges:
                        best_edges = len(improved)
                        best_result = {'n': n, 'edges': improved}
                        
        except Exception:
            pass
    
    # Final round: try to improve the best result found
    if best_result is not None and time.time() < DEADLINE - 12:
        n_best = best_result['n']
        edges_best = best_result['edges'][:]
        rng_final = random.Random(999)
        final_timeout = DEADLINE - time.time() - 8
        
        if final_timeout > 5:
            # More aggressive pruning
            improved = _greedy_prune(n_best, edges_best, rng_final, final_timeout * 0.5)
            if len(improved) < best_edges:
                best_edges = len(improved)
                best_result = {'n': n_best, 'edges': improved}
            
            if time.time() < DEADLINE - 10:
                # Local search
                improved = _local_search(n_best, best_result['edges'], rng_final, 
                                         DEADLINE - time.time() - 8)
                if len(improved) < best_edges:
                    best_edges = len(improved)
                    best_result = {'n': n_best, 'edges': improved}
    
    if best_result is None:
        return fallback
    
    return best_result
