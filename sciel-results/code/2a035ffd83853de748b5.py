import time
import random
from itertools import combinations

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle."""
    S_set = frozenset(S)
    neg_S = frozenset({(-x) % n for x in S})
    allowed = S_set | neg_S
    for x in S:
        for y in S:
            z = (x + y) % n
            if z in allowed:
                return True
    return False

def _max_independent_set_circulant(n, S):
    """Compute alpha of circulant graph using branch and bound with bitmasks."""
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    
    def dfs(candidates, current_size):
        nonlocal best
        if current_size + candidates.bit_count() <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        v = (candidates & -candidates).bit_length() - 1
        dfs(candidates & comp_mask[v], current_size + 1)
        dfs(candidates & ~(1 << v), current_size)
        
    dfs(full_mask, 0)
    return best

def _verify_circulant(n, S):
    if _has_triangle_circulant(n, S):
        return False
    alpha = _max_independent_set_circulant(n, S)
    if alpha >= 10:
        return False
    return True

def _build_edges_circulant(n, S):
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Heuristic candidate for n=39
    # S_pos = {14, 15, 16, 17, 18, 19}
    # This makes S = {14..25} which is symmetric and sum-free in Z_39
    if time.time() < deadline:
        n = 39
        S_pos = {14, 15, 16, 17, 18, 19}
        S = set()
        for x in S_pos:
            S.add(x)
            S.add((-x) % n)
        if _verify_circulant(n, S):
            return {'n': n, 'edges': _build_edges_circulant(n, S)}
            
    # Fallback: Search for n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid S for this n
        # Use random search with triangle-free constraint
        random.seed(42 + n)
        
        # Generate random symmetric sets
        for _ in range(500):
            if time.time() > deadline:
                break
                
            half = n // 2
            # Choose size of S_pos
            # Density around 0.3-0.5
            size = random.randint(3, half)
            S_pos = set(random.sample(range(1, half + 1), size))
            
            S = set()
            for x in S_pos:
                S.add(x)
                S.add((-x) % n)
                
            if _verify_circulant(n, S):
                best_n = n
                best_edges = _build_edges_circulant(n, S)
                return {'n': best_n, 'edges': best_edges}
                
    # If n=39..30 failed, try smaller n
    if best_n < 30:
        for n in range(29, 19, -1):
            if time.time() > deadline:
                break
            random.seed(42 + n)
            for _ in range(200):
                if time.time() > deadline:
                    break
                half = n // 2
                size = random.randint(3, half)
                S_pos = set(random.sample(range(1, half + 1), size))
                S = set()
                for x in S_pos:
                    S.add(x)
                    S.add((-x) % n)
                if _verify_circulant(n, S):
                    best_n = n
                    best_edges = _build_edges_circulant(n, S)
                    return {'n': best_n, 'edges': best_edges}
                    
    # Final fallback: n=20 (should always work)
    # C_20 with S={1, 2, 3, 4, 5, 6, 7, 8, 9} ? No, must be triangle free.
    # Simple cycle C_20 is triangle free, alpha=10. Not valid.
    # C_20 with chords?
    # Let's just use a known small valid graph or return empty if desperate
    # But n=20 with S={1, 4, 6, 9} ?
    # Let's try to find one quickly
    if best_n < 20:
        n = 20
        # Try random
        random.seed(123)
        for _ in range(1000):
            if time.time() > deadline:
                break
            half = n // 2
            size = random.randint(3, half)
            S_pos = set(random.sample(range(1, half + 1), size))
            S = set()
            for x in S_pos:
                S.add(x)
                S.add((-x) % n)
            if _verify_circulant(n, S):
                best_n = n
                best_edges = _build_edges_circulant(n, S)
                break
                
    if best_n == 0:
        # Absolute fallback: return a valid graph if we can find one quickly
        # Otherwise, return n=1
        best_n = 1
        best_edges = []
        
    return {'n': best_n, 'edges': best_edges}
