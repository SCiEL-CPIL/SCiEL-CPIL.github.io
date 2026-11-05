"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle.
    Triangle exists if there exist x, y in S (x can equal y) such that (x+y) mod n is in S or -S.
    Vertices 0, x, x+y form a triangle iff x in S, y in S, and (x+y) in S ∪ (-S).
    """
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
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    # Non-neighbors (excluding self)
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    # Use iterative approach with pruning
    def dfs(candidates, current_size):
        nonlocal best
        if current_size + candidates.bit_count() <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        # Pick vertex with highest degree in candidate subgraph (for better pruning)
        # For simplicity, pick the least significant bit
        v = (candidates & -candidates).bit_length() - 1
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant(n, S):
        return False
    alpha = _max_independent_set_circulant(n, S)
    if alpha >= 10:
        return False
    return True

def _build_edges_circulant(n, S):
    """Build edge list for circulant graph."""
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
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_n = 0
    best_edges = []
    
    def try_n(n):
        nonlocal best_n, best_edges
        if time.time() > deadline:
            return False
        
        # Generate candidate symmetric connection sets
        # S must be symmetric: if s in S, then -s mod n in S
        # For simplicity, we choose elements from 1..floor(n/2) and include both s and -s mod n
        # But we need to be careful: if n is even, n/2 is its own negative
        
        half = n // 2
        available = list(range(1, half + 1))
        
        candidates = []
        
        # Strategy 1: Systematic search over subsets of small size
        # For triangle-free, we need S ∪ (-S) to be sum-free
        # Try all subsets of size up to 8 from available
        # But that's too many for large n. Use greedy + random.
        
        # Strategy 2: Greedy construction with triangle avoidance
        random.seed(42 + n)
        
        for density in [0.25, 0.3, 0.35, 0.4, 0.45]:
            target_size = int(n * density / 2)  # Number of positive elements
            for attempt in range(30):
                if time.time() > deadline:
                    return False
                
                S_pos = set()
                shuffled = available[:]
                random.shuffle(shuffled)
                
                for x in shuffled:
                    if len(S_pos) >= target_size:
                        break
                    # Check if adding x (and -x) creates a triangle
                    new_S = S_pos | {x}
                    # Full S includes x and (-x) % n for each element
                    full_new_S = set()
                    for y in new_S:
                        full_new_S.add(y)
                        full_new_S.add((-y) % n)
                    
                    # Check triangle-free
                    S_set = frozenset(full_new_S)
                    neg_S = frozenset({(-x) % n for x in full_new_S})
                    allowed = S_set | neg_S
                    has_tri = False
                    for a in full_new_S:
                        for b in full_new_S:
                            c = (a + b) % n
                            if c in allowed:
                                has_tri = True
                                break
                        if has_tri:
                            break
                    if not has_tri:
                        S_pos = new_S
                
                if len(S_pos) >= 3:
                    # Build full symmetric set
                    S = set()
                    for x in S_pos:
                        S.add(x)
                        S.add((-x) % n)
                    candidates.append(S)
        
        # Strategy 3: Structured sets
        # Multiples
        for base in range(2, 10):
            S = set()
            for k in range(1, n // base + 1):
                x = k * base
                if x < n and x != 0:
                    S.add(x)
                    S.add((-x) % n)
            if S and len(S) >= 4:
                candidates.append(S)
        
        # Arithmetic progressions
        for start in range(1, 6):
            for step in range(2, 10):
                S = set()
                x = start
                while x < n:
                    S.add(x)
                    S.add((-x) % n)
                    x += step
                if S and len(S) >= 4:
                    candidates.append(S)
        
        # Strategy 4: Random sets with varying sizes
        for _ in range(50):
            if time.time() > deadline:
                return False
            size = random.randint(3, min(15, half))
            S_pos = random.sample(available, size)
            S = set()
            for x in S_pos:
                S.add(x)
                S.add((-x) % n)
            candidates.append(S)
        
        # Test candidates - sort by size descending (larger S = denser graph = smaller alpha)
        candidates.sort(key=lambda s: len(s), reverse=True)
        
        seen = set()
        for S in candidates:
            if time.time() > deadline:
                return False
            S_frozen = frozenset(S)
            if S_frozen in seen:
                continue
            seen.add(S_frozen)
            
            if _verify_circulant(n, S):
                edges = _build_edges_circulant(n, S)
                best_n = n
                best_edges = edges
                return True
        
        return False
    
    # Try n from 40 down to 20
    for n in range(40, 19, -1):
        if time.time() > deadline:
            break
        if try_n(n):
            break
    
    # Fallback: known valid graph
    if best_n < 20:
        # Use a simple valid construction
        # Clebsch graph has n=16, alpha=5, triangle-free
        # Blow up by 1 (itself)
        # Actually, let's use a known valid graph
        # The 26-vertex graph from the original fallback
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
