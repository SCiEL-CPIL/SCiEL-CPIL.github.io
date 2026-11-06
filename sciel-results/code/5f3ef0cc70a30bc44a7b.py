"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with symmetric connection set S has a triangle.
    Triangle exists if there exist x, y in S (distinct) such that (x+y) mod n is in S or -S.
    Uses bitmasks for speed.
    """
    if not S:
        return False
    S_set = set(S)
    # Build bitmask of S and -S
    mask_S = 0
    mask_neg = 0
    for x in S_set:
        mask_S |= (1 << x)
        mask_neg |= (1 << ((-x) % n))
    
    S_list = sorted(S_set)
    for i in range(len(S_list)):
        x = S_list[i]
        for j in range(i + 1, len(S_list)):
            y = S_list[j]
            z = (x + y) % n
            if (mask_S >> z) & 1 or (mask_neg >> z) & 1:
                return True
    return False

def _max_independent_set_size(n, adj_mask):
    """
    Compute alpha of a graph given its adjacency bitmasks.
    Uses branch and bound with bitmasks and degree-based vertex ordering.
    Improved: use greedy coloring bound on the complement for tighter upper bounds.
    """
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def greedy_color_bound(candidates):
        """Greedy coloring of the complement subgraph induced by candidates.
        The number of colors gives an upper bound on the clique number of the complement,
        which equals the independence number of the original graph.
        """
        if candidates == 0:
            return 0
        colors = 0
        remaining = candidates
        while remaining:
            colors += 1
            # Greedily pick a maximal independent set in the complement (clique in original)
            # Actually, we need to color the complement, so we pick vertices that are NOT adjacent in complement
            # i.e., vertices that ARE adjacent in original graph
            color_class = 0
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # v can be added to this color class if it's not adjacent in complement to any in color_class
                # Not adjacent in complement means adjacent in original
                if color_class == 0 or (color_class & adj_mask[v]) == color_class:
                    color_class |= (1 << v)
                    remaining ^= lsb
                temp ^= lsb
        return colors
    
    def dfs(candidates, current_size):
        # Upper bound using greedy coloring
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Quick bound: if we can't beat best, prune
        if current_size + candidates.bit_count() <= best[0]:
            return
        
        # Tighter bound via greedy coloring
        color_bound = greedy_color_bound(candidates)
        if current_size + color_bound <= best[0]:
            return
        
        # Pick vertex with highest degree in candidates subgraph for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (candidates & adj_mask[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v (add to independent set, remove v and its neighbors from candidates)
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _hoffman_bound(n, S):
    """
    Compute Hoffman bound for alpha of circulant graph.
    Returns upper bound on alpha. If bound < 10, alpha < 10 guaranteed.
    """
    if n <= 1:
        return n
    S_set = set(S)
    if not S_set:
        return n
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                A[i][j] = 1
    
    try:
        w = np.linalg.eigvalsh(A)
        lambda_min = w[0]
        d = len(S_set)
        if d == 0:
            return n
        if d - lambda_min <= 0:
            return n
        bound = n * (-lambda_min) / (d - lambda_min)
        return int(np.ceil(bound))
    except:
        return n

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant(n, S):
        return False, 0
    
    # Quick Hoffman bound check - if it proves alpha < 10, skip expensive B&B
    hb = _hoffman_bound(n, S)
    if hb < 10:
        return True, hb
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    S_set = frozenset(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj_mask[i] |= (1 << j)
    
    # Run exact B&B
    alpha = _max_independent_set_size(n, adj_mask)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _test_specific_circulants(n, time_limit=3.0):
    """Test specific algebraic circulant connection sets that are known to be triangle-free."""
    start = time.time()
    half = n // 2
    candidates = []
    
    # Test sets based on quadratic residues (Paley-like)
    if n % 4 == 1:
        # Quadratic residues mod n
        QR = set()
        for x in range(1, n):
            if (x * x) % n in range(1, n):
                QR.add((x * x) % n)
        # Symmetrize
        S = set()
        for x in QR:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test small structured sets
    for k in [2, 3, 5, 7]:
        S = set()
        for x in range(1, n):
            if x % k in [1, k-1]:  # +/-1 mod k
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test sets that avoid sum collisions
    for a in range(2, min(8, half)):
        for b in range(a + 1, min(12, half)):
            S = {a, b, (-a) % n, (-b) % n}
            # Check if triangle-free first (fast)
            if not _has_triangle_circulant(n, frozenset(S)):
                candidates.append(frozenset(S))
    
    # Test union of two small sets
    for a in range(2, min(6, half)):
        for b in range(a + 1, min(10, half)):
            for c in range(b + 1, min(14, half)):
                S = {a, b, c, (-a) % n, (-b) % n, (-c) % n}
                if not _has_triangle_circulant(n, frozenset(S)):
                    candidates.append(frozenset(S))
    
    # Test larger structured sets
    for base in range(2, 8):
        S = set()
        for k in range(1, n // base + 1):
            x = k * base
            if 0 < x < n and x % base in [1, base-1, 2, base-2]:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test specific known-good patterns
    patterns = [
        [1, 2, 4, 8],  # Powers of 2
        [1, 3, 5, 7],  # Odd numbers
        [2, 4, 6, 8],  # Even numbers
        [1, 4, 7, 10], # AP with step 3
        [2, 5, 8, 11], # AP with step 3, offset 2
    ]
    for pat in patterns:
        S = set()
        for x in pat:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S and not _has_triangle_circulant(n, frozenset(S)):
            candidates.append(frozenset(S))
    
    for S in candidates:
        if time.time() - start > time_limit:
            break
        valid, alpha = _verify_circulant(n, S)
        if valid:
            return S
    return None

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _enumerate_circulant(n, max_s_size=8, time_limit=10.0):
    """
    Systematically enumerate triangle-free circulant graphs for given n.
    Returns best (n, edges, alpha) or None.
    """
    start = time.time()
    half = n // 2
    available = list(range(1, half + 1))
    
    # Try S sizes from larger to smaller (larger S = denser graph = smaller alpha)
    for s_size in range(min(max_s_size, half), 2, -1):
        if time.time() - start > time_limit:
            break
        
        total_combos = len(available)
        if s_size <= total_combos:
            max_enumerate = 8000
            count = 0
            for combo in combinations(available, s_size):
                if time.time() - start > time_limit:
                    break
                if count >= max_enumerate:
                    break
                count += 1
                
                S = set()
                for x in combo:
                    S.add(x)
                    S.add((-x) % n)
                S = frozenset(S)
                
                if _has_triangle_circulant(n, S):
                    continue
                
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    return n, _build_edges(n, S), alpha
    
    return None

def _structured_candidates(n, time_limit=5.0):
    """Generate structured circulant candidates: multiples, APs, unions."""
    start = time.time()
    half = n // 2
    candidates = []
    
    # Multiples of small bases
    for base in range(2, 10):
        S = set()
        for k in range(1, n // base + 1):
            x = k * base
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Arithmetic progressions
    for start_v in range(1, 6):
        for step in range(2, 10):
            S = set()
            x = start_v
            while x < n:
                S.add(x)
                S.add((-x) % n)
                x += step
            if S:
                candidates.append(frozenset(S))
    
    # Unions of two APs
    for s1 in range(1, 4):
        for st1 in range(2, 6):
            for s2 in range(1, 4):
                for st2 in range(2, 6):
                    if st1 == st2 and s1 == s2:
                        continue
                    S = set()
                    x = s1
                    while x < n:
                        S.add(x)
                        S.add((-x) % n)
                        x += st1
                    x = s2
                    while x < n:
                        S.add(x)
                        S.add((-x) % n)
                        x += st2
                    if S:
                        candidates.append(frozenset(S))
    
    # Random symmetric sets with triangle avoidance (greedy)
    for _ in range(50):
        if time.time() - start > time_limit:
            break
        S = set()
        cands = list(range(1, half + 1))
        random.shuffle(cands)
        for c in cands:
            new_S = S | {c, (-c) % n}
            new_S = frozenset(new_S)
            if not _has_triangle_circulant(n, new_S):
                S = set(new_S)
        if S:
            candidates.append(frozenset(S))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        half = n // 2
        
        # Strategy 1: Specific algebraic circulants (fast, targeted)
        S_specific = _test_specific_circulants(n, time_limit=2.0)
        if S_specific is not None:
            best_n = n
            best_edges = _build_edges(n, S_specific)
            found = True
        
        # Strategy 2: Structured candidates (fast, targeted)
        if not found:
            struct_cands = _structured_candidates(n, time_limit=3.0)
            for S in struct_cands:
                if time.time() > deadline - 10:
                    break
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
        
        # Strategy 3: Systematic enumeration
        if not found:
            result = _enumerate_circulant(n, max_s_size=min(8, half), time_limit=10.0)
            if result:
                n_found, edges, alpha = result
                best_n = n_found
                best_edges = edges
                found = True
        
        # Strategy 4: Greedy with random restarts
        if not found:
            for k in range(min(half, 10), 2, -1):
                if time.time() > deadline - 10:
                    break
                
                num_restarts = 40 if n <= 35 else 20
                
                for restart in range(num_restarts):
                    if time.time() > deadline - 5:
                        break
                    
                    S = set()
                    cands = list(range(1, half + 1))
                    random.shuffle(cands)
                    for c in cands:
                        if len(S) >= k:
                            break
                        new_S = S | {c, (-c) % n}
                        new_S = frozenset(new_S)
                        if not _has_triangle_circulant(n, new_S):
                            S = set(new_S)
                    
                    if len(S) < k:
                        continue
                    
                    S = frozenset(S)
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        # Strategy 5: Random sampling
        if not found:
            for k in range(min(half, 10), 2, -1):
                if time.time() > deadline - 10:
                    break
                
                num_attempts = 150 if n <= 35 else 80
                
                for attempt in range(num_attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    if k > half:
                        continue
                        
                    S_half = set(random.sample(range(1, half + 1), k))
                    S = set()
                    for x in S_half:
                        S.add(x)
                        S.add((-x) % n)
                    S = frozenset(S)
                    
                    if _has_triangle_circulant(n, S):
                        continue
                    
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    # Fallback: Known small graph (Cayley graph on Z_26)
    if best_n < 20:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
