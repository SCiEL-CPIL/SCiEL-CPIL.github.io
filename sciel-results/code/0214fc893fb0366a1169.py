"""Constructive coloring using quadratic residues modulo a prime."""
import time
import random

# EVOLVE-BLOCK-START
N = 128

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) for a < p, p prime."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p) % p
    # Result: 1 if QR, p-1 (i.e., -1) if QNR, 0 if a==0

def _quadratic_residue_coloring(p):
    """
    Assign colors based on quadratic character modulo p.
    Position i gets color:
      0 if i mod p is a quadratic residue (nonzero)
      1 if i mod p is a quadratic non-residue
      2 if i mod p == 0
    This gives at most 3 colors.
    """
    colors = [0] * N
    for i in range(N):
        r = i % p
        if r == 0:
            colors[i] = 2
        else:
            ls = pow(r, (p - 1) // 2, p)
            if ls == 1:
                colors[i] = 0
            else:
                colors[i] = 1
    return colors

def _check_valid(colors, N):
    """Check that every nonconstant 4-term AP has at least 3 distinct colors."""
    for a in range(N):
        for d in range(1, N):
            e = a + 3 * d
            if e >= N:
                break
            ca = colors[a]
            cb = colors[a + d]
            cc = colors[a + 2 * d]
            ce = colors[e]
            if len({ca, cb, cc, ce}) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _try_merge(colors, c1, c2):
    """Try merging color c2 into c1. Returns True if valid."""
    changed = [i for i in range(N) if colors[i] == c2]
    if not changed:
        return False
    # Check all progressions that include at least one changed position
    changed_set = set(changed)
    for a in range(N):
        for d in range(1, N):
            e = a + 3 * d
            if e >= N:
                break
            positions = [a, a + d, a + 2 * d, e]
            if not any(p in changed_set for p in positions):
                continue
            ca = c1 if colors[a] == c2 else colors[a]
            cb = c1 if colors[a + d] == c2 else colors[a + d]
            cc = c1 if colors[a + 2 * d] == c2 else colors[a + 2 * d]
            ce = c1 if colors[e] == c2 else colors[e]
            if len({ca, cb, cc, ce}) < 3:
                return False
    return True

def _minimize_colors(colors, deadline):
    """Iteratively try to merge colors to reduce count."""
    colors = list(colors)
    unique = sorted(set(colors))
    mapping = {c: i for i, c in enumerate(unique)}
    colors = [mapping[c] for c in colors]
    k = len(unique)
    
    if k <= 1:
        return colors
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c1 in range(k):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, k):
                if _try_merge(colors, c1, c2):
                    for i in range(N):
                        if colors[i] == c2:
                            colors[i] = c1
                    present = sorted(set(colors))
                    new_map = {c: i for i, c in enumerate(present)}
                    colors = [new_map[c] for c in colors]
                    k = len(present)
                    improved = True
                    break
            if improved:
                break
    return colors

def _greedy_with_k_colors(k, deadline):
    """Greedy: assign smallest valid color in [0,k-1] to each position."""
    colors = [-1] * N
    for x in range(N):
        found = False
        for color in range(k):
            colors[x] = color
            valid = True
            for d in range(1, x // 3 + 1):
                a = x - 3 * d
                if a < 0:
                    break
                b = a + d
                c = a + 2 * d
                if len({colors[a], colors[b], colors[c], colors[x]}) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    best_result = None
    best_count = N + 1
    
    # Strategy 1: Quadratic residue coloring with various primes
    primes_to_try = [127, 113, 101, 89, 83, 71, 67, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3]
    for p in primes_to_try:
        if time.time() > deadline - 10:
            break
        colors = _quadratic_residue_coloring(p)
        if _check_valid(colors, N):
            cnt = _count_colors(colors)
            if cnt < best_count:
                best_count = cnt
                best_result = list(colors)
            # Try to minimize further
            optimized = _minimize_colors(colors, deadline - 5)
            if _check_valid(optimized, N):
                ocnt = _count_colors(optimized)
                if ocnt < best_count:
                    best_count = ocnt
                    best_result = list(optimized)
    
    # Strategy 2: Randomized greedy with small palettes
    random.seed(42)
    for k in range(3, 8):
        if time.time() > deadline - 15:
            break
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            colors = [-1] * N
            valid = True
            for x in range(N):
                # Try colors in random order
                order = list(range(k))
                random.shuffle(order)
                found = False
                for color in order:
                    colors[x] = color
                    ok = True
                    for d in range(1, x // 3 + 1):
                        a = x - 3 * d
                        if a < 0:
                            break
                        b = a + d
                        c = a + 2 * d
                        if len({colors[a], colors[b], colors[c], colors[x]}) < 3:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                if _check_valid(colors, N):
                    cnt = _count_colors(colors)
                    if cnt < best_count:
                        best_count = cnt
                        best_result = list(colors)
                    optimized = _minimize_colors(colors, deadline - 5)
                    if _check_valid(optimized, N):
                        ocnt = _count_colors(optimized)
                        if ocnt < best_count:
                            best_count = ocnt
                            best_result = list(optimized)
    
    # Strategy 3: Deterministic greedy as fallback
    if time.time() < deadline - 5:
        for k in range(3, 10):
            if time.time() > deadline - 5:
                break
            colors = _greedy_with_k_colors(k, deadline)
            if colors is not None and _check_valid(colors, N):
                cnt = _count_colors(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = list(colors)
                optimized = _minimize_colors(colors, deadline - 2)
                if _check_valid(optimized, N):
                    ocnt = _count_colors(optimized)
                    if ocnt < best_count:
                        best_count = ocnt
                        best_result = list(optimized)
    
    if best_result is not None:
        # Remap to 0..k-1 for cleanliness
        unique = sorted(set(best_result))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in best_result]
    
    # Emergency fallback
    return list(range(N))
# EVOLVE-BLOCK-END
