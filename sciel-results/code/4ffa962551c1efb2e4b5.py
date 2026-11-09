import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    n = len(coefs)
    if n == 0:
        return 0
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _count_nz(coefs):
    return sum(1 for c in coefs if c != 0)

def _additive_collision_count(support):
    """Count how many degree positions have multiple pairs of terms summing to them."""
    sums = {}
    for i in range(len(support)):
        for j in range(i, len(support)):
            s = support[i] + support[j]
            if s not in sums:
                sums[s] = 0
            sums[s] += 1
    # Count total "extra" pairs (beyond the first for each sum)
    extra = sum(max(0, v - 1) for v in sums.values())
    return extra

def _generate_high_collision_supports(rng, n_terms=12, max_deg=40):
    """Generate supports with many additive collisions."""
    supports = []
    
    # Type 1: Arithmetic progressions (maximum collisions)
    for start in range(0, 5):
        for gap in range(2, 8):
            sup = list(range(start, start + gap * n_terms, gap))
            if len(sup) == n_terms and sup[-1] <= max_deg:
                supports.append(sup)
    
    # Type 2: Two interleaved arithmetic progressions
    for start1 in range(0, 4):
        for gap1 in range(3, 7):
            for start2 in range(0, 4):
                for gap2 in range(3, 7):
                    if gap1 == gap2 and start1 == start2:
                        continue
                    sup1 = list(range(start1, start1 + gap1 * 6, gap1))
                    sup2 = list(range(start2, start2 + gap2 * 6, gap2))
                    if len(sup1) == 6 and len(sup2) == 6:
                        sup = sorted(set(sup1 + sup2))
                        if len(sup) == n_terms and sup[-1] <= max_deg:
                            supports.append(sup)
    
    # Type 3: Random supports biased toward collision-heavy structures
    for _ in range(500):
        # Start with a base structure and perturb
        base = list(range(0, n_terms * 3, 3))
        sup = base[:]
        for _ in range(rng.randint(0, 4)):
            idx = rng.randint(0, len(sup) - 1)
            sup[idx] += rng.randint(-2, 2)
        sup = sorted(set(sup))
        # Ensure exactly 12 unique values in range
        if len(sup) < n_terms:
            # Fill in missing
            available = [d for d in range(max_deg + 1) if d not in sup]
            while len(sup) < n_terms and available:
                sup.append(rng.choice(available))
                available.remove(sup[-1])
            sup = sorted(set(sup))
        if len(sup) == n_terms and all(0 <= d <= max_deg for d in sup):
            supports.append(sup)
    
    # Type 4: Palindromic supports (symmetric about center)
    center = 20
    for _ in range(200):
        half = rng.sample(range(0, 20), 5)
        sup = sorted(half + [center - h for h in half] + [center])
        sup = sorted(set(sup))
        if len(sup) == n_terms and sup[-1] <= max_deg:
            supports.append(sup)
    
    # Deduplicate
    unique = []
    seen = set()
    for s in supports:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    return unique

def _optimize_coefficients_for_support(support, rng, best):
    """
    Given a fixed support, optimize coefficients to minimize square nonzero count.
    Uses simulated annealing + local search.
    """
    n = len(support)
    if n != 12:
        return best
    
    # Ensure degree 40 is in support for validity
    # If not, we'll handle it
    
    # Initial random coefficients
    val_pool = list(range(-8, 9))
    val_pool = [v for v in val_pool if v != 0]
    
    coeffs = [rng.choice(val_pool) for _ in range(n)]
    
    # Build the polynomial
    P = [0] * 41
    for d, c in zip(support, coeffs):
        if d < 41:
            P[d] = c
    
    if P[40] == 0:
        # Force degree 40 to be nonzero
        # Find which support element maps to 40
        if 40 in support:
            idx = support.index(40)
            coeffs[idx] = 1
            P[40] = 1
        else:
            return best  # Can't satisfy constraint with this support
    
    current_score = _sq_nonzero_count(P)
    
    # Simulated annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    for _step in range(50000):
        if time.time() > _DEADLINE:
            break
        if T < min_T:
            break
        
        # Pick a random coefficient to change
        idx = rng.randint(0, n - 1)
        old_val = coeffs[idx]
        new_val = rng.choice(val_pool)
        if new_val == old_val:
            continue
        
        coeffs[idx] = new_val
        P[support[idx]] = new_val
        
        new_score = _sq_nonzero_count(P)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best[0]:
                best[0] = new_score
                best[1] = P[:]
        else:
            coeffs[idx] = old_val
            P[support[idx]] = old_val
        
        T *= cooling
    
    # Local search from best found
    P = best[1][:] if best[1] is not None else [0] * 41
    best_score = best[0]
    
    # Rebuild coeffs from P
    coeffs = [0] * n
    for i, d in enumerate(support):
        coeffs[i] = P[d]
    
    for _pass in range(20):
        if time.time() > _DEADLINE:
            break
        improved = False
        for idx in range(n):
            if time.time() > _DEADLINE:
                break
            old_val = coeffs[idx]
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                coeffs[idx] = new_val
                P[support[idx]] = new_val
                sc = _sq_nonzero_count(P)
                if sc < best_score:
                    best_score = sc
                    best[0] = sc
                    best[1] = P[:]
                    improved = True
                else:
                    coeffs[idx] = old_val
                    P[support[idx]] = old_val
        if not improved:
            break
    
    return best

def _try_structured_search(best, rng):
    """Main search: generate high-collision supports and optimize coefficients."""
    supports = _generate_high_collision_supports(rng)
    
    # Sort by collision count (more collisions = more potential for cancellation)
    supports.sort(key=lambda s: -_additive_collision_count(s))
    
    # Try top supports
    max_supports = min(len(supports), 200)
    for i, sup in enumerate(supports[:max_supports]):
        if time.time() > _DEADLINE:
            break
        best = _optimize_coefficients_for_support(sup, rng, best)
    
    return best

def _try_product_structure(best, rng):
    """
    Try P = A * B where A and B are sparse polynomials.
    P^2 = A^2 * B^2. If A^2 and B^2 have few terms, P^2 may too.
    """
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _iter in range(10000):
        if time.time() > _DEADLINE:
            break
        
        r = rng.random()
        if r < 0.4:
            k, m = 2, 6
        elif r < 0.7:
            k, m = 3, 4
        elif r < 0.9:
            k, m = 4, 3
        else:
            k, m = 6, 2
        
        exponents_a = sorted(rng.sample(range(0, 25), k))
        coeffs_a = [rng.choice(val_pool) for _ in range(k)]
        
        exponents_b = sorted(rng.sample(range(0, 25), m))
        coeffs_b = [rng.choice(val_pool) for _ in range(m)]
        
        max_deg = max(exponents_a) + max(exponents_b)
        if max_deg > 40:
            continue
        
        P = [0] * 41
        for ea, ca in zip(exponents_a, coeffs_a):
            for eb, cb in zip(exponents_b, coeffs_b):
                d = ea + eb
                if d < 41:
                    P[d] += ca * cb
        
        nz = _count_nz(P)
        if nz != 12 or P[40] == 0:
            continue
        
        if not all(-64 <= c <= 64 for c in P):
            continue
        
        sc = _sq_nonzero_count(P)
        if sc < best[0]:
            best[0] = sc
            best[1] = P[:]
    
    return best

def _try_geometric_structure(best, rng):
    """
    Try polynomials with geometric or arithmetic progression structure.
    """
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for start in range(0, 10):
        for gap in range(2, 15):
            if time.time() > _DEADLINE:
                break
            sup = []
            d = start
            while d <= 40 and len(sup) < 12:
                sup.append(d)
                d += gap
            if len(sup) < 12:
                continue
            if 40 not in sup:
                sup = sup[:11] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            
            for _ in range(100):
                if time.time() > _DEADLINE:
                    break
                P = [0] * 41
                for d in sup:
                    P[d] = rng.choice(val_pool)
                if P[40] == 0:
                    P[40] = 1
                if not _valid(P):
                    continue
                if not all(-64 <= c <= 64 for c in P):
                    continue
                sc = _sq_nonzero_count(P)
                if sc < best[0]:
                    best[0] = sc
                    best[1] = P[:]
    
    return best

def _local_search_global(best, rng):
    """
    Global local search: try to improve by moving terms and changing coefficients.
    """
    if best[1] is None:
        return best
    
    P = best[1][:]
    best_score = best[0]
    
    support = [i for i in range(41) if P[i] != 0]
    n_sup = len(support)
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    for _pass in range(30):
        if time.time() > _DEADLINE:
            break
        improved = False
        for idx in range(n_sup):
            if time.time() > _DEADLINE:
                break
            d = support[idx]
            old_val = P[d]
            
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                P[d] = new_val
                if not _valid(P):
                    P[d] = old_val
                    continue
                sc = _sq_nonzero_count(P)
                if sc < best_score:
                    best_score = sc
                    best[0] = sc
                    best[1] = P[:]
                    improved = True
                else:
                    P[d] = old_val
            
            # Try swapping: remove term at d, add at d2
            for d2 in range(41):
                if d2 in support:
                    continue
                P[d] = 0
                P[d2] = rng.choice(val_pool)
                if _valid(P):
                    sc = _sq_nonzero_count(P)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = P[:]
                        support = [i for i in range(41) if P[i] != 0]
                        n_sup = len(support)
                        improved = True
                        break
                P[d] = old_val
                P[d2] = 0
                if time.time() > _DEADLINE:
                    break
        if not improved:
            break
    
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    rng = random.Random(42)
    best = [10**9, None]
    
    # Phase 1: Structured search with high-collision supports
    best = _try_structured_search(best, rng)
    
    # Phase 2: Product structure search
    best = _try_product_structure(best, rng)
    
    # Phase 3: Geometric/arithmetic structure search
    best = _try_geometric_structure(best, rng)
    
    # Phase 4: Global local search from best found
    if best[1] is not None:
        best = _local_search_global(best, rng)
    
    # Fallback if nothing found
    if best[1] is None:
        P = [0] * 41
        for i in range(12):
            P[i] = 1
        best[0] = _sq_nonzero_count(P)
        best[1] = P
    
    return best[1]
