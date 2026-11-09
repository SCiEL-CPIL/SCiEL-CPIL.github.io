import time
import random
import itertools
from collections import defaultdict

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square of a polynomial."""
    n = len(coefs)
    # Build square coefficients
    sq = [0] * (2 * n - 1)
    for i in range(n):
        a = coefs[i]
        if a == 0:
            continue
        for j in range(i, n):
            v = a * a if i == j else 2 * a * coefs[j]
            if v != 0:
                sq[i + j] += v
    cnt = 0
    for c in sq:
        if c != 0:
            cnt += 1
    return cnt

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
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coefs if c != 0)

def _combine_basis(weights, basis_polys):
    """
    Combine basis polynomials with integer weights.
    weights: list of (weight, basis_poly) where basis_poly is a dict {degree: coeff}
    Returns a coefficient list of length 41.
    """
    result = [0] * 41
    for w, poly in weights:
        if w == 0:
            continue
        for deg, coeff in poly.items():
            if deg < 41:
                result[deg] += w * coeff
    return result

def _generate_basis_polys():
    """
    Generate a diverse set of sparse basis polynomials with known sparse squares.
    These include:
    - Binomials: x^a + x^b (square has 3 terms)
    - Trinomials with special structure
    - Palindromic polynomials
    """
    basis = []
    
    # Binomials: (x^a + x^b)^2 = x^(2a) + 2x^(a+b) + x^(2b) - 3 nonzero terms
    for a in range(0, 15):
        for b in range(a + 1, 35):
            if 2 * b <= 40:
                poly = {a: 1, b: 1}
                basis.append(poly)
    
    # Trinomials with cancellation potential: x^a + x^b - x^c
    # (x^a + x^b - x^c)^2 can have fewer terms if exponents align
    for a in range(0, 10):
        for b in range(a + 1, 20):
            for c in range(b + 1, 35):
                if 2 * c <= 40:
                    # Check if this trinomials square has fewer than 6 terms
                    poly = {a: 1, b: 1, c: -1}
                    # Quick check: compute square nonzero count
                    coefs = [0] * 41
                    coefs[a] = 1
                    coefs[b] = 1
                    coefs[c] = -1
                    sq_count = _sq_nonzero_count(coefs)
                    if sq_count <= 5:
                        basis.append(poly)
    
    # Palindromic binomials: x^a + x^b where a + b = 40
    for a in range(0, 20):
        b = 40 - a
        if b > a:
            basis.append({a: 1, b: 1})
    
    # Mixed sign binomials
    for a in range(0, 15):
        for b in range(a + 1, 35):
            if 2 * b <= 40:
                basis.append({a: 1, b: -1})
    
    return basis

def _search_basis_combinations(basis, rng, best):
    """
    Search over combinations of 2-4 basis polynomials with integer weights.
    """
    n_basis = len(basis)
    if n_basis == 0:
        return best
    
    # Try combinations of 2 basis polynomials
    for _ in range(5000):
        if time.time() > _DEADLINE:
            break
        # Pick 2 random basis polynomials
        i1 = rng.randrange(n_basis)
        i2 = rng.randrange(n_basis)
        if i1 == i2:
            continue
        p1, p2 = basis[i1], basis[i2]
        
        # Try various weight combinations
        for w1 in [-4, -3, -2, -1, 1, 2, 3, 4]:
            for w2 in [-4, -3, -2, -1, 1, 2, 3, 4]:
                coefs = _combine_basis([(w1, p1), (w2, p2)], [(w1, p1), (w2, p2)])
                nz = _count_nz(coefs)
                if nz == 12 and coefs[-1] != 0:
                    if all(-64 <= c <= 64 for c in coefs):
                        sc = _sq_nonzero_count(coefs)
                        if sc < best[0]:
                            best[0] = sc
                            best[1] = coefs[:]
    
    # Try combinations of 3 basis polynomials
    for _ in range(3000):
        if time.time() > _DEADLINE:
            break
        indices = rng.sample(range(n_basis), 3)
        polys = [basis[i] for i in indices]
        
        for w1 in [-3, -2, -1, 1, 2, 3]:
            for w2 in [-3, -2, -1, 1, 2, 3]:
                for w3 in [-3, -2, -1, 1, 2, 3]:
                    weights = [(w1, polys[0]), (w2, polys[1]), (w3, polys[2])]
                    coefs = _combine_basis(weights, weights)
                    nz = _count_nz(coefs)
                    if nz == 12 and coefs[-1] != 0:
                        if all(-64 <= c <= 64 for c in coefs):
                            sc = _sq_nonzero_count(coefs)
                            if sc < best[0]:
                                best[0] = sc
                                best[1] = coefs[:]
    
    return best

def _polish(best, support, rng):
    """Local hill-climbing polish: try changing one coefficient at a time."""
    n = len(support)
    hi = max(support)
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    # Try multiple polish passes
    for _pass in range(5):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for idx, d in enumerate(support):
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                # Try various new values
                candidates = []
                for v in [-64,-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32,64]:
                    if v != old_val:
                        candidates.append(v)
                # Also try small perturbations
                for delta in [-4,-3,-2,-1,1,2,3,4]:
                    v = old_val + delta
                    if -64 <= v <= 64 and v != 0 and v != old_val:
                        candidates.append(v)
                rng.shuffle(candidates)
                for new_val in candidates[:10]:
                    if time.time() > _DEADLINE:
                        break
                    best_local[d] = new_val
                    if not _valid(best_local):
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                    else:
                        best_local[d] = old_val
    
    return best

def _try_direct_support(best, support, rng):
    """Directly assign values to a fixed support and optimize."""
    n = len(support)
    best_local = None
    best_score = 10**9
    hi = max(support)
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    for _ in range(200):
        if time.time() > _DEADLINE:
            break
        coefs = [0] * 41
        for k, d in enumerate(support):
            c = rng.choice(val_pool)
            coefs[d] = c
        if coefs[hi] == 0:
            coefs[hi] = 1
        sc = _sq_nonzero_count(coefs)
        if sc < best_score:
            best_score = sc
            best_local = coefs[:]
        
        # Local perturbation
        if best_local is not None:
            cand = best_local[:]
            d = support[rng.randrange(n)]
            old = cand[d]
            new = old + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if -64 <= new <= 64 and new != 0:
                cand[d] = new
            if _valid(cand):
                sc2 = _sq_nonzero_count(cand)
                if sc2 < best_score:
                    best_score = sc2
                    best_local = cand[:]
    
    if best_local is not None and _valid(best_local):
        sc = _sq_nonzero_count(best_local)
        if sc < best[0]:
            best[0] = sc
            best[1] = best_local[:]
            _polish(best, support, rng)
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 80.0
    rng = random.Random(12345)
    best = [10**9, None]
    
    # Phase 1: Basis polynomial superposition search
    basis = _generate_basis_polys()
    if basis:
        best = _search_basis_combinations(basis, rng, best)
    
    # Phase 2: Direct support search with structured supports
    # Palindromic supports
    for t in range(20):
        if time.time() > _DEADLINE:
            break
        left = sorted(rng.sample(range(0, 20), 6))
        right = [40 - d for d in left]
        support = sorted(set(left + right))
        if 40 not in support:
            support = support[:11] + [40]
            support = sorted(set(support))
        if len(support) == 12:
            _try_direct_support(best, support, rng)
    
    # Arithmetic progressions
    for start in range(0, 8):
        for gap in range(2, 12):
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
                sup = sup[:-1] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            _try_direct_support(best, sup, rng)
    
    # Phase 3: Final polish
    if best[1] is not None:
        support = [i for i in range(41) if best[1][i] != 0]
        _polish(best, support, rng)
    
    if best[1] is None:
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs
    
    return best[1]
