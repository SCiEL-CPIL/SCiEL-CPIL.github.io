"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Uses structured search with exact rational verification.
Key insight: For each rank component c, U[i*m+j,c]*V[j*p+k,c] must be
independent of j. This means for fixed (i,k,c), the triple
(U[i*m+0,c], U[i*m+1,c], U[i*m+2,c]) and (V[0*p+k,c], V[1*p+k,c], V[2*p+k,c])
must satisfy a0*v0 = a1*v1 = a2*v2.
"""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 130.0  # safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _validate_exact(U, V, W, n, m, p):
    """Check that U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c.
    Uses exact integer/half-integer arithmetic."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    # Multiply by 2 to work with integers (half-integers become integers)
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True


def _to_list(M):
    """Convert numpy array to list of lists of int or float (half-integer)."""
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result


def _valid_component(a0, a1, a2, v0, v1, v2):
    """Check if a0*v0 == a1*v1 == a2*v2 using exact half-integer arithmetic."""
    # Convert to doubled integers
    A = [int(round(2*a)) for a in (a0, a1, a2)]
    V = [int(round(2*v)) for v in (v0, v1, v2)]
    # Products: A[j]*V[j] / 4 must be equal
    p0 = A[0] * V[0]
    p1 = A[1] * V[1]
    p2 = A[2] * V[2]
    return p0 == p1 == p2


def _enumerate_valid_patterns():
    """Enumerate all valid (a0,a1,a2) x (v0,v1,v2) patterns for one (i,k,c).
    Returns list of (a0,a1,a2, v0,v1,v2, w) where w = a0*v0 = a1*v1 = a2*v2."""
    vals = [0, 1, -1, 2, -2, 3, -3, 4, -4]  # doubled values, so actual = v/2
    patterns = []
    seen = set()
    for a0 in vals:
        for a1 in vals:
            for a2 in vals:
                for v0 in vals:
                    p0 = a0 * v0
                    for v1 in vals:
                        p1 = a1 * v1
                        if p1 != p0:
                            continue
                        for v2 in vals:
                            p2 = a2 * v2
                            if p2 != p0:
                                continue
                            w = p0 / 4.0  # actual value
                            key = (a0, a1, a2, v0, v1, v2)
                            if key not in seen:
                                seen.add(key)
                                patterns.append((a0/2.0, a1/2.0, a2/2.0, v0/2.0, v1/2.0, v2/2.0, w))
    return patterns


def _try_systematic_rank(r, n, m, p, seed=0, max_iters=20000):
    """Systematic search: pick random valid patterns for each (i,k,c) tuple.
    For each rank component c and each (i,k), we need a consistent pattern.
    We build U and V column by column, ensuring consistency."""
    rng = np.random.default_rng(seed)
    
    # Precompute valid patterns
    patterns = _enumerate_valid_patterns()
    # Filter out trivial all-zero patterns for efficiency
    nontrivial = [pat for pat in patterns if pat[6] != 0]
    all_patterns = patterns  # include zeros
    
    if len(nontrivial) == 0:
        return None
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # For each rank component c, assign a random valid pattern for each (i,k)
        valid = True
        for c in range(r):
            # For each (i,k), pick a random pattern
            for i in range(n):
                for k in range(p):
                    if rng.random() < 0.3:  # 30% chance of zero pattern
                        a0=a1=a2=v0=v1=v2=0.0
                        w=0.0
                    else:
                        pat = nontrivial[rng.integers(len(nontrivial))]
                        a0,a1,a2,v0,v1,v2,w = pat
                    
                    U[i*m+0, c] = a0
                    U[i*m+1, c] = a1
                    U[i*m+2, c] = a2
                    V[0*p+k, c] = v0
                    V[1*p+k, c] = v1
                    V[2*p+k, c] = v2
                    W[k*n+i, c] = w
        
        # Verify
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_constrained_search(r, n, m, p, seed=0, max_iters=50000):
    """Constrained search: build U and V such that the consistency constraint
    U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c] is automatically satisfied.
    
    Approach: For each (i,k,c), the product must be independent of j.
    We parameterize: for each (i,k,c), pick a value w_{ikc}.
    Then U[i*m+j,c] = w_{ikc} / V[j*p+k,c] for some j, but this must be consistent.
    
    Better: Pick V values first, then U values are determined.
    For fixed c, k, and V values (v0, v1, v2), for each i we need
    U[i*m+0,c]*v0 = U[i*m+1,c]*v1 = U[i*m+2,c]*v2 = w_{ikc}
    
    So if v0 != 0: U[i*m+0,c] = w/v0, U[i*m+1,c] = w/v1 (if v1!=0), etc.
    This requires w to be divisible by each non-zero v_j in the half-integer sense.
    
    Let's use a different approach: enumerate small integer/half-integer V columns,
    then determine valid U columns."""
    rng = np.random.default_rng(seed)
    
    # Small values for V entries (doubled)
    v_vals = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6]
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        valid = True
        for c in range(r):
            # Pick random V column
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = v_vals[rng.integers(len(v_vals))] / 2.0
            
            # For each (i,k), determine if there exists a consistent U row
            for i in range(n):
                for k in range(p):
                    v0 = V[0*p+k, c]
                    v1 = V[1*p+k, c]
                    v2 = V[2*p+k, c]
                    
                    # Need a0*v0 = a1*v1 = a2*v2 = w for some half-integer a0,a1,a2
                    # Try small w values
                    found = False
                    for w2 in range(-20, 21):  # w = w2/4
                        w = w2 / 4.0
                        a0 = a1 = a2 = 0.0
                        ok = True
                        for j, vj in enumerate((v0, v1, v2)):
                            if vj == 0:
                                # Then w must be 0, and a_j can be anything
                                if w != 0:
                                    ok = False
                                    break
                                a_vals = [0.0, 0.5, -0.5, 1.0, -1.0]
                                aj = a_vals[rng.integers(len(a_vals))]
                            else:
                                aj = w / vj
                                # Check if aj is a half-integer
                                aj2 = aj * 2
                                if abs(aj2 - round(aj2)) > 1e-9:
                                    ok = False
                                    break
                                aj = round(aj2) / 2.0
                            if j == 0:
                                a0 = aj
                            elif j == 1:
                                a1 = aj
                            else:
                                a2 = aj
                        if ok:
                            U[i*m+0, c] = a0
                            U[i*m+1, c] = a1
                            U[i*m+2, c] = a2
                            W[k*n+i, c] = w
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_hybrid(r, n, m, p, seed=0):
    """Hybrid: use systematic pattern approach for some columns, constrained for others."""
    if time.time() > DEADLINE - 5:
        return None
    res = _try_systematic_rank(r, n, m, p, seed=seed, max_iters=10000)
    if res is not None:
        return res
    res = _try_constrained_search(r, n, m, p, seed=seed+1000, max_iters=10000)
    if res is not None:
        return res
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p  # 72

    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24
    lower = max(n * p, m * p, n * m)
    
    # Try ranks from lower bound upward
    for r in range(lower, best_rank):
        if time.time() > DEADLINE - 5:
            break
        
        seed_base = r * 7919  # prime multiplier for varied seeds
        
        # Strategy 1: systematic pattern search
        res = _try_systematic_rank(r, n, m, p, seed=seed_base, max_iters=15000)
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Strategy 2: constrained search (V-first)
        res = _try_constrained_search(r, n, m, p, seed=seed_base+1, max_iters=15000)
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Strategy 3: hybrid with more iterations if time permits
        if time.time() < DEADLINE - 10:
            res = _try_hybrid(r, n, m, p, seed=seed_base+2)
            if res is not None:
                best = res
                best_rank = r
                break
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
