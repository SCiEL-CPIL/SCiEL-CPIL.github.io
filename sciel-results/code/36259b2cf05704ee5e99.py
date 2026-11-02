"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Uses structured search with exact rational verification.
Key insight: For each rank component c, U[i*m+j,c]*V[j*p+k,c] must be
independent of j. This means for fixed (i,k,c), the triple
(U[i*m+0,c], U[i*m+1,c], U[i*m+2,c]) and (V[0*p+k,c], V[1*p+k,c], V[2*p+k,c])
must satisfy a0*v0 = a1*v1 = a2*v2.

Strategy: Exploit rectangular geometry (3x3 x 3x8). The tensor has n*m*p = 72
nonzero entries. Lower bound is max(n*p, m*p, n*m) = max(24, 24, 9) = 24.
Use structured patterns based on the geometry to find low-rank decompositions.
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


def _build_structured_rank(r, n, m, p):
    """Build a structured decomposition exploiting rectangular geometry.
    
    For 3x3 x 3x8, we can use the fact that the second matrix has 8 columns.
    Strategy: Group the 8 output columns into blocks and use shared rank components.
    
    A rank-r decomposition has:
    - U: (9, r) matrix
    - V: (24, r) matrix  
    - W: (24, r) matrix
    
    For each rank component c, we need:
    U[i*3+j, c] * V[j*8+k, c] = W[k*3+i, c] for all i,j,k
    
    This means for fixed (i,k,c): U[i*3+0,c]*V[0*8+k,c] = U[i*3+1,c]*V[1*8+k,c] = U[i*3+2,c]*V[2*8+k,c]
    
    We can parameterize this using the ratio structure. If we set:
    U[i*3+j, c] = a_i * b_j for some structure, and V[j*8+k, c] = c_k * d_j,
    then the product is a_i * b_j * c_k * d_j, which is independent of j only if
    b_j * d_j is constant (i.e., b_j = const/d_j).
    
    Simpler: For each (i,k,c), pick a value w_{ikc}. Then we need:
    U[i*3+0,c] * V[0*8+k,c] = U[i*3+1,c] * V[1*8+k,c] = U[i*3+2,c] * V[2*8+k,c] = w_{ikc}
    
    Let's use a parameterization where for each c:
    - Pick V[0*8+k,c], V[1*8+k,c], V[2*8+k,c] for all k
    - For each i, pick U[i*3+0,c] and derive U[i*3+1,c], U[i*3+2,c] to satisfy constraints
    """
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Small half-integer values for entries
    small_vals = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    for c in range(r):
        # Pick V column entries
        for j in range(m):
            for k in range(p):
                V[j * p + k, c] = small_vals[random.randint(0, len(small_vals) - 1)]
        
        # For each (i,k), determine U row entries
        for i in range(n):
            for k in range(p):
                v0 = V[0 * p + k, c]
                v1 = V[1 * p + k, c]
                v2 = V[2 * p + k, c]
                
                # Need a0*v0 = a1*v1 = a2*v2 = w
                # Try to find valid (a0,a1,a2,w)
                found = False
                for w in [0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 1.5, -1.5, 2, -2]:
                    a0 = a1 = a2 = 0.0
                    ok = True
                    for j, vj in enumerate([v0, v1, v2]):
                        if vj == 0:
                            if w != 0:
                                ok = False
                                break
                            # a_j can be anything, pick 0
                            aj = 0.0
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
                        U[i * m + 0, c] = a0
                        U[i * m + 1, c] = a1
                        U[i * m + 2, c] = a2
                        W[k * n + i, c] = w
                        found = True
                        break
                if not found:
                    # This configuration doesn't work, reset
                    U[i * m + 0, c] = 0
                    U[i * m + 1, c] = 0
                    U[i * m + 2, c] = 0
                    W[k * n + i, c] = 0
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


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


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy broadcasting. Returns True if valid."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U2 = np.round(2 * U).astype(np.int64)
    V2 = np.round(2 * V).astype(np.int64)
    W2 = np.round(2 * W).astype(np.int64)
    
    for i in range(n):
        for k in range(p):
            # For this (i,k), check all j
            lhs = np.zeros(r, dtype=np.int64)
            for j in range(m):
                lhs += U2[i * m + j, :] * V2[j * p + k, :]
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
            # Round to nearest half-integer
            fv2 = round(fv * 2) / 2
            if fv2 == int(fv2):
                r.append(int(fv2))
            else:
                r.append(fv2)
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


def _try_structured_search(r, n, m, p, seed=0, max_iters=5000):
    """Structured search exploiting rectangular geometry.
    
    Key insight: For 3x3 x 3x8, the constraint is that for each (i,k,c):
    U[i*3+0,c]*V[0*8+k,c] = U[i*3+1,c]*V[1*8+k,c] = U[i*3+2,c]*V[2*8+k,c]
    
    We can parameterize this more efficiently. For each c, if we fix the V column,
    then for each (i,k), the U row entries are determined up to a scaling factor.
    
    Strategy: 
    1. For each c, pick V[0*8+k,c], V[1*8+k,c], V[2*8+k,c] for all k
    2. For each i, pick a scaling factor s_i
    3. Set U[i*3+j,c] = s_i * t_j where t_j are chosen to satisfy constraints
    """
    rng = np.random.default_rng(seed)
    
    # Small half-integer values
    small_vals = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3]
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        valid = True
        for c in range(r):
            # Pick V column
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] = small_vals[rng.integers(len(small_vals))]
            
            # For each i, determine U row
            for i in range(n):
                # Need U[i*3+0,c]*V[0*8+k,c] = U[i*3+1,c]*V[1*8+k,c] = U[i*3+2,c]*V[2*8+k,c] for all k
                # Let's pick U[i*3+0,c] first, then derive others
                a0 = small_vals[rng.integers(len(small_vals))]
                U[i * m + 0, c] = a0
                
                ok = True
                for k in range(p):
                    v0 = V[0 * p + k, c]
                    v1 = V[1 * p + k, c]
                    v2 = V[2 * p + k, c]
                    
                    if v0 != 0:
                        w = a0 * v0
                        # Check if w/v1 and w/v2 are half-integers
                        if v1 != 0:
                            a1 = w / v1
                            if abs(a1 * 2 - round(a1 * 2)) > 1e-9:
                                ok = False
                                break
                            a1 = round(a1 * 2) / 2
                        else:
                            if w != 0:
                                ok = False
                                break
                            a1 = 0.0
                        if v2 != 0:
                            a2 = w / v2
                            if abs(a2 * 2 - round(a2 * 2)) > 1e-9:
                                ok = False
                                break
                            a2 = round(a2 * 2) / 2
                        else:
                            if w != 0:
                                ok = False
                                break
                            a2 = 0.0
                        U[i * m + 1, c] = a1
                        U[i * m + 2, c] = a2
                        W[k * n + i, c] = w
                    else:
                        # v0 = 0, so w must be 0
                        if v1 != 0 or v2 != 0:
                            # Need a1*v1 = a2*v2 = 0, so a1=0 or v1=0, etc.
                            # This is getting complex, let's just set all to 0
                            a1 = 0.0
                            a2 = 0.0
                            U[i * m + 1, c] = a1
                            U[i * m + 2, c] = a2
                            W[k * n + i, c] = 0.0
                        else:
                            U[i * m + 1, c] = 0.0
                            U[i * m + 2, c] = 0.0
                            W[k * n + i, c] = 0.0
                
                if not ok:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        if _validate_fast(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_constrained_search(r, n, m, p, seed=0, max_iters=5000):
    """Constrained search: build U and V such that the consistency constraint
    U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c] is automatically satisfied.
    
    Improved approach: For each c, pick V column first, then for each i,
    pick U[i*3+0,c] and derive U[i*3+1,c], U[i*3+2,c] from constraints.
    This is more efficient than the previous per-(i,k) approach."""
    rng = np.random.default_rng(seed)
    
    # Small values for V entries (doubled)
    v_vals = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8]
    u_vals = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    
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
            
            # For each i, pick U[i*3+0,c] and derive others
            for i in range(n):
                a0 = u_vals[rng.integers(len(u_vals))] / 2.0
                U[i*m+0, c] = a0
                
                ok = True
                for k in range(p):
                    v0 = V[0*p+k, c]
                    v1 = V[1*p+k, c]
                    v2 = V[2*p+k, c]
                    
                    if v0 != 0:
                        w = a0 * v0
                        # Derive a1, a2
                        if v1 != 0:
                            a1 = w / v1
                            a1_2 = a1 * 2
                            if abs(a1_2 - round(a1_2)) > 1e-9:
                                ok = False
                                break
                            a1 = round(a1_2) / 2.0
                        else:
                            if w != 0:
                                ok = False
                                break
                            a1 = 0.0
                        
                        if v2 != 0:
                            a2 = w / v2
                            a2_2 = a2 * 2
                            if abs(a2_2 - round(a2_2)) > 1e-9:
                                ok = False
                                break
                            a2 = round(a2_2) / 2.0
                        else:
                            if w != 0:
                                ok = False
                                break
                            a2 = 0.0
                        
                        U[i*m+1, c] = a1
                        U[i*m+2, c] = a2
                        W[k*n+i, c] = w
                    else:
                        # v0 = 0, w must be 0
                        if v1 != 0 or v2 != 0:
                            U[i*m+1, c] = 0.0
                            U[i*m+2, c] = 0.0
                            W[k*n+i, c] = 0.0
                        else:
                            U[i*m+1, c] = 0.0
                            U[i*m+2, c] = 0.0
                            W[k*n+i, c] = 0.0
                
                if not ok:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        if _validate_fast(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_hybrid(r, n, m, p, seed=0):
    """Hybrid: try multiple search strategies."""
    if time.time() > DEADLINE - 5:
        return None
    
    # Try structured search first (exploits geometry)
    res = _try_structured_search(r, n, m, p, seed=seed, max_iters=3000)
    if res is not None:
        return res
    
    # Try constrained search
    res = _try_constrained_search(r, n, m, p, seed=seed+1000, max_iters=3000)
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
        
        # Strategy 1: structured search (exploits rectangular geometry)
        res = _try_structured_search(r, n, m, p, seed=seed_base, max_iters=5000)
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Strategy 2: constrained search (V-first)
        res = _try_constrained_search(r, n, m, p, seed=seed_base+1, max_iters=5000)
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
