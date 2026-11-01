"""Matrix multiplication 2x2x15 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations, product

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def _check(U, V, W):
    """Exact integer check: U[a,r]*V[b,r]*W[c,r] must equal T[a,b,c]."""
    n, m, p = DIMENSIONS
    # Build T as a dict of (a,b,c) -> 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = T.get((a, b, c), 0) + 1
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = T.get((a, b, c), 0)
                if s != expected:
                    return False, (a, b, c, s, expected)
    return True, None

def _strassen_decomposition():
    """Rank-53 Strassen: 7 pairs + 1 single column."""
    n, m, p = DIMENSIONS
    rank = 7 * 7 + 4  # 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    def add_strassen_block(base_r, k0, k1):
        r = [base_r + i for i in range(7)]
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        E_idx = 1 * p + k0
        F_idx = 1 * p + k1

        # M1 = (a+d)(e+F)
        U[0][r[0]] = 1; U[3][r[0]] = 1
        V[e_idx][r[0]] = 1; V[F_idx][r[0]] = 1
        # M2 = (c+d)e
        U[2][r[1]] = 1; U[3][r[1]] = 1
        V[e_idx][r[1]] = 1
        # M3 = a(F-f)
        U[0][r[2]] = 1
        V[F_idx][r[2]] = 1; V[f_idx][r[2]] = -1
        # M4 = d(E-e)
        U[3][r[3]] = 1
        V[E_idx][r[3]] = 1; V[e_idx][r[3]] = -1
        # M5 = (a+b)F
        U[0][r[4]] = 1; U[1][r[4]] = 1
        V[F_idx][r[4]] = 1
        # M6 = (c-a)(e+f)
        U[2][r[5]] = 1; U[0][r[5]] = -1
        V[e_idx][r[5]] = 1; V[f_idx][r[5]] = 1
        # M7 = (b-d)(E+F)
        U[1][r[6]] = 1; U[3][r[6]] = -1
        V[E_idx][r[6]] = 1; V[F_idx][r[6]] = 1

        # C[0,k0] = k0*n+0 = 2*k0: M1+M4-M5+M7
        W[k0*2][r[0]] = 1; W[k0*2][r[3]] = 1
        W[k0*2][r[4]] = -1; W[k0*2][r[6]] = 1
        # C[1,k0] = 2*k0+1: M2+M4
        W[k0*2+1][r[1]] = 1; W[k0*2+1][r[3]] = 1
        # C[0,k1] = 2*k1: M3+M5
        W[k1*2][r[2]] = 1; W[k1*2][r[4]] = 1
        # C[1,k1] = 2*k1+1: M1-M2+M3+M6
        W[k1*2+1][r[0]] = 1; W[k1*2+1][r[1]] = -1
        W[k1*2+1][r[2]] = 1; W[k1*2+1][r[5]] = 1

    for block in range(7):
        add_strassen_block(7 * block, 2 * block, 2 * block + 1)

    # Direct for column 14
    base_r = 49
    k = 14
    U[0][base_r] = 1; V[0*p+k][base_r] = 1; W[k*2][base_r] = 1
    U[1][base_r+1] = 1; V[1*p+k][base_r+1] = 1; W[k*2][base_r+1] = 1
    U[2][base_r+2] = 1; V[0*p+k][base_r+2] = 1; W[k*2+1][base_r+2] = 1
    U[3][base_r+3] = 1; V[1*p+k][base_r+3] = 1; W[k*2+1][base_r+3] = 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    dim_a = n * m  # 4
    dim_b = m * p  # 30
    dim_c = p * n  # 30

    # Build the multiplication tensor T as a flat dict
    # T[a,b,c] = 1 for a=i*m+j, b=j*p+k, c=k*n+i
    T_entries = []  # list of (a,b,c)
    T_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_entries.append((a, b, c))
                T_set.add((a, b, c))
    num_T = len(T_entries)  # 60

    # Known good: rank 53 Strassen
    U53, V53, W53 = _strassen_decomposition()
    ok, err = _check(U53, V53, W53)
    best_rank = 53 if ok else 60
    best_result = {"U": U53, "V": V53, "W": W53} if ok else None
    if not ok:
        # Fallback: direct rank 60
        U = [[0]*60 for _ in range(dim_a)]
        V = [[0]*60 for _ in range(dim_b)]
        W = [[0]*60 for _ in range(dim_c)]
        for idx, (a, b, c) in enumerate(T_entries):
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 60

    if _remaining() < 5:
        return best_result

    # Search for lower rank using constraint matrix approach
    # For a given rank r, we need U (4xr), V (30xr), W (30xr) integers
    # such that for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # and for (a,b,c) not in T_set: sum_r U[a,r]*V[b,r]*W[c,r] = 0
    #
    # This is a system of polynomial (quadratic in variables) equations.
    # Strategy: fix W, then the system becomes linear in U and V.
    # Specifically, for each c, let w_c = W[c,:] (a vector of length r).
    # Then for each (a,b): sum_r U[a,r]*V[b,r]*w_c[r] = T[a,b,c]
    # This is bilinear in U and V.
    #
    # Alternative: For each rank r, use a randomized approach:
    # 1. Randomly initialize W with small integers
    # 2. Solve for U and V using linear algebra
    # 3. Check if the solution is integer and valid
    #
    # Actually, let's use a smarter approach:
    # The tensor T has rank at most 60 (direct). We want to find the minimum r.
    # Known lower bound: for 2x2xp, the rank is at least 2*ceil(7p/4) for p>=2? No.
    # For 2x2x2, rank is 7 (Strassen). For 2x2xp, rank is at most 7*floor(p/2)+4*ceil(p/2).
    # For p=15: 7*7+4*1 = 53.
    #
    # Can we do better than 53? The tensor 2x2x15 = 2x2x(7*2+1).
    # There's no known way to do better than Strassen per pair + direct for single.
    # Actually, for 2x2xp the minimum rank is known to be:
    # For p=1: 4, p=2: 7, p=3: 11, p=4: 14, etc.
    # The formula is: rank(2x2xp) = 2*floor(7p/4) for p even? Let me check.
    # p=2: 7, p=4: 14, p=6: 21, p=8: 28...
    # For p odd: p=1: 4, p=3: 11, p=5: 18, p=7: 25, p=9: 32, p=11: 39, p=13: 46, p=15: 53
    # Actually the known result is: rank(2x2xp) = 7*floor(p/2) + 4*ceil(p/2) for p>=1?
    # p=1: 0+4=4, p=2: 7+0=7, p=3: 7+4=11, p=4: 14+0=14, ..., p=15: 49+4=53.
    # This matches. And it's known to be optimal for these small cases.
    #
    # So 53 is likely optimal. But let's verify our Strassen is correct.
    
    if ok:
        # Try to find rank < 53 (unlikely but let's check a few)
        # For rank r < 53, the system is overdetermined.
        # Quick check: can rank 52 work?
        # The constraint matrix approach: for each r, the number of free variables is:
        # 4r + 30r + 30r = 64r, and the number of constraints is 4*30*30 = 3600 (but many are 0=0).
        # Actually the constraints are: for each (a,b,c), one equation.
        # Total (a,b,c) tuples: 4*30*30 = 3600.
        # Of these, 60 are =1 and 3540 are =0.
        # The equations are quadratic (bilinear in U,V for fixed W, etc.)
        
        # Given the known optimality result, 53 is the answer.
        # Let's just return the verified rank-53 solution.
        return best_result
    else:
        return best_result

# EVOLVE-BLOCK-END
