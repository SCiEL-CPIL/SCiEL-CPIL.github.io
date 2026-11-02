"""Matrix multiplication 2x3x8: find low-rank exact decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
DEADLINE = 150.0
SAFETY = 10.0

def _time_left():
    return DEADLINE - SAFETY - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p):
    """Exact verification: for all i,k, sum_c U[i*m+? ...] reconstructs C_{i,k}=sum_j A_{ij}B_{jk}."""
    # Tensor T[a,b,c] where a in [n*m], b in [m*p], c in [p*n]
    # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
    # Decomposition: T = sum_c U[a,c]*V[b,c]*W[c_idx,c]  (W indexed by c_idx in [p*n])
    # Check: for all (a,b,c_idx), T[a,b,c_idx] == sum_r U[a,r]*V[b,r]*W[c_idx,r]
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # Build tensor as dict of nonzeros
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[(a, b, c_idx)] = 1
    # Compute sum_r U[a,r]V[b,r]W[c_idx,r] for all (a,b,c_idx) in support and also check zeros
    # We'll check all (a,b,c_idx) that are in support, plus a sample of zeros
    for (a, b, c_idx), val in T.items():
        s = 0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c_idx, r]
        if s != val:
            return False
    # Check a few zero entries (all zeros must be 0)
    all_a = list(range(n * m))
    all_b = list(range(m * p))
    all_c = list(range(p * n))
    for a in all_a:
        for b in all_b:
            for c_idx in all_c:
                if (a, b, c_idx) in T:
                    continue
                s = 0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c_idx, r]
                if s != 0:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Build a 2x3x2 rank-11 block via Strassen(7) on the 2x2 (i,k) cross + 4 diagonal j-terms.
# For 2x3x2: A (2x3), B (3x2), C (2x2). C_{i,k}=sum_j A_{ij}B_{jk}.
# Use 4 terms for j=0 (diagonal in i,k): A_{i0}B_{0k}C_{ik} for i,k in {0,1} -> 4 terms.
# Remaining: C_{i,k} += A_{i1}B_{1k} + A_{i2}B_{2k}. This is a 2x2 (i,k) "matrix" where
# entry (i,k) = A_{i1}B_{1k}+A_{i2}B_{2k} = sum_{j in {1,2}} A_{ij}B_{jk}.
# This is a 2x2x2 tensor (i, j in {1,2}, k) which is exactly 2x2x2 matrix mult rank 7 (Strassen).
# So total 4 + 7 = 11 per 2x3x2 block. 4 blocks (k in pairs) -> 44.

def _strassen_2x2x2():
    # Returns 7 triples (u_i, v_j, w_k) for 2x2x2: C_{ik}=sum_j A_{ij}B_{jk}
    # Standard Strassen indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # Represent each M as sum of rank-1 (A-linear, B-linear, C-linear) with coefficients.
    # A-linear part: for each M, a vector in A-space (2x2=4 coords) ; B-linear: 4 coords; C-linear: 4 coords (coeffs for C_ik).
    terms = []
    # M1
    terms.append(([(1,0),(1,3)], [(1,0),(1,3)], [(1,0),(1,3)]))  # A00+A11, B00+B11, C00+C11
    # M2
    terms.append(([(1,2),(1,3)], [(1,0)], [(1,2),(1,3)]))  # A10+A11, B00, C10+C11
    # M3
    terms.append(([(1,0)], [(1,2),(-1,3)], [(1,1),(1,2)]))  # A00, B10-B11, C01+C10
    # M4
    terms.append(([(1,3)], [(-1,1),(1,2)], [(1,0),(1,2)]))  # A11, B01-B10, C00+C10
    # M5
    terms.append(([(1,0),(1,1)], [(1,3)], [(1,1),(1,3)]))  # A00+A01, B11, C01+C11
    # M6
    terms.append(([(1,2),(-1,0)], [(1,0),(1,2)], [(1,3)]))  # A10-A00, B00+B10, C11
    # M7
    terms.append(([(1,1),(-1,3)], [(1,1),(1,3)], [(1,0),(1,3)]))  # A01-A11, B01+B11, C00+C11
    return terms

def _build_2x3x2_block():
    # Returns list of (U_vec over 6 A-coords, V_vec over 6 B-coords, W_vec over 4 C-coords)
    # A coords: (i,j) i in 0..1, j in 0..2 -> index i*3+j
    # B coords: (j,k) j in 0..2, k in 0..1 -> index j*2+k
    # C coords: (k,i) k in 0..1, i in 0..1 -> index k*2+i
    blocks = []
    # 4 diagonal terms for j=0: A_{i0} B_{0k} C_{ik}
    for i in range(2):
        for k in range(2):
            U = [0]*6; V = [0]*6; W = [0]*4
            U[i*3+0] = 1
            V[0*2+k] = 1
            W[k*2+i] = 1
            blocks.append((U, V, W))
    # Strassen-7 for the 2x2 (i,k) with j in {1,2}
    # Map Strassen A-coord (si,sj) in 2x2 to real A index: i=si, j=1+sj
    # Map Strassen B-coord (sj,sk) to real B index: j=1+sj, k=sk
    # Map Strassen C-coord (sk,si) to real C index: k=sk, i=si
    for (a_list, b_list, c_list) in _strassen_2x2x2():
        U = [0]*6; V = [0]*6; W = [0]*4
        for (coef, (si, sj)) in a_list:
            U[si*3 + (1+sj)] += coef
        for (coef, (sj, sk)) in b_list:
            V[(1+sj)*2 + sk] += coef
        for (coef, (sk, si)) in c_list:
            W[sk*2 + si] += coef
        blocks.append((U, V, W))
    return blocks

def _build_44(n, m, p):
    # p=8 = 4 blocks of 2 (k-pairs). m=3, n=2.
    # Each block covers k in [2*b, 2*b+2).
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for b in range(p // 2):
        for (Ub, Vb, Wb) in _build_2x3x2_block():
            # Ub over A (n*m=6), Vb over B within block (m*2=6) -> map to global B index j*p + (2*b + k)
            # Wb over C within block (2*n=4) -> map to global C index (2*b+k)*n + i
            for a in range(n*m):
                if Ub[a]:
                    U[a][col] += Ub[a]
            for j in range(m):
                for k in range(2):
                    if Vb[j*2+k]:
                        V[j*p + (2*b+k)][col] += Vb[j*2+k]
            for k in range(2):
                for i in range(n):
                    if Wb[k*2+i]:
                        W[(2*b+k)*n + i][col] += Wb[k*2+i]
            col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    # Try the 44 construction
    try:
        cand = _build_44(n, m, p)
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            best = cand
    except Exception:
        pass
    if _time_left() <= 0:
        return best
    # Try to reduce further by searching for linear dependence / lower rank via random exact search is risky;
    # keep the verified 44 (or 48 fallback).
    return best
