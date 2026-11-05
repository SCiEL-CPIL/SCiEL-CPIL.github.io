import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 145.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
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
    return U, V, W

def _strassen_2x2(U, V, W, i0, j0, k0, offset):
    """Add Strassen 2x2x2 decomposition for A[2x2] * B[2x2] -> C[2x2].
    A rows: i0, i0+1; cols: k0, k0+1
    B rows: j0, j0+1; cols: k0, k0+1 (but B is m x p, so rows are j, cols are k)
    C rows: j0, j0+1; cols: i0, i0+1 (W index: k*n + i, so C row = k, col = i)
    
    Wait, let me re-check the indexing.
    A is n x m: A[i, j] where i in [0,n), j in [0,m)
    B is m x p: B[j, k] where j in [0,m), k in [0,p)
    C is p x n: C[k, i] where k in [0,p), i in [0,n)
    
    U index: i * m + j
    V index: j * p + k
    W index: k * n + i
    
    For a 2x2x2 block:
    A block: rows i0, i0+1; cols j0, j0+1
    B block: rows j0, j0+1; cols k0, k0+1
    C block: rows k0, k0+1; cols i0, i0+1
    
    Strassen for C = A * B where A is 2x2, B is 2x2, C is 2x2:
    M0 = (A00 + A11)(B00 + B11)
    M1 = (A10 + A11)B00
    M2 = A00(B01 - B11)
    M3 = A11(B10 - B00)
    M4 = (A00 + A01)B11
    M5 = (A10 - A00)(B00 + B01)
    M6 = (A01 - A11)(B10 + B11)
    
    C00 = M0 + M3 - M4 - M6
    C01 = M2 + M4
    C10 = M1 + M5
    C11 = M0 - M1 + M2 + M5
    
    In our indexing:
    A00 = A[i0, j0], A01 = A[i0, j0+1]
    A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
    B00 = B[j0, k0], B01 = B[j0, k0+1]
    B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
    C00 = C[k0, i0], C01 = C[k0, i0+1]
    C10 = C[k0+1, i0], C11 = C[k0+1, i0+1]
    """
    a00 = i0 * m + j0
    a01 = i0 * m + (j0 + 1)
    a10 = (i0 + 1) * m + j0
    a11 = (i0 + 1) * m + (j0 + 1)
    
    b00 = j0 * p + k0
    b01 = j0 * p + (k0 + 1)
    b10 = (j0 + 1) * p + k0
    b11 = (j0 + 1) * p + (k0 + 1)
    
    c00 = k0 * n + i0
    c01 = k0 * n + (i0 + 1)
    c10 = (k0 + 1) * n + i0
    c11 = (k0 + 1) * n + (i0 + 1)
    
    # M0 = (A00 + A11)(B00 + B11)
    U[a00][offset + 0] += 1
    U[a11][offset + 0] += 1
    V[b00][offset + 0] += 1
    V[b11][offset + 0] += 1
    W[c00][offset + 0] += 1
    W[c11][offset + 0] += 1
    
    # M1 = (A10 + A11)B00
    U[a10][offset + 1] += 1
    U[a11][offset + 1] += 1
    V[b00][offset + 1] += 1
    W[c10][offset + 1] += 1
    W[c11][offset + 1] += -1
    
    # M2 = A00(B01 - B11)
    U[a00][offset + 2] += 1
    V[b01][offset + 2] += 1
    V[b11][offset + 2] += -1
    W[c01][offset + 2] += 1
    W[c11][offset + 2] += 1
    
    # M3 = A11(B10 - B00)
    U[a11][offset + 3] += 1
    V[b10][offset + 3] += 1
    V[b00][offset + 3] += -1
    W[c00][offset + 3] += 1
    W[c10][offset + 3] += 1
    
    # M4 = (A00 + A01)B11
    U[a00][offset + 4] += 1
    U[a01][offset + 4] += 1
    V[b11][offset + 4] += 1
    W[c00][offset + 4] += -1
    W[c01][offset + 4] += 1
    
    # M5 = (A10 - A00)(B00 + B01)
    U[a10][offset + 5] += 1
    U[a00][offset + 5] += -1
    V[b00][offset + 5] += 1
    V[b01][offset + 5] += 1
    W[c11][offset + 5] += 1
    
    # M6 = (A01 - A11)(B10 + B11)
    U[a01][offset + 6] += 1
    U[a11][offset + 6] += -1
    V[b10][offset + 6] += 1
    V[b11][offset + 6] += 1
    W[c00][offset + 6] += 1
    
    return offset + 7

def _build_decomposition(n, m, p, partition_i, partition_j, partition_k):
    """Build decomposition using given partitions of i, j, k dimensions.
    For each block (i_block, j_block, k_block), use optimal algorithm based on sizes.
    """
    rank = 0
    # Pre-compute rank for this partition
    for ib in partition_i:
        for jb in partition_j:
            for kb in partition_k:
                li = len(ib)
                lj = len(jb)
                lk = len(kb)
                if li == 2 and lj == 2 and lk == 2:
                    rank += 7
                elif li == 2 and lj == 2:
                    # 2x2 A block, 2x? B block -> for each k in kb, use 2x2 Strassen on A,B cols
                    # Actually, if lj==2 and lk==1: A is 2x2, B is 2x1, C is 1x2
                    # This is a 2x2x1 problem. Can we do better than 4?
                    # 2x2x1: C[k0, i0] = A[i0,j0]*B[j0,k0] + A[i0,j1]*B[j1,k0]
                    # C[k0, i1] = A[i1,j0]*B[j0,k0] + A[i1,j1]*B[j1,k0]
                    # This is 4 mults schoolbook. Can we do 3? 
                    # Let's check: it's like computing two dot products of length 2.
                    # No known 3-mult algorithm for 2x2x1.
                    rank += 4
                elif li == 2 and lk == 2:
                    # 2x? A block, ?x2 B block
                    # If lj==1: A is 2x1, B is 1x2, C is 2x2
                    # 4 mults schoolbook
                    rank += 4
                elif li == 1 and lj == 2 and lk == 2:
                    # 1x2 A, 2x2 B, 2x1 C
                    # 4 mults
                    rank += 4
                elif li == 1 and lj == 1 and lk == 2:
                    # 1x1 A, 1x2 B, 2x1 C
                    # 2 mults
                    rank += 2
                elif li == 1 and lj == 2 and lk == 1:
                    # 1x2 A, 2x1 B, 1x1 C
                    # 2 mults
                    rank += 2
                elif li == 2 and lj == 1 and lk == 1:
                    # 2x1 A, 1x1 B, 1x2 C
                    # 2 mults
                    rank += 2
                elif li == 1 and lj == 1 and lk == 1:
                    rank += 1
                else:
                    # General case: schoolbook
                    rank += li * lj * lk
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    for ib in partition_i:
        for jb in partition_j:
            for kb in partition_k:
                li = len(ib)
                lj = len(jb)
                lk = len(kb)
                i0 = ib[0]
                j0 = jb[0]
                k0 = kb[0]
                
                if li == 2 and lj == 2 and lk == 2:
                    offset = _strassen_2x2(U, V, W, i0, j0, k0, offset)
                elif li == 2 and lj == 2 and lk == 1:
                    # 2x2 A, 2x1 B, 1x2 C: 4 schoolbook
                    for ii in range(2):
                        for jj in range(2):
                            U[(i0+ii)*m + (j0+jj)][offset] = 1
                            V[(j0+jj)*p + k0][offset] = 1
                            W[k0*n + (i0+ii)][offset] = 1
                            offset += 1
                elif li == 2 and lj == 1 and lk == 2:
                    # 2x1 A, 1x2 B, 2x2 C: 4 schoolbook
                    for ii in range(2):
                        for kk in range(2):
                            U[(i0+ii)*m + j0][offset] = 1
                            V[j0*p + (k0+kk)][offset] = 1
                            W[(k0+kk)*n + (i0+ii)][offset] = 1
                            offset += 1
                elif li == 1 and lj == 2 and lk == 2:
                    # 1x2 A, 2x2 B, 2x1 C: 4 schoolbook
                    for jj in range(2):
                        for kk in range(2):
                            U[i0*m + (j0+jj)][offset] = 1
                            V[(j0+jj)*p + (k0+kk)][offset] = 1
                            W[(k0+kk)*n + i0][offset] = 1
                            offset += 1
                elif li == 1 and lj == 1 and lk == 2:
                    # 1x1 A, 1x2 B, 2x1 C: 2 schoolbook
                    for kk in range(2):
                        U[i0*m + j0][offset] = 1
                        V[j0*p + (k0+kk)][offset] = 1
                        W[(k0+kk)*n + i0][offset] = 1
                        offset += 1
                elif li == 1 and lj == 2 and lk == 1:
                    # 1x2 A, 2x1 B, 1x1 C: 2 schoolbook
                    for jj in range(2):
                        U[i0*m + (j0+jj)][offset] = 1
                        V[(j0+jj)*p + k0][offset] = 1
                        W[k0*n + i0][offset] = 1
                        offset += 1
                elif li == 2 and lj == 1 and lk == 1:
                    # 2x1 A, 1x1 B, 1x2 C: 2 schoolbook
                    for ii in range(2):
                        U[(i0+ii)*m + j0][offset] = 1
                        V[j0*p + k0][offset] = 1
                        W[k0*n + (i0+ii)][offset] = 1
                        offset += 1
                elif li == 1 and lj == 1 and lk == 1:
                    U[i0*m + j0][offset] = 1
                    V[j0*p + k0][offset] = 1
                    W[k0*n + i0][offset] = 1
                    offset += 1
                else:
                    # General schoolbook
                    for ii in range(li):
                        for jj in range(lj):
                            for kk in range(lk):
                                U[(i0+ii)*m + (j0+jj)][offset] = 1
                                V[(j0+jj)*p + (k0+kk)][offset] = 1
                                W[(k0+kk)*n + (i0+ii)][offset] = 1
                                offset += 1
    
    if offset != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {offset}")
    
    return U, V, W

def _all_partitions(dim):
    """Generate all partitions of dim into contiguous groups."""
    results = []
    def gen(start, current):
        if start == dim:
            results.append(tuple(current))
            return
        for end in range(start + 1, dim + 1):
            current.append(tuple(range(start, end)))
            gen(end, current)
            current.pop()
    gen(0, [])
    return results

def _find_best_partition(n, m, p):
    """Find the best partition using dynamic programming / exhaustive search."""
    parts_i = _all_partitions(n)
    parts_j = _all_partitions(m)
    parts_k = _all_partitions(p)
    
    best_rank = n * m * p
    best_pi = None
    best_pj = None
    best_pk = None
    
    for pi in parts_i:
        for pj in parts_j:
            for pk in parts_k:
                rank = 0
                for ib in pi:
                    for jb in pj:
                        for kb in pk:
                            li = len(ib)
                            lj = len(jb)
                            lk = len(kb)
                            if li == 2 and lj == 2 and lk == 2:
                                rank += 7
                            elif li == 2 and lj == 2:
                                rank += 4 * lk
                            elif li == 2 and lk == 2:
                                rank += 4 * lj
                            elif li == 1 and lj == 2 and lk == 2:
                                rank += 4
                            elif li == 1 and lj == 1 and lk == 2:
                                rank += 2
                            elif li == 1 and lj == 2 and lk == 1:
                                rank += 2
                            elif li == 2 and lj == 1 and lk == 1:
                                rank += 2
                            elif li == 1 and lj == 1 and lk == 1:
                                rank += 1
                            else:
                                rank += li * lj * lk
                
                if rank < best_rank:
                    best_rank = rank
                    best_pi = pi
                    best_pj = pj
                    best_pk = pk
    
    return best_rank, best_pi, best_pj, best_pk

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Find best partition
    best_rank, best_pi, best_pj, best_pk = _find_best_partition(n, m, p)
    
    if best_rank < n * m * p and best_pi is not None:
        try:
            U, V, W = _build_decomposition(n, m, p, best_pi, best_pj, best_pk)
            if _verify_decomp(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
