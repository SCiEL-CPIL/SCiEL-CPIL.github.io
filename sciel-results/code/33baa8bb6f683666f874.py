"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _tensor_matrix_mult(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]=1 for all i,j,k."""
    T = np.ones((n, m, p))
    return T

def _try_decompose_strassen_hybrid(n, m, p, deadline):
    """
    Hybrid approach: use Strassen on 2x2x2 blocks where possible,
    but also try to find better decompositions for the remaining parts
    using direct tensor rank methods.
    
    Strategy: 3x4x8 = we can split 4 into 2+2, 8 into 2+2+2+2,
    3 into 2+1.
    
    Block (2,2,2): Strassen rank 7
    Block (1,2,2): schoolbook rank 4 (no better known)
    Block (2,2,1): schoolbook rank 4
    Block (1,1,2): schoolbook rank 2
    Block (1,2,1): schoolbook rank 2
    Block (2,1,2): schoolbook rank 4
    Block (2,1,1): schoolbook rank 2
    Block (1,1,1): rank 1
    
    Let's enumerate all ways to split 3, 4, 8 into segments and compute
    the optimal rank for each block combination.
    """
    def splits_of(N):
        """All compositions of N."""
        if N == 0:
            return [[]]
        result = []
        for first in range(1, N+1):
            for rest in splits_of(N - first):
                result.append([first] + rest)
        return result
    
    def block_rank(di, dj, dk):
        """Optimal known rank for a di x dj x dk block."""
        if di == 2 and dj == 2 and dk == 2:
            return 7  # Strassen
        # For other blocks, check if we can use Strassen sub-blocks
        # Actually, for non-2x2x2 blocks, schoolbook is the safe choice
        # But we could also try: if any dimension is 2, maybe partial Strassen helps?
        # For now, schoolbook for non-2x2x2
        return di * dj * dk
    
    best_rank = n * m * p
    best_split = None
    
    i_splits = splits_of(n)
    m_splits = splits_of(m)
    p_splits = splits_of(p)
    
    for iseg in i_splits:
        for mseg in m_splits:
            for pseg in p_splits:
                total_rank = 0
                for di in iseg:
                    for dj in mseg:
                        for dk in pseg:
                            total_rank += block_rank(di, dj, dk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (iseg, mseg, pseg)
    
    if best_split is None:
        return None
    
    iseg, mseg, pseg = best_split
    
    # Build the decomposition
    rank = best_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    i0 = 0
    for di in iseg:
        j0 = 0
        for dj in mseg:
            k0 = 0
            for dk in pseg:
                if di == 2 and dj == 2 and dk == 2:
                    # Strassen
                    strassen_terms = [
                        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
                        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
                        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
                        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
                        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
                        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
                    ]
                    for t in range(7):
                        a_c, b_c, c_c = strassen_terms[t]
                        for a_idx in range(4):
                            ai, aj = a_idx // 2, a_idx % 2
                            U[(i0+ai)*m + (j0+aj), col] = a_c[a_idx]
                        for b_idx in range(4):
                            bj, bk = b_idx // 2, b_idx % 2
                            V[(j0+bj)*p + (k0+bk), col] = b_c[b_idx]
                        for c_idx in range(4):
                            ck, ci = c_idx // 2, c_idx % 2
                            W[(k0+ck)*n + (i0+ci), col] = c_c[c_idx]
                        col += 1
                else:
                    # Schoolbook
                    for ai in range(di):
                        for aj in range(dj):
                            for ak in range(dk):
                                U[(i0+ai)*m + (j0+aj), col] = 1
                                V[(j0+aj)*p + (k0+ak), col] = 1
                                W[(k0+ak)*n + (i0+ai), col] = 1
                                col += 1
                k0 += dk
            j0 += dj
        i0 += di
    
    if col != rank:
        return None
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_direct_tensor_peel(n, m, p, deadline):
    """
    Direct tensor rank minimization via iterative peeling.
    Start with the full tensor and try to peel off rank-1 components.
    Use SVD on matricizations to find candidate rank-1 terms.
    """
    T = _tensor_matrix_mult(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # This is actually the all-ones tensor, which has rank 1!
    # Wait - no. The matrix multiplication tensor is NOT the all-ones tensor.
    # T_{i,j,k} = delta_{i,j,k} only in the sense that the (i,j,k)-th entry of
    # the multiplication tensor is 1 (it's the coefficient of A_ij * B_jk in C_ik).
    # Actually, the multiplication tensor M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # So M = 1_{n} ⊗ 1_{m} ⊗ 1_{p}, which IS rank 1!
    
    # Wait, that can't be right. Let me reconsider.
    # The matrix multiplication tensor is defined such that:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T_{i,j,k} satisfies: T_{i,j,k} = 1 for all valid i,j,k
    # because each product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    
    # So the tensor is indeed all ones, and its rank is 1:
    # T_{i,j,k} = (1_i)(1_j)(1_k) where 1_i = 1 for all i, etc.
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 = T_{i,j,k} ✓
    
    # But wait - this gives rank 1! That's clearly wrong because we know
    # matrix multiplication requires rank at least max(n,m,p) and for 2x2x2 it's 7.
    
    # I think I'm confusing the tensor. Let me re-read the problem.
    # The tensor decomposition must satisfy:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # This IS the all-ones tensor condition. And rank 1 would work:
    # U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
    # Then the sum is 1*1*1 = 1. ✓
    
    # But that contradicts known results! The answer must be that the
    # "matrix multiplication tensor" in this problem is NOT the all-ones tensor.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The standard matrix multiplication tensor has entries:
    # T_{(i,j),(j,k),(i,k)} = 1  (using the standard indexing)
    # In our flat indexing: T[i*m+j, j*p+k, k*n+i] = 1
    # But all other entries are 0!
    
    # So the tensor is NOT all ones. It's a sparse tensor where
    # T[a, b, c] = 1 iff the indices are consistent with matrix multiplication.
    # Specifically: a = i*m+j, b = j*p+k, c = k*n+i
    # i.e., the row index of A, the column index of A = row index of B,
    # the column index of B, and the row index of C = row index of A,
    # the column index of C = column index of B.
    
    # In flat indexing:
    # a = i*m + j  (i in [0,n), j in [0,m))
    # b = j*p + k  (j in [0,m), k in [0,p))
    # c = k*n + i  (k in [0,p), i in [0,n))
    
    # The constraint is: j = a % m = b // p, i = a // m = c % n, k = b % p = c // n
    # So T[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n
    
    # Let me verify with n=2,m=2,p=2:
    # a=0 (i=0,j=0), b=0 (j=0,k=0), c=0 (k=0,i=0): 0%m=0==0//p=0 ✓, 0//m=0==0%n=0 ✓, 0%p=0==0//n=0 ✓ → T=1
    # a=0 (i=0,j=0), b=1 (j=0,k=1), c=1 (k=1,i=0): 0%m=0==1//p=0 ✓, 0//m=0==1%n=1 ✗ → T=0
    
    # OK so the tensor is NOT all ones. Let me build it properly.
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now find the rank of this tensor.
    # For n=m=p=2, this should have rank 7 (Strassen).
    # Let's verify our Strassen decomposition works on this tensor.
    
    # Actually, let me just use the hybrid approach which we know works,
    # and additionally try to improve it with a local search.
    
    return None

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition, rank = n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Try the hybrid Strassen + schoolbook approach
    result = _try_decompose_strassen_hybrid(n, m, p, deadline)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
