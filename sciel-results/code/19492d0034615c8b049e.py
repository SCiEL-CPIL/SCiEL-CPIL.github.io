"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Strategy: Use block partitioning with Strassen for 2x2 sub-blocks and search
for low-rank decompositions of rectangular sub-problems.

Partition:
A (4x5) -> rows split 2+2, cols split 2+3
  A11(2x2), A12(2x3), A21(2x2), A22(2x3)
B (5x5) -> rows split 2+3, cols split 2+3
  B11(2x2), B12(2x3), B21(3x2), B22(3x3)
C (4x5) -> rows split 2+2, cols split 2+3
  C11(2x2), C12(2x3), C21(2x2), C22(2x3)

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Use Strassen (rank 7) for 2x2*2x2, and search for low-rank algorithms
for the rectangular products.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
    
    Returns U, V, W tensors where:
    U has shape (4, 7) - extracts A elements
    V has shape (4, 7) - extracts B elements  
    W has shape (4, 7) - places C elements
    
    A = [[a11, a12], [a21, a22]] indexed as A[0]=a11, A[1]=a12, A[2]=a21, A[3]=a22
    B = [[b11, b12], [b21, b22]] indexed as B[0]=b11, B[1]=b12, B[2]=b21, B[3]=b22
    C = [[c11, c12], [c21, c22]] indexed as C[0]=c11, C[1]=c21, C[2]=c12, C[3]=c22
    (column-major for output: C[k*n+i,r] where n=p=2)
    
    Wait, the output convention is W[k*n+i, r] where C is n x p.
    For 2x2: C[0]=C[0,0], C[1]=C[0,1], C[2]=C[1,0], C[3]=C[1,1]
    So W index: k*n+i where k is col, i is row.
    C[0,0]: k=0,i=0 -> idx 0
    C[0,1]: k=1,i=0 -> idx 1
    C[1,0]: k=0,i=1 -> idx 2
    C[1,1]: k=1,i=1 -> idx 3
    """
    # Strassen's 7 products:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: 0=a11, 1=a12, 2=a21, 3=a22
    # B indices: 0=b11, 1=b12, 2=b21, 3=b22
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)*(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c11 += M1, c22 += M1
    
    # M2 = (a21+a22)*b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c21 += M2, c22 -= M2
    
    # M3 = a11*(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c12 += M3, c22 += M3
    
    # M4 = a22*(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # c11 += M4, c21 += M4
    
    # M5 = (a11+a12)*b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c11 -= M5, c12 += M5
    
    # M6 = (a21-a11)*(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c22 += M6
    
    # M7 = (a12-a22)*(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c11 += M7
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct.
    
    For the tensor T[i,j,k] = delta_{j,k} (where A is n x m, B is m x p, C is n x p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if j==k else 0.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple sub-decompositions into one for the full problem.
    
    Each decomp is (U_sub, V_sub, W_sub, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols)
    where the sub-problem computes C_sub = A_sub * B_sub.
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for U_sub, V_sub, W_sub, a_r, a_c, b_r, b_c, c_r, c_c in decomps:
        sub_rank = U_sub.shape[1]
        for ri in range(U_sub.shape[0]):
            i = a_r + ri // a_c
            j = a_c + (ri % a_c) if a_c > 0 else 0
            # Actually need to map sub-index to global
            # U_sub has shape (a_r*a_c, sub_rank) where index = local_i * a_c + local_j
            pass
        
        # Redo with proper mapping
        for ri in range(U_sub.shape[0]):
            local_i = ri // a_c
            local_j = ri % a_c
            gi = a_r + local_i
            gj = a_c + local_j
            for r in range(sub_rank):
                U[gi*m + gj, col_offset + r] = U_sub[ri, r]
        
        for ri in range(V_sub.shape[0]):
            local_j = ri // b_c
            local_k = ri % b_c
            gj = b_r + local_j
            gk = b_c + local_k
            for r in range(sub_rank):
                V[gj*p + gk, col_offset + r] = V_sub[ri, r]
        
        for ri in range(W_sub.shape[0]):
            local_k = ri // c_r
            local_i = ri % c_r
            gk = c_r + local_k
            gi = c_c + local_i
            for r in range(sub_rank):
                W[gk*n + gi, col_offset + r] = W_sub[ri, r]
        
        col_offset += sub_rank
    
    return U, V, W

def _build_block_decomposition():
    """Build decomposition using Strassen for 2x2 blocks and schoolbook for rest.
    
    Partition:
    A (4x5): rows 0-1, 2-3; cols 0-1, 2-4
    B (5x5): rows 0-1, 2-4; cols 0-1, 2-4
    C (4x5): rows 0-1, 2-3; cols 0-1, 2-4
    
    C11(2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    C12(2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    C21(2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    C22(2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
    """
    n, m, p = 4, 5, 5
    
    # Get Strassen decomposition for 2x2
    S_U, S_V, S_W = _strassen_2x2()
    
    decomps = []
    
    # C11 = A11*B11 + A12*B21
    # Term 1: A11(2x2) * B11(2x2) -> C11(2x2), use Strassen
    decomps.append((S_U, S_V, S_W, 0, 0, 0, 0, 0, 0))  # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    
    # Term 2: A12(2x3) * B21(3x2) -> C11(2x2), schoolbook rank 12
    U12_B21 = np.zeros((6, 12))
    V12_B21 = np.zeros((6, 12))
    W12_B21 = np.zeros((4, 12))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U12_B21[i*3+j, col] = 1
                V12_B21[j*2+k, col] = 1
                W12_B21[k*2+i, col] = 1
                col += 1
    decomps.append((U12_B21, V12_B21, W12_B21, 0, 2, 2, 2, 0, 0))
    
    # C12 = A11*B12 + A12*B22
    # Term 1: A11(2x2) * B12(2x3) -> C12(2x3), schoolbook rank 12
    U11_B12 = np.zeros((4, 12))
    V11_B12 = np.zeros((6, 12))
    W11_B12 = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U11_B12[i*2+j, col] = 1
                V11_B12[j*3+k, col] = 1
                W11_B12[k*2+i, col] = 1
                col += 1
    decomps.append((U11_B12, V11_B12, W11_B12, 0, 0, 0, 2, 0, 2))
    
    # Term 2: A12(2x3) * B22(3x3) -> C12(2x3), schoolbook rank 18
    U12_B22 = np.zeros((6, 18))
    V12_B22 = np.zeros((9, 18))
    W12_B22 = np.zeros((6, 18))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U12_B22[i*3+j, col] = 1
                V12_B22[j*3+k, col] = 1
                W12_B22[k*2+i, col] = 1
                col += 1
    decomps.append((U12_B22, V12_B22, W12_B22, 0, 2, 2, 2, 0, 2))
    
    # C21 = A21*B11 + A22*B21
    # Term 1: A21(2x2) * B11(2x2) -> C21(2x2), use Strassen
    decomps.append((S_U, S_V, S_W, 2, 0, 0, 0, 2, 0))
    
    # Term 2: A22(2x3) * B21(3x2) -> C21(2x2), schoolbook rank 12
    U22_B21 = np.zeros((6, 12))
    V22_B21 = np.zeros((6, 12))
    W22_B21 = np.zeros((4, 12))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U22_B21[i*3+j, col] = 1
                V22_B21[j*2+k, col] = 1
                W22_B21[k*2+i, col] = 1
                col += 1
    decomps.append((U22_B21, V22_B21, W22_B21, 2, 2, 2, 2, 2, 0))
    
    # C22 = A21*B12 + A22*B22
    # Term 1: A21(2x2) * B12(2x3) -> C22(2x3), schoolbook rank 12
    U21_B12 = np.zeros((4, 12))
    V21_B12 = np.zeros((6, 12))
    W21_B12 = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U21_B12[i*2+j, col] = 1
                V21_B12[j*3+k, col] = 1
                W21_B12[k*2+i, col] = 1
                col += 1
    decomps.append((U21_B12, V21_B12, W21_B12, 2, 0, 0, 2, 2, 2))
    
    # Term 2: A22(2x3) * B22(3x3) -> C22(2x3), schoolbook rank 18
    U22_B22 = np.zeros((6, 18))
    V22_B22 = np.zeros((9, 18))
    W22_B22 = np.zeros((6, 18))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U22_B22[i*3+j, col] = 1
                V22_B22[j*3+k, col] = 1
                W22_B22[k*2+i, col] = 1
                col += 1
    decomps.append((U22_B22, V22_B22, W22_B22, 2, 2, 2, 2, 2, 2))
    
    # Total rank: 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try block decomposition with Strassen
    try:
        U, V, W = _build_block_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            rank = U.shape[1]
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
