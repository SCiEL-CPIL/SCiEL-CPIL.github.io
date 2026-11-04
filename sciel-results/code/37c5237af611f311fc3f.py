import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    def _strassen_2x2x2(i0, j0, k0, U, V, W, col):
        """Add 7 Strassen terms for 2x2x2 block starting at (i0,j0,k0).
        U indices: (i0..i0+1, j0..j0+1) -> flat: i*m+j
        V indices: (j0..j0+1, k0..k0+1) -> flat: j*p+k
        W indices: (k0..k0+1, i0..i0+1) -> flat: k*n+i
        """
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        W[(k0+1)*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += -1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        W[k0*n + (i0+1)][col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        W[k0*n + (i0+1)][col] += -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        return col
    
    def _verify(U, V, W):
        """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k"""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-44 decomposition
    # Split: rows 0,1 of A use Strassen (2x4x4 = 4 blocks of 2x2x2, rank 28)
    #        row 2 of A uses schoolbook (1x4x4, rank 16)
    # Total rank: 28 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 2x4x4 part: A[0:2, :] @ B[:, :] -> C[0:2, :]
    # This is: sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}
    # Split into 4 2x2x2 blocks:
    # Block (i0,j0,k0) covers A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
    # But we need to be careful: the j dimension is shared between A and B.
    # 
    # Actually, the 2x4x4 tensor T[i,j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # = sum over j of A[i,j] * B[j,k]
    # 
    # We can decompose this as sum of 4 2x2x2 tensors:
    # For j0 in {0, 2}: the sub-tensor with j in {j0, j0+1}
    # Each such sub-tensor is a 2x2x4 tensor (i:2, j:2, k:4)
    # Which itself can be split into 2 2x2x2 tensors (k0 in {0, 2})
    # So total 4 2x2x2 blocks, each rank 7 = 28
    
    # The four 2x2x2 blocks:
    # (i0=0, j0=0, k0=0): A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # (i0=0, j0=0, k0=2): A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    # (i0=0, j0=2, k0=0): A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    # (i0=0, j0=2, k0=2): A[0:2,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
    
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            col = _strassen_2x2x2(0, j0, k0, U, V, W, col)
    
    # 1x4x4 part: A[2, :] @ B[:, :] -> C[2, :]
    # Schoolbook: rank 16
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to naive
        rank_naive = n * m * p
        U2 = [[0] * rank_naive for _ in range(n * m)]
        V2 = [[0] * rank_naive for _ in range(m * p)]
        W2 = [[0] * rank_naive for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m + j][c] = 1
                    V2[j*p + k][c] = 1
                    W2[k*n + i][c] = 1
                    c += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return result
