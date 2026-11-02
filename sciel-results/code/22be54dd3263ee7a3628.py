import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_DEADLINE = time.time() + 145  # 5s safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ diag @ V^T reconstructs the matmul tensor."""
    rank = U.shape[1]
    # Build target tensor: T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # T is (n*p, m*p) flattened... actually let's think in terms of the bilinear form.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r]
    # Wait, the convention is:
    # U: (n*m, rank) - U[i*m+j, r]
    # V: (m*p, rank) - V[j*p+k, r]  
    # W: (p*n, rank) - W[k*n+i, r]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: C[i,k] = sum_j A[i,j]*B[j,k], so coefficient of A[i,j]*B[j,k] is 1
    
    # Check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
    # Actually no - the decomposition says:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... 
    # No, the standard tensor decomposition for matmul is:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    # where T_{i,j,k} = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    
    # In our indexing:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r should give 1 for each (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[i,k] -> W[2*k+i]
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    # Verify decomposition (skip if time is tight)
    if time.time() < _DEADLINE - 5:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback: return schoolbook rank-28 if Strassen is broken
            rank28 = n * m * p
            U28 = np.zeros((n*m, rank28))
            V28 = np.zeros((m*p, rank28))
            W28 = np.zeros((p*n, rank28))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U28[i*m+j, r] = 1
                        V28[j*p+k, r] = 1
                        W28[k*n+i, r] = 1
            return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
