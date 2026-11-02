import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Split the 7 columns of B into three pairs (0,1), (2,3), (4,5) and one single column (6).
    - Use Strassen's rank-7 algorithm for each 2x2 block multiplication (A * B_pair).
    - Use schoolbook (rank 4) for the single column multiplication (A * B_single).
    - Total rank: 3 * 7 + 4 = 25.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=int)  # (4, 25)
    V = np.zeros((m * p, rank), dtype=int)  # (14, 25)
    W = np.zeros((p * n, rank), dtype=int)  # (14, 25)
    
    # Indexing:
    # A[i,j] -> U[i*m + j] = U[2*i + j]
    # B[j,k] -> V[j*p + k] = V[7*j + k]
    # C[i,k] -> W[k*n + i] = W[2*k + i]
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        c0 = col_start
        c1 = col_start + 1
        c2 = col_start + 2
        c3 = col_start + 3
        c4 = col_start + 4
        c5 = col_start + 5
        c6 = col_start + 6
        
        ks = k_start
        ke = k_start + 1
        
        # M1 = (A00 + A11) * (B0,ks + B1,ke)
        # C0,ks += M1, C1,ke += M1
        U[0, c0] += 1
        U[3, c0] += 1
        V[7 * 0 + ks, c0] += 1
        V[7 * 1 + ke, c0] += 1
        W[2 * ks + 0, c0] += 1
        W[2 * ke + 1, c0] += 1
        
        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ke -= M2
        U[2, c1] += 1
        U[3, c1] += 1
        V[7 * 0 + ks, c1] += 1
        W[2 * ks + 1, c1] += 1
        W[2 * ke + 1, c1] -= 1
        
        # M3 = A00 * (B0,ke - B1,ke)
        # C0,ke += M3, C1,ke += M3
        U[0, c2] += 1
        V[7 * 0 + ke, c2] += 1
        V[7 * 1 + ke, c2] -= 1
        W[2 * ke + 0, c2] += 1
        W[2 * ke + 1, c2] += 1
        
        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, c3] += 1
        V[7 * 1 + ks, c3] += 1
        V[7 * 0 + ks, c3] -= 1
        W[2 * ks + 0, c3] += 1
        W[2 * ks + 1, c3] += 1
        
        # M5 = (A00 + A01) * B1,ke
        # C0,ks -= M5, C0,ke += M5
        U[0, c4] += 1
        U[1, c4] += 1
        V[7 * 1 + ke, c4] += 1
        W[2 * ks + 0, c4] -= 1
        W[2 * ke + 0, c4] += 1
        
        # M6 = (A10 - A00) * (B0,ks + B0,ke)
        # C1,ke += M6
        U[2, c5] += 1
        U[0, c5] -= 1
        V[7 * 0 + ks, c5] += 1
        V[7 * 0 + ke, c5] += 1
        W[2 * ke + 1, c5] += 1
        
        # M7 = (A01 - A11) * (B1,ks + B1,ke)
        # C0,ks += M7
        U[1, c6] += 1
        U[3, c6] -= 1
        V[7 * 1 + ks, c6] += 1
        V[7 * 1 + ke, c6] += 1
        W[2 * ks + 0, c6] += 1

    # Add Strassen for columns (0,1)
    add_strassen(0, 0)
    # Add Strassen for columns (2,3)
    add_strassen(7, 2)
    # Add Strassen for columns (4,5)
    add_strassen(14, 4)
    
    # Schoolbook for column 6 (4 terms)
    col_start = 21
    k6 = 6
    
    # A00 * B0,6 -> C0,6
    U[0, col_start + 0] += 1
    V[7 * 0 + k6, col_start + 0] += 1
    W[2 * k6 + 0, col_start + 0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start + 1] += 1
    V[7 * 1 + k6, col_start + 1] += 1
    W[2 * k6 + 0, col_start + 1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start + 2] += 1
    V[7 * 0 + k6, col_start + 2] += 1
    W[2 * k6 + 1, col_start + 2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start + 3] += 1
    V[7 * 1 + k6, col_start + 3] += 1
    W[2 * k6 + 1, col_start + 3] += 1
    
    # Verify the decomposition
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # A is 2x2, B is 2x7, C is 2x7
    # A[i,j] -> U[2*i+j, r]
    # B[j,k] -> V[7*j+k, r]
    # C[i,k] -> W[2*k+i, r]
    
    # Check that the decomposition correctly represents matrix multiplication
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i, j, k] should be 1 if we're multiplying A[i,j] * B[j,k] -> C[i,k]
    # But our tensor is structured differently...
    
    # Actually, let's verify by computing a sample product
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.array([[5, 6, 7, 8, 9, 10, 11],
                       [12, 13, 14, 15, 16, 17, 18]], dtype=float)
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            C_val = 0.0
            for r in range(rank):
                a_coeff = 0.0
                for j in range(m):
                    a_coeff += A_test[i, j] * U_mat[2*i + j, r]
                b_coeff = 0.0
                for j in range(m):
                    b_coeff += B_test[j, k] * V_mat[7*j + k, r]
                C_val += a_coeff * b_coeff * W_mat[2*k + i, r]
            C_computed[i, k] = C_val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
