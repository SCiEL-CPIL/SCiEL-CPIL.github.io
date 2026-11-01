"""Matrix multiplication 2x2 by 2x7 - rank-25 exact tensor decomposition using block-Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns (0,1), (2,3), (4,5)
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank))  # (4, 25)
    V = np.zeros((m * p, rank))  # (14, 25)
    W = np.zeros((p * n, rank))  # (14, 25)
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
