"""Matrix multiplication 4x4x7 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

DEADLINE = time.time() + 130.0


def _construct_strassen_4x4x4():
    """
    Construct rank-56 decomposition for 4x4 * 4x4 -> 4x4 using 8 Strassen 2x2x2 blocks.
    Returns (U, V, W) each of shape (16, 56).
    """
    rank = 56
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for i in range(2):  # output block row
        for j in range(2):  # output block col
            for k in range(2):  # A block col / B block row
                # A[i,k]: rows 2i,2i+1; cols 2k,2k+1
                # B[k,j]: rows 2k,2k+1; cols 2j,2j+1
                # C[i,j]: rows 2i,2i+1; cols 2j,2j+1
                
                ar0, ar1 = 2*i, 2*i+1
                ac0, ac1 = 2*k, 2*k+1
                br0, br1 = 2*k, 2*k+1
                bc0, bc1 = 2*j, 2*j+1
                cr0, cr1 = 2*i, 2*i+1
                cc0, cc1 = 2*j, 2*j+1
                
                # M1 = (A[ar0,ac0] + A[ar1,ac1]) * (B[br0,bc0] + B[br1,bc1])
                # C11 += M1, C22 += M1
                U[ar0, col] = 1; U[ar1, col] = 1
                V[br0, col] = 1; V[br1, col] = 1
                W[cr0, cc0] = 1; W[cr1, cc1] = 1
                col += 1
                
                # M2 = (A[ar1,ac0] + A[ar1,ac1]) * B[br0,bc0]
                # C21 += M2, C22 -= M2
                U[ar1, col] = 1; U[ar1, col] += 1
                V[br0, col] = 1
                W[cr1, cc0] = 1; W[cr1, cc1] = -1
                col += 1
                
                # M3 = A[ar0,ac0] * (B[br0,bc1] - B[br1,bc1])
                # C12 += M3, C22 += M3
                U[ar0, col] = 1
                V[br0, col] = 1; V[br1, col] = -1
                W[cr0, cc1] = 1; W[cr1, cc1] = 1
                col += 1
                
                # M4 = A[ar1,ac1] * (B[br1,bc0] - B[br0,bc0])
                # C11 += M4, C21 += M4
                U[ar1, col] = 1
                V[br1, col] = 1; V[br0, col] = -1
                W[cr0, cc0] = 1; W[cr1, cc0] = 1
                col += 1
                
                # M5 = (A[ar0,ac0] + A[ar0,ac1]) * B[br1,bc1]
                # C11 -= M5, C12 += M5
                U[ar0, col] = 1; U[ar0, col] += 1
                V[br1, col] = 1
                W[cr0, cc0] = -1; W[cr0, cc1] = 1
                col += 1
                
                # M6 = (A[ar1,ac0] - A[ar0,ac0]) * (B[br0,bc0] + B[br0,bc1])
                # C22 += M6
                U[ar1, col] = 1; U[ar0, col] = -1
                V[br0, col] = 1; V[br0, col] += 1
                W[cr1, cc1] = 1
                col += 1
                
                # M7 = (A[ar0,ac1] - A[ar1,ac1]) * (B[br1,bc0] + B[br1,bc1])
                # C11 += M7
                U[ar0, col] = 1; U[ar1, col] = -1
                V[br1, col] = 1; V[br1, col] += 1
                W[cr0, cc0] = 1
                col += 1
                
    assert col == rank
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    rank1 = 56  # for 4x4x4 part
    rank2 = 48  # for 4x4x3 schoolbook part
    total_rank = rank1 + rank2  # 104
    
    U = np.zeros((n*m, total_rank))  # 16 x 104
    V = np.zeros((m*p, total_rank))  # 28 x 104
    W = np.zeros((p*n, total_rank))  # 28 x 104
    
    # Part 1: C1 = A * B1 (4x4 * 4x4 -> 4x4)
    U1, V1, W1 = _construct_strassen_4x4x4()
    U[:, :rank1] = U1
    V[0:16, :rank1] = V1  # B1 is first 4 columns of B (4x4)
    W[0:16, :rank1] = W1  # C1 is first 4 rows of C (4x4)
    
    # Part 2: C2 = A * B2 (4x4 * 4x3 -> 4x3)
    col = rank1
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j, col] = 1
                V[j*p + (4+k), col] = 1  # B[j, 4+k]
                W[(4+k)*n + i, col] = 1  # C[4+k, i]
                col += 1
    
    assert col == total_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
