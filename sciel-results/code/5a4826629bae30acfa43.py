"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(ni, mj, pk):
        nonlocal col
        # Indices:
        # A: (ni, mj), (ni+1, mj), (ni, mj+1), (ni+1, mj+1)
        # B: (mj, pk), (mj, pk+1), (mj+1, pk), (mj+1, pk+1)
        # C: (pk, ni), (pk+1, ni), (pk, ni+1), (pk+1, ni+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[ni*m + mj, col] += 1
        U[(ni+1)*m + (mj+1), col] += 1
        V[mj*p + pk, col] += 1
        V[(mj+1)*p + (pk+1), col] += 1
        W[pk*n + ni, col] += 1
        W[(pk+1)*n + (ni+1), col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(ni+1)*m + mj, col] += 1
        U[(ni+1)*m + (mj+1), col] += 1
        V[mj*p + pk, col] += 1
        W[pk*n + (ni+1), col] += 1
        W[(pk+1)*n + (ni+1), col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[ni*m + mj, col] += 1
        V[(mj+1)*p + pk, col] += 1
        V[(mj+1)*p + (pk+1), col] -= 1
        W[(pk+1)*n + ni, col] += 1
        W[(pk+1)*n + (ni+1), col] += 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[(ni+1)*m + (mj+1), col] += 1
        V[mj*p + (pk+1), col] += 1
        V[mj*p + pk, col] -= 1
        W[pk*n + ni, col] += 1
        W[pk*n + (ni+1), col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[ni*m + mj, col] += 1
        U[ni*m + (mj+1), col] += 1
        V[(mj+1)*p + (pk+1), col] += 1
        W[pk*n + ni, col] -= 1
        W[(pk+1)*n + ni, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[(ni+1)*m + mj, col] += 1
        U[ni*m + mj, col] -= 1
        V[mj*p + pk, col] += 1
        V[(mj+1)*p + pk, col] += 1
        W[(pk+1)*n + (ni+1), col] += 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[ni*m + (mj+1), col] += 1
        U[(ni+1)*m + (mj+1), col] -= 1
        V[mj*p + (pk+1), col] += 1
        V[(mj+1)*p + (pk+1), col] += 1
        W[pk*n + ni, col] += 1
        col += 1

    # Helper to add rank-1 terms for 1x2x2 (n=1, m=2, p=2)
    # A[ni, mj:mj+2] * B[mj:mj+2, pk:pk+2] -> C[pk:pk+2, ni]
    def add_1x2x2(ni, mj, pk):
        nonlocal col
        for jj in range(2):
            for kk in range(2):
                U[ni*m + (mj+jj), col] = 1
                V[(mj+jj)*p + (pk+kk), col] = 1
                W[(pk+kk)*n + ni, col] = 1
                col += 1

    # Helper to add rank-1 terms for 2x1x2 (n=2, m=1, p=2)
    # A[ni:ni+2, mj] * B[mj, pk:pk+2] -> C[pk:pk+2, ni:ni+2]
    def add_2x1x2(ni, mj, pk):
        nonlocal col
        for ii in range(2):
            for kk in range(2):
                U[(ni+ii)*m + mj, col] = 1
                V[mj*p + (pk+kk), col] = 1
                W[(pk+kk)*n + (ni+ii), col] = 1
                col += 1

    # Helper to add rank-1 terms for 1x1x2 (n=1, m=1, p=2)
    # A[ni, mj] * B[mj, pk:pk+2] -> C[pk:pk+2, ni]
    def add_1x1x2(ni, mj, pk):
        nonlocal col
        for kk in range(2):
            U[ni*m + mj, col] = 1
            V[mj*p + (pk+kk), col] = 1
            W[(pk+kk)*n + ni, col] = 1
            col += 1

    # Apply Strassen blocks for n in {0,1}, m in {0,1},{2,3}, p in {0,1},{2,3},{4,5}
    for ni in [0]: # n=0,1
        for mj in [0, 2]: # m=0,1 and 2,3
            for pk in [0, 2, 4]: # p=0,1 and 2,3 and 4,5
                add_strassen(ni, mj, pk)
                
    # Apply 1x2x2 blocks for n=2, m in {0,1},{2,3}, p in {0,1},{2,3},{4,5}
    for ni in [2]:
        for mj in [0, 2]:
            for pk in [0, 2, 4]:
                add_1x2x2(ni, mj, pk)
                
    # Apply 2x1x2 blocks for n in {0,1}, m=4, p in {0,1},{2,3},{4,5}
    for ni in [0]:
        for mj in [4]:
            for pk in [0, 2, 4]:
                add_2x1x2(ni, mj, pk)
                
    # Apply 1x1x2 blocks for n=2, m=4, p in {0,1},{2,3},{4,5}
    for ni in [2]:
        for mj in [4]:
            for pk in [0, 2, 4]:
                add_1x1x2(ni, mj, pk)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
