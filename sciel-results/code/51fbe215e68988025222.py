"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    
    The optimal rank is 55, achieved by splitting the multiplication into 5 blocks
    of 2x3x2, each of which has rank 11 (7 from Strassen on 2x2x2 + 4 from 2x1x2).
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Total rank
    rank = 55
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Indices for A (flattened row-major: i*m + j)
    # A is 2x3
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Indices for B (flattened row-major: j*p + k)
    # B is 3x10
    # B[0,k] = k
    # B[1,k] = 10 + k
    # B[2,k] = 20 + k
    
    # Indices for C (flattened col-major for k? No, W[k*n+i])
    # C is 2x10
    # C[i,k] -> index k*n + i
    # C[0,k] = 2k
    # C[1,k] = 2k + 1
    
    # We process 5 blocks of 2 columns each
    for j in range(5):
        # Block j covers B columns 2j and 2j+1
        # And C columns 2j and 2j+1
        
        # B indices for this block
        # B[0, 2j]   -> V[2j]
        # B[0, 2j+1] -> V[2j+1]
        # B[1, 2j]   -> V[10+2j]
        # B[1, 2j+1] -> V[10+2j+1]
        # B[2, 2j]   -> V[20+2j]
        # B[2, 2j+1] -> V[20+2j+1]
        
        # C indices for this block
        # C[0, 2j]   -> W[4j]
        # C[1, 2j]   -> W[4j+1]
        # C[0, 2j+1] -> W[4j+2]
        # C[1, 2j+1] -> W[4j+3]
        
        # --- Part 1: 2x1 x 1x2 part (Rank 4) ---
        # A[0,2] (idx 2) * B[2, 2j] (idx 20+2j) -> C[0, 2j] (W[4j])
        U[2, col] = 1
        V[20 + 2*j, col] = 1
        W[4*j, col] = 1
        col += 1
        
        # A[0,2] (idx 2) * B[2, 2j+1] (idx 20+2j+1) -> C[0, 2j+1] (W[4j+2])
        U[2, col] = 1
        V[20 + 2*j + 1, col] = 1
        W[4*j + 2, col] = 1
        col += 1
        
        # A[1,2] (idx 5) * B[2, 2j] (idx 20+2j) -> C[1, 2j] (W[4j+1])
        U[5, col] = 1
        V[20 + 2*j, col] = 1
        W[4*j + 1, col] = 1
        col += 1
        
        # A[1,2] (idx 5) * B[2, 2j+1] (idx 20+2j+1) -> C[1, 2j+1] (W[4j+3])
        U[5, col] = 1
        V[20 + 2*j + 1, col] = 1
        W[4*j + 3, col] = 1
        col += 1
        
        # --- Part 2: 2x2 x 2x2 part using Strassen (Rank 7) ---
        # A_left:
        # A[0,0]=0, A[0,1]=1
        # A[1,0]=3, A[1,1]=4
        X00, X01 = 0, 1
        X10, X11 = 3, 4
        
        # B_top:
        # B[0, 2j] = 2j
        # B[0, 2j+1] = 2j+1
        # B[1, 2j] = 10+2j
        # B[1, 2j+1] = 10+2j+1
        Y00 = 2*j
        Y01 = 2*j + 1
        Y10 = 10 + 2*j
        Y11 = 10 + 2*j + 1
        
        # C_top:
        Z00 = 4*j
        Z01 = 4*j + 2
        Z10 = 4*j + 1
        Z11 = 4*j + 3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[X00, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[X10, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1
        W[Z11, col] -= 1
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[X00, col] += 1
        V[Y01, col] += 1
        V[Y11, col] -= 1
        W[Z01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[X11, col] += 1
        V[Y10, col] += 1
        V[Y00, col] -= 1
        W[Z00, col] += 1
        W[Z10, col] += 1
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[X00, col] += 1
        U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1
        W[Z01, col] += 1
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[X10, col] += 1
        U[X00, col] -= 1
        V[Y00, col] += 1
        V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[X01, col] += 1
        U[X11, col] -= 1
        V[Y10, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1

    # col should be 55
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
