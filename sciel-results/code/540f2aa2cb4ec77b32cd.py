"""Optimized matrix multiplication tensor decomposition for 6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Rank 189 decomposition: R(6) <= R(3)*R(2) = 27*7
    # We construct the tensor by combining the 3x3 schoolbook structure 
    # with the 2x2 Strassen structure.
    
    rank = 189
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Define Strassen 2x2 components (indices into 2x2 matrices)
    # U_strassen: coefficients for A
    # V_strassen: coefficients for B
    # W_strassen: coefficients for C
    # Each is a list of 7 lists of (row, col, coeff)
    
    sU = [
        [(0,0,1), (1,1,1)], 
        [(1,0,1), (1,1,1)], 
        [(0,0,1)],           
        [(1,1,1)],           
        [(0,0,1), (0,1,1)],  
        [(1,0,1), (0,0,-1)], 
        [(0,1,1), (1,1,-1)]  
    ]
    
    sV = [
        [(0,0,1), (1,1,1)], 
        [(0,0,1)],           
        [(0,1,1), (1,1,-1)], 
        [(1,0,1), (0,0,-1)], 
        [(1,1,1)],           
        [(0,0,1), (0,1,1)],  
        [(1,0,1), (1,1,1)]   
    ]
    
    sW = [
        [(0,0,1), (1,1,1)],   
        [(1,0,1), (1,1,-1)],  
        [(0,1,1), (1,1,1)],   
        [(0,0,1), (1,0,1)],   
        [(0,0,-1), (0,1,1)],  
        [(1,1,1)],            
        [(0,0,1)]             
    ]
    
    # Iterate over the 3x3 grid of blocks
    # i: block row (0..2), j: block col (0..2), k: block output col (0..2)
    # Wait, the 3x3 schoolbook part implies:
    # C[i][k] = sum_j A[i][j] B[j][k]
    # So we have 27 blocks in the schoolbook sum.
    # Each block multiplication is a 2x2 Strassen.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Base index for this block's 7 Strassen terms
                base_idx = (i * 9 + j * 3 + k) * 7
                
                for t in range(7):
                    col = base_idx + t
                    
                    # Update U: A[i*2..i*2+1][j*2..j*2+1]
                    for r, c, val in sU[t]:
                        row_u = (i * 2 + r) * m + (j * 2 + c)
                        U[row_u, col] += val
                        
                    # Update V: B[j*2..j*2+1][k*2..k*2+1]
                    for r, c, val in sV[t]:
                        row_v = (j * 2 + r) * p + (k * 2 + c)
                        V[row_v, col] += val
                        
                    # Update W: C[i*2..i*2+1][k*2..k*2+1]
                    # Convention: W[k*n + i, r] where i is row, k is col
                    for r, c, val in sW[t]:
                        row_w = (k * 2 + c) * n + (i * 2 + r)
                        W[row_w, col] += val
                        
    if time.time() > deadline:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
