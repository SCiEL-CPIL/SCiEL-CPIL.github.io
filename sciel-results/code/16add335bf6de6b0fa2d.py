"""Rank-56 Strassen decomposition for 2x4 * 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_factors(a_idx, b_idx, c_idx):
    """
    Return the 7x8 factor triples for a single 2x2x2x2 Strassen block.
    a_idx, b_idx, c_idx are lists of 4 indices each for A, B, C respectively.
    Returns list of (u_idx_or_None, v_idx_or_None, w_entries) tuples.
    """
    a11, a12, a21, a22 = a_idx
    b11, b12, b21, b22 = b_idx
    c11, c12, c21, c22 = c_idx
    
    blocks = []
    
    # M1 = (X11+X22)*(Y11+Y22)  -> C11+=1, C22+=1
    blocks.append(([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)]))
    # M2 = (X21+X22)*Y11        -> C21+=1, C22-=1
    blocks.append(([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)]))
    # M3 = X11*(Y12-Y22)        -> C12+=1, C22+=1
    blocks.append(([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)]))
    # M4 = X22*(Y21-Y11)        -> C11+=1, C21+=1
    blocks.append(([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)]))
    # M5 = (X11+X12)*Y22        -> C11-=1, C12+=1
    blocks.append(([(a11,1),(a12,1)], [(b22,1)], [(c11,-1),(c12,1)]))
    # M6 = (X21-X11)*(Y11+Y12)  -> C22+=1
    blocks.append(([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)]))
    # M7 = (X12-X22)*(Y21+Y22)  -> C11+=1
    blocks.append(([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)]))
    
    return blocks

def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Splits into 8 independent 2x2 * 2x2 sub-blocks, each using Strassen's
    7-multiplication algorithm. Total rank = 8 * 7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56
    
    # Allocate factor matrices
    U = np.zeros((n * m, total_rank), dtype=float)   # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=float)   # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=float)   # 16 x 56
    
    # Iterate over 8 sub-blocks: u in {0,1} (A column block), v in {0..3} (B col block)
    for u in range(2):
        for v in range(4):
            if time.time() > deadline:
                # Return best result found so far (partial construction)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            base_col = 7 * (4 * u + v)
            
            # A block (2x2) at columns 2u, 2u+1
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1
            a_idx = [a11, a12, a21, a22]
            
            # B block (2x2) at rows 2u, 2u+1 and cols 2v, 2v+1
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1
            b_idx = [b11, b12, b21, b22]
            
            # C block (2x2) at cols 2v, 2v+1, with trace convention W[k*n+i, r]
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1
            c_idx = [c11, c12, c21, c22]
            
            blocks = _strassen_2x2_factors(a_idx, b_idx, c_idx)
            
            for k, (u_entries, v_entries, w_entries) in enumerate(blocks):
                col = base_col + k
                for idx, val in u_entries:
                    U[idx, col] += val
                for idx, val in v_entries:
                    V[idx, col] += val
                for idx, val in w_entries:
                    W[idx, col] += val
    
    # Convert to lists to match interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
