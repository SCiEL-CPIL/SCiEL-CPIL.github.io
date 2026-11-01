import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 5
    rank = 28
    
    # Initialize matrices as zero arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(u_idx, v_idx, w_idx, val, col_idx):
        if u_idx is not None:
            U[u_idx, col_idx] = val
        if v_idx is not None:
            V[v_idx, col_idx] = val
        if w_idx is not None:
            W[w_idx, col_idx] = val
            
    def add_strassen_block(i_off, j_off, k_off, start_col):
        """
        Adds a 2x2 matrix multiplication block (Strassen-like or standard) 
        using the provided offsets. 
        i_off: offset for first dimension (0..1)
        j_off: offset for middle dimension (0..1)
        k_off: offset for last dimension (0..1)
        
        Note: The standard 2x2 matrix multiplication tensor has rank 7.
        Indices are local 0,1 for i,j,k.
        """
        c = start_col
        
        # Term 1: (i0+j0)*(j0+k0) -> (i0+k0) and (i1+k1)
        # Actually, let's stick to the specific terms used in the previous working solution to ensure correctness.
        # The previous solution used a specific set of 7 terms. I will replicate them exactly.
        
        # M1
        U[(0 + i_off) * m + (0 + j_off), c] = 1
        U[(1 + i_off) * m + (1 + j_off), c] = 1
        V[(0 + j_off) * p + (0 + k_off), c] = 1
        V[(1 + j_off) * p + (1 + k_off), c] = 1
        W[(0 + k_off) * n + (0 + i_off), c] = 1
        W[(1 + k_off) * n + (1 + i_off), c] = 1
        c += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off), c] = 1
        U[(1 + i_off) * m + (1 + j_off), c] = 1
        V[(0 + j_off) * p + (0 + k_off), c] = 1
        W[(0 + k_off) * n + (1 + i_off), c] = 1
        W[(1 + k_off) * n + (1 + i_off), c] = -1
        c += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off), c] = 1
        V[(0 + j_off) * p + (1 + k_off), c] = 1
        V[(1 + j_off) * p + (1 + k_off), c] = -1
        W[(1 + k_off) * n + (0 + i_off), c] = 1
        W[(1 + k_off) * n + (1 + i_off), c] = 1
        c += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off), c] = 1
        V[(1 + j_off) * p + (0 + k_off), c] = 1
        V[(0 + j_off) * p + (0 + k_off), c] = -1
        W[(0 + k_off) * n + (0 + i_off), c] = 1
        W[(0 + k_off) * n + (1 + i_off), c] = 1
        c += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off), c] = 1
        U[(0 + i_off) * m + (1 + j_off), c] = 1
        V[(1 + j_off) * p + (1 + k_off), c] = 1
        W[(0 + k_off) * n + (0 + i_off), c] = -1
        W[(1 + k_off) * n + (0 + i_off), c] = 1
        c += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off), c] = 1
        U[(0 + i_off) * m + (0 + j_off), c] = -1
        V[(0 + j_off) * p + (0 + k_off), c] = 1
        V[(0 + j_off) * p + (1 + k_off), c] = 1
        W[(1 + k_off) * n + (1 + i_off), c] = 1
        c += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off), c] = 1
        U[(1 + i_off) * m + (1 + j_off), c] = -1
        V[(1 + j_off) * p + (0 + k_off), c] = 1
        V[(1 + j_off) * p + (1 + k_off), c] = 1
        W[(0 + k_off) * n + (0 + i_off), c] = 1
        c += 1
        
        return c

    col = 0
    
    # Block 1: j=0 (2x1 * 1x5)
    # Rank = 2 * 1 * 5 = 10
    # Indices: i in [0,1], k in [0..4], j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2: j=1,2 (2x2 * 2x5)
    # We split the 5 columns into 2, 2, and 1.
    # Part A: k=0,1 (j=1,2) -> 2x2 * 2x2. Rank 7.
    col = add_strassen_block(0, 1, 0, col)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part B: k=2,3 (j=1,2) -> 2x2 * 2x2. Rank 7.
    col = add_strassen_block(0, 1, 2, col)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part C: k=4 (j=1,2) -> 2x2 * 2x1. Rank 4.
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
