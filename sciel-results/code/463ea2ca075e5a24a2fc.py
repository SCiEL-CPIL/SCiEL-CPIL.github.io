import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition strategy.
    # Split the 9 columns of B into four blocks of size 2 and one block of size 1.
    # Block sizes: 2, 2, 2, 2, 1.
    # For each block of size s, we compute 2x3 x s multiplication.
    
    # We define helper functions to generate U, V, W for sub-problems.
    
    # Sub-problem 1: 2x2 x 2 (Strassen, rank 7)
    # A is 2x2, B is 2x2, C is 2x2
    # U indices: i*m + j -> i*2 + j (0..3)
    # V indices: j*p + k -> j*2 + k (0..3)
    # W indices: k*n + i -> k*2 + i (0..3)
    
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        # (A00 + A11) * (B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # Contribution to C:
        # C00 += M1, C11 += M1
        W[0, 0] = 1 # C00 (k=0, i=0 -> 0*2+0=0)
        W[3, 0] = 1 # C11 (k=1, i=1 -> 1*2+1=3)
        
        # M2
        # (A10 + A11) * B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2
        W[2, 1] = 1 # C10 (k=0, i=1 -> 0*2+1=2)
        
        # M3
        # A00 * (B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3
        W[1, 2] = 1 # C01 (k=1, i=0 -> 1*2+0=2? No. k=1, i=0 -> 1*2+0=2? 
        # W index is k*n + i. n=2.
        # C01 is i=0, k=1. Index = 1*2 + 0 = 2.
        # Wait, W[2, 1] was C10. C10 is i=1, k=0. Index = 0*2 + 1 = 1? No.
        # Let's re-verify W index.
        # W[k*n + i, r].
        # C[i, k].
        # C00: i=0, k=0. idx = 0*2+0=0.
        # C01: i=0, k=1. idx = 1*2+0=2.
        # C10: i=1, k=0. idx = 0*2+1=1.
        # C11: i=1, k=1. idx = 1*2+1=3.
        
        # Correction:
        # M1 contributes to C00 (idx 0) and C11 (idx 3).
        # M2 contributes to C10 (idx 1).
        # M3 contributes to C01 (idx 2).
        # M4 contributes to C00 (idx 0) and C10 (idx 1).
        # M5 contributes to C00 (idx 0) and C01 (idx 2).
        # M6 contributes to C11 (idx 3).
        # M7 contributes to C00 (idx 0) and C11 (idx 3).
        
        W[:, 0] = [1, 0, 0, 1] # M1: C00, C11
        W[:, 1] = [0, 1, 0, 0] # M2: C10
        W[:, 2] = [0, 0, 1, 0] # M3: C01
        
        # M4
        # A11 * (B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 += M4, C10 += M4
        W[:, 3] = [1, 1, 0, 0]
        
        # M5
        # (A00 + A01) * B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 -= M5, C01 += M5
        W[:, 4] = [-1, 0, 1, 0]
        
        # M6
        # (A10 - A00) * (B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += M6
        W[:, 5] = [0, 0, 0, 1]
        
        # M7
        # (A01 - A11) * (B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 += M7, C11 += M7
        W[:, 6] = [1, 0, 0, 1]
        
        return U, V, W

    # Sub-problem 2: 2x1 x s (Schoolbook, rank 2s)
    # A is 2x1 (col j=2), B is 1xs (row j=2), C is 2xs
    # U indices: i*3 + 2 (0, 1) -> 2, 5
    # V indices: 2*p + k (0..s-1) -> 2p, 2p+1, ...
    # W indices: k*2 + i
    
    def get_schoolbook_2x1xs(s, p_offset=0, i_offset=0, k_offset=0):
        # This function is for the general case where we need to place these in the larger matrix.
        # But for now, let's just generate the local blocks and assemble.
        pass

    # We will assemble the full U, V, W.
    # Total rank = 4 * 11 + 1 * 6 = 50.
    
    U = np.zeros((n*m, 50))
    V = np.zeros((m*p, 50))
    W = np.zeros((p*n, 50))
    
    col = 0
    
    # 4 blocks of size 2
    for block_idx in range(4):
        # Get Strassen 2x2x2
        Us, Vs, Ws = get_strassen_2x2x2()
        
        # Map to global indices
        # A indices: i*3 + j. j is 0,1 for Strassen.
        # U_strassen indices: i*2 + j.
        # Mapping: u_idx -> i*3 + j.
        # i = u_idx // 2, j = u_idx % 2.
        for u_idx in range(4):
            i = u_idx // 2
            j = u_idx % 2
            U[i*3 + j, col:col+7] += Us[u_idx, :]
            
        # B indices: j*9 + k. j is 0,1. k is 0,1.
        # V_strassen indices: j*2 + k.
        # Mapping: v_idx -> j*9 + (k + block_idx*2).
        for v_idx in range(4):
            j = v_idx // 2
            k = v_idx % 2
            V[j*9 + (k + block_idx*2), col:col+7] += Vs[v_idx, :]
            
        # C indices: k*2 + i. k is 0,1. i is 0,1.
        # W_strassen indices: k*2 + i.
        # Mapping: w_idx -> (k + block_idx*2)*2 + i.
        for w_idx in range(4):
            k = w_idx // 2
            i = w_idx % 2
            W[(k + block_idx*2)*2 + i, col:col+7] += Ws[w_idx, :]
            
        col += 7
        
        # Add Schoolbook 2x1x2 (Rank 4)
        # A indices: i*3 + 2.
        # B indices: 2*9 + (k + block_idx*2).
        # C indices: (k + block_idx*2)*2 + i.
        
        # Term 1: A02 B2k0 -> C0k0
        U[0*3 + 2, col] = 1
        V[2*9 + (0 + block_idx*2), col] = 1
        W[(0 + block_idx*2)*2 + 0, col] = 1
        col += 1
        
        # Term 2: A02 B2k1 -> C0k1
        U[0*3 + 2, col] = 1
        V[2*9 + (1 + block_idx*2), col] = 1
        W[(1 + block_idx*2)*2 + 0, col] = 1
        col += 1
        
        # Term 3: A12 B2k0 -> C1k0
        U[1*3 + 2, col] = 1
        V[2*9 + (0 + block_idx*2), col] = 1
        W[(0 + block_idx*2)*2 + 1, col] = 1
        col += 1
        
        # Term 4: A12 B2k1 -> C1k1
        U[1*3 + 2, col] = 1
        V[2*9 + (1 + block_idx*2), col] = 1
        W[(1 + block_idx*2)*2 + 1, col] = 1
        col += 1

    # 1 block of size 1
    # Schoolbook 2x2x1 (Rank 4)
    # A indices: i*3 + j (j=0,1)
    # B indices: j*9 + 8 (k=8)
    # C indices: 8*2 + i
    
    # Term 1: A00 B08 -> C08
    U[0*3 + 0, col] = 1
    V[0*9 + 8, col] = 1
    W[8*2 + 0, col] = 1
    col += 1
    
    # Term 2: A01 B18 -> C08
    U[0*3 + 1, col] = 1
    V[1*9 + 8, col] = 1
    W[8*2 + 0, col] = 1
    col += 1
    
    # Term 3: A10 B08 -> C18
    U[1*3 + 0, col] = 1
    V[0*9 + 8, col] = 1
    W[8*2 + 1, col] = 1
    col += 1
    
    # Term 4: A11 B18 -> C18
    U[1*3 + 1, col] = 1
    V[1*9 + 8, col] = 1
    W[8*2 + 1, col] = 1
    col += 1
    
    # Schoolbook 2x1x1 (Rank 2)
    # Term 5: A02 B28 -> C08
    U[0*3 + 2, col] = 1
    V[2*9 + 8, col] = 1
    W[8*2 + 0, col] = 1
    col += 1
    
    # Term 6: A12 B28 -> C18
    U[1*3 + 2, col] = 1
    V[2*9 + 8, col] = 1
    W[8*2 + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
