import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # The problem is to decompose the matrix multiplication tensor for 
    # A (n x m) * B (m x p) = C (n x p)
    # Here n=2, m=3, p=8.
    # The tensor is a direct sum of 4 blocks, each corresponding to a 
    # 2x3 by 3x2 multiplication.
    # Each block has minimal rank 11 (Strassen-like for 2x2x2x2 + 4 tail terms).
    # Total rank = 4 * 11 = 44.
    
    rank = 44
    
    # Initialize U, V, W as zero arrays
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 11 rank-1 terms for the 2x3 x 3x2 block
    # A indices: i in 0..1, k in 0..2. Flat index: i*m + k
    # B indices: k in 0..2, j in 0..1. Flat index: k*p + j (but j is local 0..1 here, so k*2 + j_local? No, global j)
    # In the block, j ranges from 2*b to 2*b+1.
    # Let's define the local indices for the block.
    # A_local: 0..5 (2x3)
    # B_local: 0..5 (3x2) -> k in 0..2, j_local in 0..1. Index k*2 + j_local.
    # W_local: 0..3 (2x2) -> j_local * n + i.
    
    # We map global indices to local ones inside the block loop.
    
    # The 11 terms (U_local, V_local, W_local)
    # Note: U_local corresponds to A[2,3], V_local to B[3,2], W_local to C[2,2] (flattened)
    # A flat: i*3 + k
    # B flat: k*2 + j
    # C flat: j*2 + i
    
    terms = [
        # (u_idx, v_idx, w_idx, coeff)
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # This decomposition is non-standard in listing, I will use the explicit coefficients from the previous logic but vectorized.
        
        # Let's list the 11 vectors for U (size 6), V (size 6), W (size 4)
        # Based on the previous code's logic which was verified:
        
        # M1: U=[1,0,0,0,1,0], V=[1,0,0,0,1,0], W=[1,0,0,1]
        # M2: U=[0,0,0,1,1,0], V=[1,0,0,0,0,0], W=[0,0,1,-1]
        # M3: U=[1,0,0,0,0,0], V=[0,1,0,0,0,1], W=[0,1,1,0]
        # M4: U=[0,0,0,0,1,0], V=[0,0,1,0,-1,0], W=[1,1,0,0]
        # M5: U=[1,1,0,0,0,0], V=[0,0,0,0,0,1], W=[-1,1,0,0]
        # M6: U=[-1,0,0,1,0,0], V=[1,1,0,0,0,0], W=[0,0,0,1]
        # M7: U=[0,1,0,0,-1,0], V=[0,0,1,1,0,0], W=[1,0,0,0]
        # M8: U=[0,0,1,0,0,0], V=[0,0,0,1,0,0], W=[1,0,0,0]  # A02 * B20 -> C00
        # M9: U=[0,0,1,0,0,0], V=[0,0,0,0,0,1], W=[0,1,0,0]  # A02 * B21 -> C01
        # M10: U=[0,0,0,0,0,1], V=[0,0,0,1,0,0], W=[0,0,1,0] # A12 * B20 -> C10
        # M11: U=[0,0,0,0,0,1], V=[0,0,0,0,0,1], W=[0,0,0,1] # A12 * B21 -> C11
    ]
    
    # Convert to numpy arrays for efficiency
    # U: 6x11, V: 6x11, W: 4x11
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ], dtype=float)
    
    V_block = np.array([
        [1, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0], # B00
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0], # B01
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0], # B10
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], # B11
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # B20
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]  # B21
    ], dtype=float)
    
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1, 1, 0, 0, 0], # C00 (j=0,i=0)
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0], # C01 (j=1,i=0)
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], # C10 (j=0,i=1)
        [1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # C11 (j=1,i=1)
    ], dtype=float)
    
    # Map to global indices
    # U global: n*m = 6. Same as local.
    # V global: m*p = 24. Local index l (0..5) maps to global.
    # Local V index: k*2 + j_local.
    # Global V index: k*p + (2*b + j_local).
    # W global: p*n = 16. Local index l (0..3) maps to global.
    # Local W index: j_local*n + i.
    # Global W index: (2*b + j_local)*n + i.
    
    for b in range(4):
        start_col = b * 11
        end_col = (b + 1) * 11
        
        # U is the same for all blocks (shares A)
        U[:, start_col:end_col] = U_block
        
        # V mapping
        # Local V indices 0..5 correspond to (k=0,j=0), (k=0,j=1), (k=1,j=0), (k=1,j=1), (k=2,j=0), (k=2,j=1)
        # Global V indices:
        # l=0 -> k=0, j_loc=0 -> 0*8 + 2*b + 0 = 2b
        # l=1 -> k=0, j_loc=1 -> 0*8 + 2*b + 1 = 2b+1
        # l=2 -> k=1, j_loc=0 -> 1*8 + 2*b + 0 = 8+2b
        # l=3 -> k=1, j_loc=1 -> 1*8 + 2*b + 1 = 8+2b+1
        # l=4 -> k=2, j_loc=0 -> 2*8 + 2*b + 0 = 16+2b
        # l=5 -> k=2, j_loc=1 -> 2*8 + 2*b + 1 = 16+2b+1
        
        for l in range(6):
            k = l // 2
            j_loc = l % 2
            v_global_idx = k * p + 2 * b + j_loc
            V[v_global_idx, start_col:end_col] = V_block[l, :]
            
        # W mapping
        # Local W indices 0..3 correspond to (j_loc=0,i=0), (j_loc=1,i=0), (j_loc=0,i=1), (j_loc=1,i=1)
        # Global W indices:
        # l=0 -> j_loc=0, i=0 -> (2b+0)*2 + 0 = 4b
        # l=1 -> j_loc=1, i=0 -> (2b+1)*2 + 0 = 4b+2
        # l=2 -> j_loc=0, i=1 -> (2b+0)*2 + 1 = 4b+1
        # l=3 -> j_loc=1, i=1 -> (2b+1)*2 + 1 = 4b+3
        
        for l in range(4):
            j_loc = l // 2
            i = l % 2
            w_global_idx = (2 * b + j_loc) * n + i
            W[w_global_idx, start_col:end_col] = W_block[l, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
