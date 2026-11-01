import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen matrices (Rank 7)
    # U7, V7, W7 are 4x7
    # Rows correspond to indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  <-- Note: k*2+i convention
    #    c00 (i=0,k=0) -> 0
    #    c10 (i=1,k=0) -> 1
    #    c01 (i=0,k=1) -> 2
    #    c11 (i=1,k=1) -> 3
    
    U7 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V7 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W7 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker Product
    # U_kr = U7 x U7 (16x49)
    U_kr = np.kron(U7, U7)
    V_kr = np.kron(V7, V7)
    W_kr = np.kron(W7, W7)
    
    # Permutations
    # We need to map Kronecker indices to the required flat indices.
    
    # For U (A): Required index i*m + j (row-major)
    # Kronecker index: br*8 + bc*4 + ir*2 + ic
    # Required index: (br*2+ir)*4 + (bc*2+ic)
    
    def get_permutation_U():
        perm = []
        for i in range(4):
            for j in range(4):
                req_idx = i * 4 + j
                br = i // 2
                bc = j // 2
                ir = i % 2
                ic = j % 2
                kr_idx = br * 8 + bc * 4 + ir * 2 + ic
                perm.append(kr_idx)
        return perm

    def get_permutation_V():
        # B is also row-major in the seed: j*p + k
        # Same as U
        return get_permutation_U()

    def get_permutation_W():
        # W index: k*n + i (col-major for C, or just specific formula)
        # k is col, i is row.
        # Required index: k * 4 + i
        # Kronecker index for W (C):
        # C is result of A (blocks) * B (blocks).
        # The Kronecker structure for C is based on C blocks.
        # C block (br, bc) contains entries C[br*2+ir, bc*2+ic].
        # Kronecker index: br*8 + bc*4 + ir*2 + ic
        # Required index: k * 4 + i = (bc*2 + ic) * 4 + (br*2 + ir)
        
        perm = []
        for k in range(4):
            for i in range(4):
                req_idx = k * 4 + i
                bc = k // 2
                br = i // 2
                ic = k % 2
                ir = i % 2
                kr_idx = br * 8 + bc * 4 + ir * 2 + ic
                perm.append(kr_idx)
        return perm

    perm_U = get_permutation_U()
    perm_V = get_permutation_V()
    perm_W = get_permutation_W()
    
    U_final = U_kr[perm_U, :]
    V_final = V_kr[perm_V, :]
    W_final = W_kr[perm_W, :]
    
    # Convert to lists of lists as required by interface
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
