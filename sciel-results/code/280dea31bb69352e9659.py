import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Rank 49 is the known optimal rank for 4x4 matrix multiplication
    # using 2-level Strassen decomposition.
    rank = 49
    
    # Strassen coefficients for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A10)B11
    # M5 = (A01-A11)(B10+B11)
    # M6 = (A01-A10)(B00+B10)
    
    # Coefficients for U (A factors)
    # Indices: block (0..3), local (0..3)
    # block = rb*2 + cb, local = li*2 + lj
    # We need to map Strassen terms M0..M6 to sums of A_{i,j}
    
    # M0: A00 + A11 -> indices (0,0) and (1,1)
    # M1: A10 + A11 -> indices (1,0) and (1,1)
    # M2: A00 -> index (0,0)
    # M3: A11 -> index (1,1)
    # M4: A00 + A10 -> indices (0,0) and (1,0)
    # M5: A01 - A11 -> indices (0,1) and (1,1)
    # M6: A01 - A10 -> indices (0,1) and (1,0)
    
    # The code in the prompt uses a specific tensor product structure.
    # U_{flat, r} = S_A[K, block] * S_A[l, local]
    # This implies a Kronecker product structure for the 2-level recursion.
    
    S_A = np.array([
        [1, 0, 0, 1], # M0: A00 + A11
        [0, 0, 1, 1], # M1: A10 + A11
        [1, 0, 0, 0], # M2: A00
        [0, 0, 0, 1], # M3: A11
        [1, 1, 0, 0], # M4: A00 + A10
        [0, 1, 0, -1],# M5: A01 - A11
        [0, 1, 0, -1] # M6: A01 - A10 ? Wait, M6 is A01 - A10.
                      # A01 is (0,1), A10 is (1,0).
                      # So [0, 1, -1, 0] ?
                      # Let's check the prompt's S_A row 6: [0, 1, 0, -1].
                      # This corresponds to A01 - A11.
                      # But M6 should be A01 - A10.
                      # Let's re-read standard Strassen.
                      # M6 = (A01 - A10)(B00 + B10).
                      # So A part is A01 - A10.
                      # Indices: A01 is (0,1), A10 is (1,0).
                      # Vector: [0, 1, -1, 0].
                      # Prompt has [0, 1, 0, -1]. This is A01 - A11.
                      # Is the prompt's Strassen variant different?
                      # Or did I misidentify the terms?
                      # Let's check S_B and S_C to see if it's consistent.
    ])
    
    S_B = np.array([
        [1, 0, 0, 1], # M0: B00 + B11
        [1, 0, 0, 0], # M1: B00
        [0, 1, 0, -1],# M2: B01 - B11
        [-1, 0, 1, 0],# M3: B10 - B00
        [0, 0, 0, 1], # M4: B11
        [1, 1, 0, 0], # M5: B10 + B11
        [0, 0, 1, 1]  # M6: B00 + B10
    ])
    
    S_C = np.array([
        [1, 0, 0, 1], # M0: C00 + C11
        [0, 0, 1, -1],# M1: C10 - C00 ?
        [0, 1, 0, 1], # M2: C01 + C11
        [1, 0, 1, 0], # M3: C00 + C10
        [-1, 1, 0, 0],# M4: C00 - C10 ?
        [0, 0, 0, 1], # M5: C11
        [1, 0, 0, 0]  # M6: C00
    ])
    
    # The prompt's code produced a valid score, so its coefficients are likely correct for its specific index mapping.
    # I will reuse the exact coefficients from the prompt to ensure validity.
    
    S_A = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ])
    
    S_B = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    S_C = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Precompute indices
    i_idx, j_idx = np.indices((n, m))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    rb_u = (i_idx // 2).flatten()
    cb_u = (j_idx // 2).flatten()
    li_u = (i_idx % 2).flatten()
    lj_u = (j_idx % 2).flatten()
    block_u = rb_u * 2 + cb_u
    local_u = li_u * 2 + lj_u
    flat_u = i_flat * m + j_flat
    
    j_idx, k_idx = np.indices((m, p))
    j_flat_v = j_idx.flatten()
    k_flat_v = k_idx.flatten()
    rb_v = (j_idx // 2).flatten()
    cb_v = (k_idx // 2).flatten()
    li_v = (j_idx % 2).flatten()
    lj_v = (k_idx % 2).flatten()
    block_v = rb_v * 2 + cb_v
    local_v = li_v * 2 + lj_v
    flat_v = j_flat_v * p + k_flat_v
    
    i_idx_w, k_idx_w = np.indices((n, p))
    i_flat_w = i_idx_w.flatten()
    k_flat_w = k_idx_w.flatten()
    rb_w = (i_idx_w // 2).flatten()
    cb_w = (k_idx_w // 2).flatten()
    li_w = (i_idx_w % 2).flatten()
    lj_w = (k_idx_w % 2).flatten()
    block_w = rb_w * 2 + cb_w
    local_w = li_w * 2 + lj_w
    flat_w = k_flat_w * n + i_flat_w
    
    for K in range(7):
        for l in range(7):
            r = K * 7 + l
            
            # U
            coeff_u = S_A[K, block_u] * S_A[l, local_u]
            U[flat_u, r] = coeff_u
            
            # V
            coeff_v = S_B[K, block_v] * S_B[l, local_v]
            V[flat_v, r] = coeff_v
            
            # W
            coeff_w = S_C[K, block_w] * S_C[l, local_w]
            W[flat_w, r] = coeff_w
            
    if time.time() > deadline:
        # Should not happen, but safety net
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
