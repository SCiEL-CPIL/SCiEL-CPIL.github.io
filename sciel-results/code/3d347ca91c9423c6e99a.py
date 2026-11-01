"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 x 2x12 matrix multiplication.
    Uses Strassen's algorithm for 2x2 x 2x2 blocks.
    Rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7

    # U: (n*m, rank) -> (4, 42)
    # V: (m*p, rank) -> (24, 42)
    # W: (p*n, rank) -> (24, 42)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Define Strassen terms for 2x2 block
    # A (2x2) @ B (2x2) -> C (2x2)
    # A indices: 0:11, 1:12, 2:21, 3:22
    # B indices: 0:11, 1:12, 2:21, 3:22
    # C indices: 0:11, 1:12, 2:21, 3:22
    
    # Term definitions: (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # We map A, B, C local 2x2 indices to linear indices
    # A lin: 0,1,2,3 corresponds to (0,0), (0,1), (1,0), (1,1)
    # B lin: 0,1,2,3 corresponds to (0,0), (0,1), (1,0), (1,1)
    # C lin: 0,1,2,3 corresponds to (0,0), (0,1), (1,0), (1,1)
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Pattern 0: M1
    p0_U = ([0, 3], [1, 1])
    p0_V = ([0, 3], [1, 1]) # B00, B11 are indices 0 and 3 in linearized 2x2? 
    # Wait. B is 2x2.
    # B[j,k]. j is row, k is col.
    # Linear index for B[j,k] is j*2+k.
    # B00 is 0. B01 is 1. B10 is 2. B11 is 3.
    # So for M1: (A00+A11)(B00+B11)
    # A00 is A[0] (idx 0). A11 is A[3] (idx 3).
    # B00 is B[0] (idx 0). B11 is B[3] (idx 3).
    # Correct.
    # W: C00 (0), C11 (3).
    # C00 gets M1. C11 gets M1.
    # So W indices 0 and 3 with coeff 1.
    
    strassen_terms = []
    
    # Helper to add term
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        term = {
            'U': [], 'V': [], 'W': []
        }
        term['U'] = (u_idx, u_coeff)
        term['V'] = (v_idx, v_coeff)
        term['W'] = (w_idx, w_coeff)
        strassen_terms.append(term)
        
    # Term 1: M1
    add_term([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1])
    
    # Term 2: M2: (A10+A11)B00
    # A10 is A[2] (idx 2). A11 is A[3].
    # B00 is B[0].
    add_term([2, 3], [1, 1], [0], [1], [2], [1]) # C10 is W[2]
    
    # Term 3: M3: A00(B01-B11)
    # A00 is A[0].
    # B01 is B[1]. B11 is B[3].
    add_term([0], [1], [1, 3], [1, -1], [1], [1]) # C01 is W[1]
    
    # Term 4: M4: A11(B10-B00)
    # A11 is A[3].
    # B10 is B[2]. B00 is B[0].
    add_term([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]) # C00 is W[0], C10 is W[2]
    
    # Term 5: M5: (A00+A01)B11
    # A00 is A[0], A01 is A[1].
    # B11 is B[3].
    add_term([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]) # C00 -M5, C01 +M5
    
    # Term 6: M6: (A10-A00)(B00+B01)
    # A10 is A[2], A00 is A[0].
    # B00 is B[0], B01 is B[1].
    add_term([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]) # C11 +M6
    
    # Term 7: M7: (A01-A11)(B10+B11)
    # A01 is A[1], A11 is A[3].
    # B10 is B[2], B11 is B[3].
    add_term([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]) # C00 +M7

    for b in range(num_blocks):
        base_c = b * 7
        k_offset = b * 2
        
        for s in range(7):
            c = base_c + s
            term = strassen_terms[s]
            
            # U
            for u_idx, coeff in zip(term['U'][0], term['U'][1]):
                U[u_idx, c] += coeff
                
            # V
            for v_idx, coeff in zip(term['V'][0], term['V'][1]):
                # v_idx is local 2x2 index (0..3)
                # global V index: j*p + k
                # local v_idx = j_loc * 2 + k_loc
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                global_k = k_offset + k_loc
                global_v_idx = j_loc * p + global_k
                V[global_v_idx, c] += coeff
                
            # W
            for w_idx, coeff in zip(term['W'][0], term['W'][1]):
                # w_idx is local 2x2 index for C (0..3)
                # C[i_loc, k_loc]. Linear index i_loc * 2 + k_loc.
                i_loc = w_idx // 2
                k_loc = w_idx % 2
                global_k = k_offset + k_loc
                # W index convention: W[k*n + i]
                global_w_idx = global_k * n + i_loc
                W[global_w_idx, c] += coeff
                
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
