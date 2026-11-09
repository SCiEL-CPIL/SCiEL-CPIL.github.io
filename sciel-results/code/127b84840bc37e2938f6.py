import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 solution using Strassen for M_2,2,2 blocks
    # M_2,3,8 is split into 4 M_2,3,2 blocks
    # Each M_2,3,2 uses 11 terms (7 Strassen + 4 for j=2)
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for M_2,2,2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: (i, j) -> i*2 + j (local)
    # B indices: (j, k) -> j*2 + k (local)
    # C indices: (i, k) -> i*2 + k (local)
    
    # Strassen terms: list of (U_indices, V_indices, W_indices, coeff)
    # We need to map local indices to global indices for each block
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (1, 1)], # A00, A11
            'V': [(0, 1), (1, 1)], # B00, B11
            'W': [(0, 1), (1, 1)], # C00, C11
            'coeff': 1
        },
        # M2
        {
            'U': [(1, 1), (1, 1)], # A10, A11
            'V': [(0, 1)],         # B00
            'W': [(1, 1)],         # C10
            'coeff': 1
        },
        # M3
        {
            'U': [(0, 1)],         # A00
            'V': [(0, 1), (1, -1)],# B01, B11
            'W': [(0, 1)],         # C01
            'coeff': 1
        },
        # M4
        {
            'U': [(1, 1)],         # A11
            'V': [(1, 1), (0, -1)],# B10, B00
            'W': [(0, 1), (1, 1)], # C00, C11
            'coeff': 1
        },
        # M5
        {
            'U': [(0, 1), (0, 1)], # A00, A01
            'V': [(1, 1)],         # B11
            'W': [(0, -1), (1, 1)],# C00, C11
            'coeff': 1
        },
        # M6
        {
            'U': [(1, 1), (0, -1)],# A10, A00
            'V': [(0, 1), (0, 1)], # B00, B01
            'W': [(1, 1)],         # C11
            'coeff': 1
        },
        # M7
        {
            'U': [(0, 1), (1, -1)],# A01, A11
            'V': [(1, 1), (1, 1)], # B10, B11
            'W': [(0, 1), (1, 1)], # C00, C11
            'coeff': 1
        }
    ]
    
    # Note: The above Strassen implementation is slightly messy with indices.
    # Let's use a cleaner representation.
    # M_t = (sum u_ij A_ij) (sum v_jk B_jk)
    # C_ik = sum w_ik M_t
    
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the terms as (U_dict, V_dict, W_dict)
    # U_dict: {(i,j): coeff}
    # V_dict: {(j,k): coeff}
    # W_dict: {(i,k): coeff}
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}) # M7
    ]
    
    # Add j=2 terms for M_2,3,2
    # C_ik += A_i2 * B_2k
    # This requires 4 terms:
    # (i=0, k=0): U{(0,2):1}, V{(2,0):1}, W{(0,0):1}
    # (i=0, k=1): U{(0,2):1}, V{(2,1):1}, W{(0,1):1}
    # (i=1, k=0): U{(1,2):1}, V{(2,0):1}, W{(1,0):1}
    # (i=1, k=1): U{(1,2):1}, V{(2,1):1}, W{(1,1):1}
    
    j2_terms = [
        ({(0,2):1}, {(2,0):1}, {(0,0):1}),
        ({(0,2):1}, {(2,1):1}, {(0,1):1}),
        ({(1,2):1}, {(2,0):1}, {(1,0):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1})
    ]
    
    all_terms_232 = terms + j2_terms
    
    # Now map to M_2,3,8
    # 4 blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    
    term_idx = 0
    for block in range(4):
        k_offset = block * 2
        for U_dict, V_dict, W_dict in all_terms_232:
            # U indices: i, j -> i*3 + j (global)
            for (i, j), coeff in U_dict.items():
                U[i * m + j, term_idx] += coeff
            
            # V indices: j, k -> j*8 + k (global)
            for (j, k), coeff in V_dict.items():
                V[j * p + (k + k_offset), term_idx] += coeff
            
            # W indices: i, k -> k*2 + i (global)
            # Note: W index in prompt is k*n + i.
            # Local k is 0 or 1. Global k is k + k_offset.
            for (i, k), coeff in W_dict.items():
                W[(k + k_offset) * n + i, term_idx] += coeff
            
            term_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    # Verify
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Construct tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Compute decomposition
    # D[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    D = np.einsum('ar,br,cr->abc', U, V, W)
    
    error = np.linalg.norm(D - T)
    print(f"Error: {error}")
    if error < 1e-9:
        print("Valid decomposition")
    else:
        print("Invalid decomposition")
