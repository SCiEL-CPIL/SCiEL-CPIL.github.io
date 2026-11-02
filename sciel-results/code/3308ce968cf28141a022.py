import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    The minimal rank is 49, achieved by applying the rank-7 Strassen decomposition
    to each of the 7 disjoint 2x2 blocks of the 2x14 matrix.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # The rank-7 Strassen terms for 2x2x2 multiplication
    # U, V, W indices are 0..3 for 2x2 matrices (row-major: 0,1,2,3)
    # Terms: (U_dict, V_dict, W_dict)
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11 (Local to block, mapped later)
    
    # Strassen Products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstructing C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the coefficients for each rank-1 term (M1..M7)
    # Term s has U_s, V_s, W_s
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # C10 gets M2, C11 gets -M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # C01 gets M3, C11 gets M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # C00 gets M4, C10 gets M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # C00 gets -M5, C01 gets M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # C11 gets M6
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # C00 gets M7
    ]
    
    num_blocks = p // 2  # 7
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank  # 49
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        k0 = 2 * t
        k1 = k0 + 1
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            u_coeffs, v_local, w_local = strassen_terms[s]
            
            # Fill U
            # U index is i*m + j = 2*i + j
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] = coeff
            
            # Fill V
            # v_local keys are 0,1,2,3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1] in the block
            # B is 2x14. Block t corresponds to columns k0, k1.
            # B[j, k0 + v_col] where v_loc = j*2 + v_col
            # V index is j*p + k = 14*j + k
            for v_loc, coeff in v_local.items():
                j = v_loc // 2
                v_col = v_loc % 2
                k_col = k0 + v_col
                v_idx = j * p + k_col
                V[v_idx][col] = coeff
            
            # Fill W
            # w_local keys are 0,1,2,3 corresponding to C[0,0], C[0,1], C[1,0], C[1,1] in the block
            # C is 14x2 (conceptually for the index formula W[k*n+i]).
            # C[k*n + i] -> C[row_in_vector, col_in_matrix]?
            # Prompt: W[k*n+i,r]. k is column index of B/C (0..13), i is row index of A/C (0..1).
            # So W index = k*2 + i.
            # w_loc 0: C[0,0] local -> i=0, k=k0 -> idx = k0*2 + 0
            # w_loc 1: C[0,1] local -> i=0, k=k1 -> idx = k1*2 + 0
            # w_loc 2: C[1,0] local -> i=1, k=k0 -> idx = k0*2 + 1
            # w_loc 3: C[1,1] local -> i=1, k=k1 -> idx = k1*2 + 1
            
            for w_loc, coeff in w_local.items():
                # w_loc 0: C00 (local) -> i=0, k=k0
                # w_loc 1: C01 (local) -> i=0, k=k1
                # w_loc 2: C10 (local) -> i=1, k=k0
                # w_loc 3: C11 (local) -> i=1, k=k1
                
                if w_loc == 0:
                    i = 0
                    k_row = k0
                elif w_loc == 1:
                    i = 0
                    k_row = k1
                elif w_loc == 2:
                    i = 1
                    k_row = k0
                else: # w_loc == 3
                    i = 1
                    k_row = k1
                
                w_idx = k_row * n + i
                W[w_idx][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
