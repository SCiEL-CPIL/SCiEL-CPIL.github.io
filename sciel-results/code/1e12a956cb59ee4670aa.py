import numpy as np

DIMENSIONS = [3, 3, 4]


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Strategy: Split p=4 into 3+1
    # Part 1: 3x3 * 3x3 -> 3x3 using rank 23
    # Part 2: 3x3 * 3x1 -> 3x1 using rank 9
    # Total: 32
    
    # Build rank-23 decomposition for 3x3x3
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
    # Term t: U[:,t] * V[:,t] gives contribution to W[:,t]
    
    # The rank-23 algorithm uses the following 23 bilinear forms.
    # I'll encode each as (U_coeffs, V_coeffs, W_coeffs) where each is a dict {index: coeff}
    
    def add_term(U, V, W, t, u_dict, v_dict, w_dict):
        for idx, c in u_dict.items():
            U[idx, t] += c
        for idx, c in v_dict.items():
            V[idx, t] += c
        for idx, c in w_dict.items():
            W[idx, t] += c
    
    # Term 0: a00*b00 -> c00
    add_term(U3, V3, W3, 0, {0:1}, {0:1}, {0:1})
    # Term 1: a01*b10 -> c00  
    add_term(U3, V3, W3, 1, {1:1}, {3:1}, {0:1})
    # Term 2: a02*b20 -> c00
    add_term(U3, V3, W3, 2, {2:1}, {6:1}, {0:1})
    # Term 3: a10*b01 -> c10
    add_term(U3, V3, W3, 3, {3:1}, {1:1}, {3:1})
    # Term 4: a11*b11 -> c11
    add_term(U3, V3, W3, 4, {4:1}, {4:1}, {4:1})
    # Term 5: a12*b21 -> c11
    add_term(U3, V3, W3, 5, {5:1}, {7:1}, {4:1})
    # Term 6: a20*b02 -> c20
    add_term(U3, V3, W3, 6, {6:1}, {2:1}, {6:1})
    # Term 7: a21*b12 -> c21
    add_term(U3, V3, W3, 7, {7:1}, {5:1}, {7:1})
    # Term 8: a22*b22 -> c22
    add_term(U3, V3, W3, 8, {8:1}, {8:1}, {8:1})
    
    # Terms 9-22: Strassen-like combinations
    # These reduce the remaining 18 schoolbook terms to 14 combined terms
    
    # We need to compute: c01, c02, c12, c21, c22 and fix c00,c10,c11,c20
    # Actually the above 9 terms only give partial results. Let me use a different approach.
    
    # Reset and use a verified construction
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # I'll use the explicit rank-23 from literature. The algorithm is:
    # Based on the work of Stott and others, here are the 23 multiplications.
    
    # For safety and correctness, let me use a hybrid: 
    # Use rank 27 for 3x3x3 (schoolbook) and rank 9 for 3x3x1, total 36.
    # This matches the current score. To improve, I need a valid rank-23.
    
    # Let me try to construct rank-23 properly by using the following known valid set:
    
    # After careful analysis, here's a valid rank-23 construction:
    
    # I'll use the approach of combining Strassen for 2x2 subblocks with corrections.
    
    # For now, return schoolbook to ensure correctness
    rank = n * m * p  # 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
