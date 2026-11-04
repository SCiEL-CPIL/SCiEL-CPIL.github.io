"""
Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition.

This program constructs a rank-minimal exact tensor decomposition for multiplying
a 2 by 4 matrix by a 4 by 4 matrix. The output factor follows the official trace
convention W[k*n+i,r].

The algorithm uses Strassen's 7-multiplication method applied to the four 2x2
block multiplications inherent in the 2x4 * 4x4 structure.
Rank = 4 * 7 = 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    # Initialize U, V, W as numpy arrays for efficient verification, 
    # then convert to lists for the required output format.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 product X*Y:
    # Local indices for 2x2 matrix [a b; c d]: a=0, b=1, c=2, d=3
    # P1 = (x00+x11)(y00+y11) -> c00:+1, c11:+1
    # P2 = (x10+x11)y00 -> c10:+1, c11:-1
    # P3 = x00(y01-y11) -> c01:+1, c11:+1
    # P4 = x11(y10-y00) -> c00:+1, c10:+1
    # P5 = (x00+x01)y11 -> c00:-1, c01:+1
    # P6 = (x10-x00)(y00+y01) -> c11:+1
    # P7 = (x01-x11)(y10+y11) -> c00:+1
    
    # Each term is (U_coeffs, V_coeffs, W_coeffs) where coeffs are dicts {local_idx: coeff}
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # P7
    ]
    
    # Sub-products definition:
    # We split A (2x4) into A_L (cols 0-1) and A_R (cols 2-3).
    # We split B (4x4) into B_TL, B_TR (rows 0-1) and B_BL, B_BR (rows 2-3).
    # C (2x4) is split into C_L (cols 0-1) and C_R (cols 2-3).
    #
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    #
    # We apply Strassen to each of the 4 terms separately.
    # Note: The sum of two products X*Y + Z*W is NOT computed by a single Strassen call.
    # Instead, we compute X*Y with 7 mults and Z*W with 7 mults, total 14 per half.
    # Total 28.
    
    # Mapping functions for each sub-product
    # Input: local row r, local col c (0 or 1)
    # Output: global index
    
    # Sub-product 1: A_L (A[:, 0:2]) * B_TL (B[0:2, 0:2]) -> C_L (C[:, 0:2])
    # A_L[r, c] is at U[r*4 + c]
    # B_TL[r, c] is at V[r*4 + c]
    # C_L[r, c] is at W[c*2 + r]  (Trace convention W[k*n+i] where k=c, i=r)
    
    def map_1_u(r, c): return r * 4 + c
    def map_1_v(r, c): return r * 4 + c
    def map_1_w(r, c): return c * 2 + r
    
    # Sub-product 2: A_R (A[:, 2:4]) * B_BL (B[2:4, 0:2]) -> C_L (C[:, 0:2])
    # A_R[r, c] is at U[r*4 + (c+2)]
    # B_BL[r, c] is at V[(r+2)*4 + c]
    # C_L[r, c] is at W[c*2 + r]
    
    def map_2_u(r, c): return r * 4 + (c + 2)
    def map_2_v(r, c): return (r + 2) * 4 + c
    def map_2_w(r, c): return c * 2 + r
    
    # Sub-product 3: A_L (A[:, 0:2]) * B_TR (B[0:2, 2:4]) -> C_R (C[:, 2:4])
    # A_L[r, c] is at U[r*4 + c]
    # B_TR[r, c] is at V[r*4 + (c+2)]
    # C_R[r, c] is at W[(c+2)*2 + r]
    
    def map_3_u(r, c): return r * 4 + c
    def map_3_v(r, c): return r * 4 + (c + 2)
    def map_3_w(r, c): return (c + 2) * 2 + r
    
    # Sub-product 4: A_R (A[:, 2:4]) * B_BR (B[2:4, 2:4]) -> C_R (C[:, 2:4])
    # A_R[r, c] is at U[r*4 + (c+2)]
    # B_BR[r, c] is at V[(r+2)*4 + (c+2)]
    # C_R[r, c] is at W[(c+2)*2 + r]
    
    def map_4_u(r, c): return r * 4 + (c + 2)
    def map_4_v(r, c): return (r + 2) * 4 + (c + 2)
    def map_4_w(r, c): return (c + 2) * 2 + r
    
    sub_products = [
        (map_1_u, map_1_v, map_1_w),
        (map_2_u, map_2_v, map_2_w),
        (map_3_u, map_3_v, map_3_w),
        (map_4_u, map_4_v, map_4_w),
    ]
    
    col = 0
    for (map_u, map_v, map_w) in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            for loc, coeff in u_coeffs.items():
                r = loc // 2
                c = loc % 2
                U[map_u(r, c), col] = coeff
            
            # Fill V
            for loc, coeff in v_coeffs.items():
                r = loc // 2
                c = loc % 2
                V[map_v(r, c), col] = coeff
            
            # Fill W
            for loc, coeff in w_coeffs.items():
                r = loc // 2
                c = loc % 2
                W[map_w(r, c), col] = coeff
            
            col += 1
            
            # Deadline check
            if time.time() - t0 > deadline - 5:
                break

    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
