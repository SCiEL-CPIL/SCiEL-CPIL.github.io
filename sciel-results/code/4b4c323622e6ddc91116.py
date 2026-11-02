import numpy as np
import time

DIMENSIONS = [3, 3, 3]
DEADLINE = 100  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Define the rank-26 decomposition terms
    # Each term is (u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w)
    # Or simpler: list of (u_idx, v_idx, w_idx, coeff)
    # But we need to handle multiple indices per matrix.
    
    # We will build the U, V, W matrices by iterating over the 26 terms.
    # Term structure: 
    # A term is a sum of products of linear forms.
    # For a rank-1 tensor: u * v * w
    # u = sum c_ua * e_ua
    # v = sum c_vb * e_vb
    # w = sum c_wc * e_wc
    
    # The 26 terms are:
    # 1-7: Strassen for top-left 2x2
    # 8-11: A_TR * B_BL
    # 12-15: A_TL * B_TR
    # 16-17: A_TR * B_BR
    # 18-21: A_BL * B_TL
    # 22-23: A_BR * B_BL
    # 24-25: A_BL * B_TR
    # 26: A_BR * B_BR

    terms = []
    
    # Indices mapping:
    # A: (i, j) -> i*3 + j
    # B: (j, k) -> j*3 + k
    # C: (i, k) -> k*3 + i  (W[k*n+i])
    
    # 1. Strassen M1 = (A00+A11)(B00+B11) -> C00, C11
    # A00:0, A11:4. B00:0, B11:4. C00:0, C11:4.
    terms.append({
        'u': [(0, 1), (4, 1)],
        'v': [(0, 1), (4, 1)],
        'w': [(0, 1), (4, 1)]
    })
    
    # 2. Strassen M2 = (A10+A11)B00 -> C10, C11
    # A10:3, A11:4. B00:0. C10:1, C11:4.
    terms.append({
        'u': [(3, 1), (4, 1)],
        'v': [(0, 1)],
        'w': [(1, 1), (4, -1)]
    })
    
    # 3. Strassen M3 = A00(B01-B11) -> C01, C11
    # A00:0. B01:1, B11:4. C01:3, C11:4.
    terms.append({
        'u': [(0, 1)],
        'v': [(1, 1), (4, -1)],
        'w': [(3, 1), (4, 1)]
    })
    
    # 4. Strassen M4 = A11(B10-B00) -> C00, C10
    # A11:4. B10:3, B00:0. C00:0, C10:1.
    terms.append({
        'u': [(4, 1)],
        'v': [(3, 1), (0, -1)],
        'w': [(0, 1), (1, 1)]
    })
    
    # 5. Strassen M5 = (A00+A01)B11 -> C00, C01
    # A00:0, A01:1. B11:4. C00:0, C01:3.
    terms.append({
        'u': [(0, 1), (1, 1)],
        'v': [(4, 1)],
        'w': [(0, -1), (3, 1)]
    })
    
    # 6. Strassen M6 = (A10-A00)(B00+B01) -> C11
    # A10:3, A00:0. B00:0, B01:1. C11:4.
    terms.append({
        'u': [(3, 1), (0, -1)],
        'v': [(0, 1), (1, 1)],
        'w': [(4, 1)]
    })
    
    # 7. Strassen M7 = (A01-A11)(B10+B11) -> C00
    # A01:1, A11:4. B10:3, B11:4. C00:0.
    terms.append({
        'u': [(1, 1), (4, -1)],
        'v': [(3, 1), (4, 1)],
        'w': [(0, 1)]
    })
    
    # 8-11. A_TR * B_BL
    # A_TR: (0,2)->2, (1,2)->5
    # B_BL: (2,0)->6, (2,1)->7
    # C_TL: (0,0)->0, (0,1)->3, (1,0)->1, (1,1)->4
    # C00 += A02 B20
    terms.append({'u': [(2, 1)], 'v': [(6, 1)], 'w': [(0, 1)]})
    # C01 += A02 B21
    terms.append({'u': [(2, 1)], 'v': [(7, 1)], 'w': [(3, 1)]})
    # C10 += A12 B20
    terms.append({'u': [(5, 1)], 'v': [(6, 1)], 'w': [(1, 1)]})
    # C11 += A12 B21
    terms.append({'u': [(5, 1)], 'v': [(7, 1)], 'w': [(4, 1)]})
    
    # 12-15. A_TL * B_TR
    # A_TL: 0,1,3,4
    # B_TR: (0,2)->2, (1,2)->5
    # C_TR: (0,2)->6, (1,2)->7
    # C02 += A00 B02 + A01 B12
    terms.append({'u': [(0, 1)], 'v': [(2, 1)], 'w': [(6, 1)]})
    terms.append({'u': [(1, 1)], 'v': [(5, 1)], 'w': [(6, 1)]})
    # C12 += A10 B02 + A11 B12
    terms.append({'u': [(3, 1)], 'v': [(2, 1)], 'w': [(7, 1)]})
    terms.append({'u': [(4, 1)], 'v': [(5, 1)], 'w': [(7, 1)]})
    
    # 16-17. A_TR * B_BR
    # A_TR: 2,5
    # B_BR: 8
    # C_TR: 6,7
    # C02 += A02 B22
    terms.append({'u': [(2, 1)], 'v': [(8, 1)], 'w': [(6, 1)]})
    # C12 += A12 B22
    terms.append({'u': [(5, 1)], 'v': [(8, 1)], 'w': [(7, 1)]})
    
    # 18-21. A_BL * B_TL
    # A_BL: (2,0)->6, (2,1)->7
    # B_TL: 0,1,3,4
    # C_BL: (2,0)->2, (2,1)->5
    # C20 += A20 B00 + A21 B10
    terms.append({'u': [(6, 1)], 'v': [(0, 1)], 'w': [(2, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(3, 1)], 'w': [(2, 1)]})
    # C21 += A20 B01 + A21 B11
    terms.append({'u': [(6, 1)], 'v': [(1, 1)], 'w': [(5, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(4, 1)], 'w': [(5, 1)]})
    
    # 22-23. A_BR * B_BL
    # A_BR: 8
    # B_BL: 6,7
    # C_BL: 2,5
    # C20 += A22 B20
    terms.append({'u': [(8, 1)], 'v': [(6, 1)], 'w': [(2, 1)]})
    # C21 += A22 B21
    terms.append({'u': [(8, 1)], 'v': [(7, 1)], 'w': [(5, 1)]})
    
    # 24-25. A_BL * B_TR
    # A_BL: 6,7
    # B_TR: 2,5
    # C_BR: 8
    # C22 += A20 B02 + A21 B12
    terms.append({'u': [(6, 1)], 'v': [(2, 1)], 'w': [(8, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(5, 1)], 'w': [(8, 1)]})
    
    # 26. A_BR * B_BR
    # A_BR: 8
    # B_BR: 8
    # C_BR: 8
    terms.append({'u': [(8, 1)], 'v': [(8, 1)], 'w': [(8, 1)]})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, coeff in term['u']:
            U[idx][r] += coeff
        for idx, coeff in term['v']:
            V[idx][r] += coeff
        for idx, coeff in term['w']:
            W[idx][r] += coeff
            
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the matrix multiplication tensor
    # Tensor T[i,j,k] = sum_l A[i,l] * B[l,k]
    # Our decomposition: T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    
    # Build the expected tensor
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i, j, k] = 1  # coefficient of A[i,j]*B[j,k]*C[i,k] is 1
    
    # Build the decomposed tensor
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
    
    # Check correctness
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # If incorrect, return the schoolbook rank-27 decomposition as fallback
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U, "V": V, "W": W}
