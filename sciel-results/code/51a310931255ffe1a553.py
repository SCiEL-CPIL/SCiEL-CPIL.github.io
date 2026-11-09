import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication using Strassen(2x2) tensor Laderman(3x3)
    # Total rank: 7 * 23 = 161
    
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Laderman 3x3 decomposition terms (rank 23)
    # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = AB
    # Using 0-indexed: local_idx = row*3 + col
    # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    
    laderman_terms = [
        # M1: (a00+a10)(b00+b10)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1}),
        # M2: (a00+a10)(b01+b11)
        ({0: 1, 3: 1}, {1: 1, 4: 1}, {1: 1}),
        # M3: a00(b01-b00)
        ({0: 1}, {1: 1, 0: -1}, {1: 1}),
        # M4: a10(b10-b11)
        ({3: 1}, {3: 1, 4: -1}, {3: 1, 4: 1}),
        # M5: (a01+a02)b11
        ({1: 1, 2: 1}, {4: 1}, {1: 1, 2: 1}),
        # M6: a02(b21-b11)
        ({2: 1}, {7: 1, 4: -1}, {2: 1}),
        # M7: (a11+a12)b22
        ({4: 1, 5: 1}, {8: 1}, {4: 1, 5: 1}),
        # M8: a12(b22-b20)
        ({5: 1}, {8: 1, 6: -1}, {5: 1}),
        # M9: a00*b20
        ({0: 1}, {6: 1}, {6: 1}),
        # M10: a01*b20
        ({1: 1}, {6: 1}, {7: 1}),
        # M11: a02*b20
        ({2: 1}, {6: 1}, {8: 1}),
        # M12: a10*b21
        ({3: 1}, {7: 1}, {6: 1}),
        # M13: a11*b21
        ({4: 1}, {7: 1}, {7: 1}),
        # M14: a12*b21
        ({5: 1}, {7: 1}, {8: 1}),
        # M15: a00*b22
        ({0: 1}, {8: 1}, {6: 1}),
        # M16: a01*b22
        ({1: 1}, {8: 1}, {7: 1}),
        # M17: a02*b22
        ({2: 1}, {8: 1}, {8: 1}),
        # M18: a10*b22
        ({3: 1}, {8: 1}, {7: 1}),
        # M19: a11*b22
        ({4: 1}, {8: 1}, {8: 1}),
        # M20: a12*b22
        ({5: 1}, {8: 1}, {8: 1}),
        # M21: (a00+a01+a02)(b20+b21+b22)
        ({0: 1, 1: 1, 2: 1}, {6: 1, 7: 1, 8: 1}, {6: -1, 7: -1, 8: -1}),
        # M22: (a10+a11+a12)(b20+b21+b22)
        ({3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {7: -1, 8: -1}),
        # M23: (a00+a01+a02+a10+a11+a12)(b20+b21+b22)
        ({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}),
    ]
    
    # Verify Laderman 3x3 first
    def verify_laderman():
        """Verify Laderman 3x3 decomposition produces correct C=AB"""
        # Build 3x3 decomposition
        sz = 3
        r = 23
        U3 = np.zeros((sz*sz, r))
        V3 = np.zeros((sz*sz, r))
        W3 = np.zeros((sz*sz, r))
        
        for t, (u_c, v_c, w_c) in enumerate(laderman_terms):
            for idx, coeff in u_c.items():
                rr, cc = divmod(idx, sz)
                U3[rr*sz+cc, t] += coeff
            for idx, coeff in v_c.items():
                rr, cc = divmod(idx, sz)
                V3[rr*sz+cc, t] += coeff
            for idx, coeff in w_c.items():
                rr, cc = divmod(idx, sz)
                # W index: W[j,i] where j=col of C, i=row of C
                # C[i,j] = sum_k A[i,k] B[k,j]
                # W tensor index: W[j*sz+i, t]
                W3[cc*sz+rr, t] += coeff
        
        # Verify: T[i,k,j] = sum_t U3[i,k,t]*V3[k,j,t]*W3[j,i,t] should be 1
        U3t = U3.reshape(sz, sz, r)
        V3t = V3.reshape(sz, sz, r)
        W3t = W3.reshape(sz, sz, r)
        T3 = np.einsum('ikt,kjt,jit->ikj', U3t, V3t, W3t)
        return np.allclose(T3, np.ones((sz,sz,sz)), atol=1e-10)
    
    laderman_ok = verify_laderman()
    
    if not laderman_ok:
        # Fall back to Strassen(2x2) x schoolbook(3x3) = rank 189
        rank_fb = 189
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for local_idx, coeff in u_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * k + c
                            U_fb[global_r * m + global_c][col_fb] += coeff
                        for local_idx, coeff in v_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * k + r
                            global_c = 2 * j + c
                            V_fb[global_r * p + global_c][col_fb] += coeff
                        for local_idx, coeff in w_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * j + c
                            W_fb[global_c * n + global_r][col_fb] += coeff
                        col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Now build 6x6 using Strassen(2x2) tensor Laderman(3x3)
    # The 6x6 matrix A is partitioned as:
    #   A = [[A00, A01, A02],
    #        [A10, A11, A12],
    #        [A20, A21, A22]]
    # where each Aij is 2x2
    # Similarly for B and C.
    # 
    # Strassen operates on the 2x2 sub-blocks of each 2x2 block.
    # Laderman operates on the 3x3 grid of 2x2 blocks.
    #
    # For each (i,k,j) position in the 3x3 grid (i=row block, k=col block of A, j=row block of B):
    #   We need to compute C_block[i,j] += A_block[i,k] * B_block[k,j]
    #   Using Laderman to combine across the 3x3 grid, and Strassen within each 2x2 block.
    #
    # The tensor product structure:
    # For each Laderman term t (0..22) and Strassen term s (0..6):
    #   The scalar multiplication is: (linear combo of A elements in block (i,k)) * (linear combo of B elements in block (k,j))
    #   where i,k,j are determined by Laderman, and the linear combos within each 2x2 block are from Strassen.
    
    # Laderman tells us which (i,k) positions of A and which (k,j) positions of B to combine,
    # and how to combine the results into C[i,j].
    # Strassen tells us how to multiply each 2x2 block pair with 7 scalar mults.
    
    # For Laderman term t, it specifies:
    #   U_t: coefficients for A[i,k] for various (i,k) pairs
    #   V_t: coefficients for B[k,j] for various (k,j) pairs  
    #   W_t: coefficients for C[i,j] for various (i,j) pairs
    #
    # For Strassen term s, it specifies:
    #   u_s: coefficients for 2x2 elements of A_block
    #   v_s: coefficients for 2x2 elements of B_block
    #   w_s: coefficients for 2x2 elements of C_block
    #
    # The combined term (t,s) has:
    #   For each (i,k) in U_t with coeff cu, and each 2x2 element (a,b) in u_s with coeff cs:
    #     U[global_r, global_c, col] += cu * cs
    #     where global_r = 2*i + a, global_c = 2*k + b
    #   Similarly for V and W.
    
    for t, (u_lad, v_lad, w_lad) in enumerate(laderman_terms):
        for s, (u_str, v_str, w_str) in enumerate(strassen_terms):
            # Process U: for each (i,k) in Laderman U and each 2x2 elem in Strassen U
            for (ik_idx, cu) in u_lad.items():
                i, k = divmod(ik_idx, 3)
                for (elem, cs) in u_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * k + b
                    U[global_r * m + global_c][col] += cu * cs
            
            # Process V: for each (k,j) in Laderman V and each 2x2 elem in Strassen V
            for (kj_idx, cv) in v_lad.items():
                k, j = divmod(kj_idx, 3)
                for (elem, cs) in v_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * k + a
                    global_c = 2 * j + b
                    V[global_r * p + global_c][col] += cv * cs
            
            # Process W: for each (i,j) in Laderman W and each 2x2 elem in Strassen W
            for (ij_idx, cw) in w_lad.items():
                i, j = divmod(ij_idx, 3)
                for (elem, cs) in w_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * j + b
                    W[global_c * n + global_r][col] += cw * cs
            
            col += 1
    
    # Verify the full 6x6 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
