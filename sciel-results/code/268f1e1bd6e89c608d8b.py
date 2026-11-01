import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    deadline = time.time() + 120.0
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We seek rank-r decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    # Known: Strassen-based block decomposition gives rank 28.
    # Schoolbook gives rank 32.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(4, 16, 16) = 16.
    # Also, for M_{2,2,p}, the rank is known to be 7*ceil(p/2) for even p via block Strassen.
    # For p=8: rank = 7*4 = 28.
    # 
    # Can we do better than 28? Let's try to verify if rank 28 is achievable
    # and whether lower ranks are possible.
    #
    # For M_{2,2,p}, the rank is actually known:
    # - M_{2,2,2} has rank 7 (Strassen)
    # - M_{2,2,4} has rank 14 (2 blocks of Strassen)
    # - M_{2,2,8} has rank 28 (4 blocks of Strassen)
    #
    # But wait - can we share factors across blocks? The key question is whether
    # the 2x2x8 tensor has rank less than 28.
    #
    # Actually, for matrix multiplication tensors M_{n,m,p}, the rank is:
    # - For M_{2,2,p} with p even: rank = 7*p/2
    # - This is because each 2x2x2 block needs 7 terms, and the A-factors
    #   cannot be shared across different k-columns in a way that reduces total rank.
    #
    # However, let me try to construct a rank-28 decomposition and verify it.
    # If it works, that's our answer. Let me also try to see if rank 27 or lower
    # is achievable through a more clever construction.
    #
    # For now, let's construct the rank-28 block Strassen decomposition carefully.
    
    def build_block_strassen():
        """Construct rank-28 decomposition using 4 Strassen blocks."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Strassen's 7 products for 2x2 matrix multiplication.
        # For a 2x2 * 2x2 = 2x2 multiplication:
        #   A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
        #   C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        #
        # Strassen:
        #   M1 = (a+d)(e+h)
        #   M2 = (c+d)e
        #   M3 = a(f-h)
        #   M4 = d(g-e)
        #   M5 = (a+b)h
        #   M6 = (c-a)(e+f)
        #   M7 = (b-d)(g+h)
        #
        #   C[0,0] = M1 - M5 + M7
        #   C[0,1] = M3 + M5
        #   C[1,0] = M2 + M4
        #   C[1,1] = M1 - M2 + M3 + M6
        
        # For each group of 2 columns (k1, k2), we apply Strassen.
        # A is shared, B_sub is the 2x2 slice of B for columns k1,k2.
        # C_sub is the 2x2 slice of C for columns k1,k2.
        
        # Strassen factors:
        # For product r, we have:
        #   U_vec: 4-dim vector (coefficients for A entries a,b,c,d)
        #   V_vec: 4-dim vector (coefficients for B_sub entries e,f,g,h)
        #   W_vec: 4-dim vector (coefficients for C_sub entries C00,C01,C10,C11)
        
        # A entries: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B_sub entries: e=B[0,k1], f=B[0,k2], g=B[1,k1], h=B[1,k2]
        # C_sub entries: C00=C[0,k1], C01=C[0,k2], C10=C[1,k1], C11=C[1,k2]
        
        strassen_factors = [
            # (U_coeffs [a,b,c,d], V_coeffs [e,f,g,h], W_coeffs [C00,C01,C10,C11])
            # M1 = (a+d)(e+h)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (c+d)e
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3 = a(f-h)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4 = d(g-e)
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            # M5 = (a+b)h
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6 = (c-a)(e+f)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (b-d)(g+h)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for q in range(p // 2):  # 4 groups
            k1 = 2 * q
            k2 = 2 * q + 1
            for r, (u_coeff, v_coeff, w_coeff) in enumerate(strassen_factors):
                t = q * 7 + r
                
                # Fill U: A coefficients
                # A entries: a=A[0,0] (idx 0), b=A[0,1] (idx 1), c=A[1,0] (idx 2), d=A[1,1] (idx 3)
                U[0, t] = u_coeff[0]  # a
                U[1, t] = u_coeff[1]  # b
                U[2, t] = u_coeff[2]  # c
                U[3, t] = u_coeff[3]  # d
                
                # Fill V: B_sub coefficients
                # B_sub: e=B[0,k1] (idx 0), f=B[0,k2] (idx 1), g=B[1,k1] (idx 2), h=B[1,k2] (idx 3)
                V[0 * p + k1, t] = v_coeff[0]  # e = B[0,k1]
                V[0 * p + k2, t] = v_coeff[1]  # f = B[0,k2]
                V[1 * p + k1, t] = v_coeff[2]  # g = B[1,k1]
                V[1 * p + k2, t] = v_coeff[3]  # h = B[1,k2]
                
                # Fill W: C_sub coefficients
                # C_sub: C00=C[0,k1], C01=C[0,k2], C10=C[1,k1], C11=C[1,k2]
                # W index: k*n + i
                W[k1 * n + 0, t] = w_coeff[0]  # C[0,k1]
                W[k2 * n + 0, t] = w_coeff[1]  # C[0,k2]
                W[k1 * n + 1, t] = w_coeff[2]  # C[1,k1]
                W[k2 * n + 1, t] = w_coeff[3]  # C[1,k2]
        
        return U, V, W
    
    # Build and verify the rank-28 decomposition
    U, V, W = build_block_strassen()
    
    # Verify
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    if np.allclose(T_computed, T_target, atol=1e-10):
        # Success! Return the rank-28 decomposition
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: try to debug and fix, or use schoolbook
        # Let's try a direct schoolbook construction
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
