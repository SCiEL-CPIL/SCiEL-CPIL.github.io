import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def verify_decomposition(U, V, W):
        """
        Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        # Shapes must match
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        rank = U.shape[1]
        if rank == 0:
            return False
        
        # Check for integer/half-integer entries (float check for stability)
        # The problem allows integers/half-integers. We generated integers.
        # We check if values are close to 0.5 * integer
        entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        if not np.allclose(entries * 2, np.round(entries * 2), atol=1e-5):
             # If not close to half-integers, it might still be valid if they are just floats,
             # but the contract says "integer/half-integer". 
             # However, usually strict integer is safer. 
             # Let's just check if the tensor product is exact.
             pass

        # Construct Target Tensor T_target
        # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
        # a in [0, nm), b in [0, mp), c in [0, pn)
        
        # Efficient construction of target tensor without full 4D loop if possible,
        # but 4*5*5*5*5*5 is small.
        # T[a, b, c] is 1 if:
        # a % m == b // p  (j == j)
        # b % p == c // n  (k == k)
        # c % n == a // m  (i == i)
        
        # Let's compute T_calc and compare to T_target
        # T_calc[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # We can reshape U, V, W to use matmul? 
        # T_calc = U @ V.T is not quite right because W depends on c.
        # T_calc is a 3-tensor.
        # Let's use einsum. It's the most robust way.
        
        try:
            T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        except Exception:
            return False

        # Build T_target
        # Dimensions: (n*m, m*p, p*n)
        T_target = np.zeros((n * m, m * p, p * n), dtype=np.int8)
        
        # Indices:
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1

        # Compare
        if not np.allclose(T_calc, T_target, atol=1e-6):
            return False
            
        return True

    def build_rank_85_decomposition():
        """
        Constructs the U, V, W matrices for rank 85.
        """
        rank = 85
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        
        # --- Part 1: 4x4x4 Block (Strassen) ---
        # Indices:
        # A: rows 0-3, cols 0-3 -> global U index: i*5 + j
        # B: rows 0-3, cols 0-3 -> global V index: j*5 + k
        # C: rows 0-3, cols 0-3 -> global W index: k*4 + i
        
        # Strassen 2x2 factors (for 2x2 matrices)
        # We need to map 4x4 (2x2 block of 2x2) to these.
        # Actually, it's easier to use the 4x4 Strassen directly or compose 2x2s.
        # Let's use the explicit 7-term formulas for 4x4 multiplication 
        # which decomposes into 7 products of 2x2 blocks? No, standard Strassen is for 2x2.
        # For 4x4, we can view it as 2x2 blocks of size 2x2.
        # Let A = [[A00, A01], [A10, A11]] where Ai are 2x2.
        # Then C = A*B.
        # The 7 scalar products in Strassen are replaced by 7 matrix products of 2x2 matrices.
        # Each 2x2 matrix product requires rank 7.
        # So total rank for 4x4x4 is 7 * 7 = 49.
        
        # Let's define the 2x2 Strassen factors (rank 7)
        # U_2x2, V_2x2, W_2x2 of shape (4, 7)
        # Indices for 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        s_u = np.zeros((4, 7), dtype=np.int8)
        s_v = np.zeros((4, 7), dtype=np.int8)
        s_w = np.zeros((4, 7), dtype=np.int8)
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        s_u[0, 0] = 1; s_u[3, 0] = 1
        s_v[0, 0] = 1; s_v[3, 0] = 1
        s_w[0, 0] = 1; s_w[3, 0] = 1
        
        # Term 1: M2 = (A10+A11)B00
        s_u[2, 1] = 1; s_u[3, 1] = 1
        s_v[0, 1] = 1
        s_w[2, 1] = 1; s_w[3, 1] = -1
        
        # Term 2: M3 = A00(B01-B11)
        s_u[0, 2] = 1
        s_v[1, 2] = 1; s_v[3, 2] = -1
        s_w[1, 2] = 1; s_w[3, 2] = 1
        
        # Term 3: M4 = A11(B10-B00)
        s_u[3, 3] = 1
        s_v[2, 3] = 1; s_v[0, 3] = -1
        s_w[0, 3] = 1; s_w[2, 3] = 1
        
        # Term 4: M5 = (A00+A01)B11
        s_u[0, 4] = 1; s_u[1, 4] = 1
        s_v[3, 4] = 1
        s_w[0, 4] = -1; s_w[1, 4] = 1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        s_u[2, 5] = 1; s_u[0, 5] = -1
        s_v[0, 5] = 1; s_v[1, 5] = 1
        s_w[3, 5] = 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        s_u[1, 6] = 1; s_u[3, 6] = -1
        s_v[2, 6] = 1; s_v[3, 6] = 1
        s_w[0, 6] = 1

        # Now construct the 4x4x4 decomposition using these 2x2 blocks.
        # The 4x4 matrices are indexed by (block_row, block_col, local_row, local_col)
        # Global index for A: i*5 + j.
        # For 4x4 block (cols 0-3), i in 0-3, j in 0-3.
        # Block row br = i//2, block col bc = i%2? No.
        # Let's map local 2x2 indices to global 4x4 indices.
        # Local 2x2 index l in 0..3.
        # If we treat 4x4 as 2x2 blocks of 2x2:
        # Block (0,0) covers global (0,0),(0,1),(1,0),(1,1)
        # Block (0,1) covers global (0,2),(0,3),(1,2),(1,3)
        # Block (1,0) covers global (2,0),(2,1),(3,0),(3,1)
        # Block (1,1) covers global (2,2),(2,3),(3,2),(3,3)
        
        # We need to combine the 7 Strassen terms (each involving 2x2 matrix multiplications)
        # into the global U, V, W vectors.
        
        # For each of the 7 Strassen terms t (0..6):
        #   The 2x2 matrix product uses factors from s_u[:, t], s_v[:, t], s_w[:, t]
        #   However, these are for *scalar* multiplication of 2x2 blocks? No.
        #   Strassen for 4x4:
        #   Let A = [[A00, A01], [A10, A11]]
        #   The 7 terms M1..M7 are 2x2 matrix products.
        #   Each M_k = X_k * Y_k where X_k, Y_k are linear combos of A00..A11 and B00..B11.
        #   Wait, the standard Strassen works on scalars. For matrices, the "scalar" multiplication becomes matrix multiplication.
        #   So if we decompose 4x4 into 2x2 blocks, and apply Strassen, we get 7 matrix multiplications of size 2x2.
        #   Each 2x2 matrix multiplication has rank 7 (over integers? No, rank 7 is for field).
        #   So total rank is 7 * 7 = 49.
        
        # Let's iterate through the 7 Strassen terms for the 4x4 level.
        # Term t (0..6) corresponds to a specific linear combination of blocks.
        # We need the coefficients for the blocks A_br,bc and B_br,bc.
        
        # The coefficients are derived from the 2x2 Strassen applied to the blocks.
        # Let's define the block-level Strassen coefficients.
        # Same as s_u, s_v, s_w but now each "entry" is a 2x2 matrix of coefficients?
        # No, s_u[r, t] is a scalar coefficient for the element r in the 2x2 matrix.
        # But here, the "elements" are 2x2 matrices.
        # So for Term t, the matrix X_t is sum_{br,bc} coeff_u[br, bc, t] * A_{br, bc}
        # And Y_t is sum_{br,bc} coeff_v[br, bc, t] * B_{br, bc}
        
        # We can just reuse s_u, s_v, s_w as the coefficients for the block indices.
        # Block index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
        
        for t in range(7):
            # Get block coefficients for U, V, W
            u_blocks = s_u[:, t] # shape (4,)
            v_blocks = s_v[:, t] # shape (4,)
            w_blocks = s_w[:, t] # shape (4,)
            
            # Now, each "block" multiplication is a 2x2 matrix multiplication.
            # We need to decompose this 2x2 matrix multiplication into 7 rank-1 terms.
            # Let's use the same s_u, s_v, s_w for the *inner* 2x2 decomposition.
            # Wait, this creates a Kronecker product structure.
            # Rank 49 = 7 * 7.
            
            for inner_t in range(7):
                col_idx = t * 7 + inner_t
                
                inner_u = s_u[:, inner_t] # shape (4,)
                inner_v = s_v[:, inner_t] # shape (4,)
                inner_w = s_w[:, inner_t] # shape (4,)
                
                # Fill U
                for br in range(2):
                    for bc in range(2):
                        block_idx = br * 2 + bc
                        coeff_outer = u_blocks[block_idx]
                        if coeff_outer != 0:
                            # Block A_{br, bc} occupies global rows/cols
                            # Global i = br*2 + lr, Global j = bc*2 + lc
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    coeff_inner = inner_u[local_idx]
                                    if coeff_inner != 0:
                                        global_i = br * 2 + lr
                                        global_j = bc * 2 + lc
                                        u_row = global_i * 5 + global_j
                                        U[u_row, col_idx] += coeff_outer * coeff_inner
                                    
                # Fill V
                for br in range(2):
                    for bc in range(2):
                        block_idx = br * 2 + bc
                        coeff_outer = v_blocks[block_idx]
                        if coeff_outer != 0:
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    coeff_inner = inner_v[local_idx]
                                    if coeff_inner != 0:
                                        global_j = br * 2 + lr
                                        global_k = bc * 2 + lc
                                        v_row = global_j * 5 + global_k
                                        V[v_row, col_idx] += coeff_outer * coeff_inner

                # Fill W
                for br in range(2):
                    for bc in range(2):
                        block_idx = br * 2 + bc
                        coeff_outer = w_blocks[block_idx]
                        if coeff_outer != 0:
                            for lr in range(2):
                                for lc in range(2):
                                    local_idx = lr * 2 + lc
                                    coeff_inner = inner_w[local_idx]
                                    if coeff_inner != 0:
                                        global_k = bc * 2 + lc
                                        global_i = br * 2 + lr
                                        w_row = global_k * 4 + global_i
                                        W[w_row, col_idx] += coeff_outer * coeff_inner

        # --- Part 2: 4x4x1 Block ---
        # A (4x4) * B (4x1) -> C (4x1)
        # Global A: i*5 + j (j=0..3)
        # Global B: j*5 + 4 (j=0..3, col=4)
        # Global C: 4*4 + i (col=4, row=i)
        
        start_col = 49
        for i in range(4):
            for j in range(4):
                U[i * 5 + j, start_col] = 1
                V[j * 5 + 4, start_col] = 1
                W[4 * 4 + i, start_col] = 1
                start_col += 1
        
        # --- Part 3: 4x1x4 Block ---
        # A (4x1) * B (1x4) -> C (4x4)
        # Global A: i*5 + 4 (row=i, col=4)
        # Global B: 4*5 + k (row=4, col=k)
        # Global C: k*4 + i (row=k, col=i)
        
        for i in range(4):
            for k in range(4):
                U[i * 5 + 4, start_col] = 1
                V[4 * 5 + k, start_col] = 1
                W[k * 4 + i, start_col] = 1
                start_col += 1
        
        # --- Part 4: 4x1x1 Block ---
        # A (4x1) * B (1x1) -> C (4x1)
        # Global A: i*5 + 4
        # Global B: 4*5 + 4
        # Global C: 4*4 + i
        
        for i in range(4):
            U[i * 5 + 4, start_col] = 1
            V[4 * 5 + 4, start_col] = 1
            W[4 * 4 + i, start_col] = 1
            start_col += 1
            
        return U, V, W

    # Attempt optimized construction
    if time.time() < deadline:
        U, V, W = build_rank_85_decomposition()
        if verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback if verification fails (should not happen)
            pass

    # Fallback: Schoolbook Rank 100
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int8)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int8)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
