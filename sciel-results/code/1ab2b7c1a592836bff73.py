"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    # Schoolbook decomposition (rank 40) as fallback
    U_school = np.zeros((n * m, n * m * p))
    V_school = np.zeros((m * p, n * m * p))
    W_school = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
                
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check all 40 one-positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that zero positions are zero
        for idx_u in range(n * m):
            i, j = divmod(idx_u, m)
            for idx_v in range(m * p):
                j2, k = divmod(idx_v, p)
                if j2 != j:
                    continue
                for idx_w in range(p * n):
                    k2, i2 = divmod(idx_w, n)
                    if k2 != k or i2 != i:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-10:
                            return False
        return True

    # Strategy: Global construction for mu_{2,4,5}
    # We use the identity R(mu_{2,4,5}) <= 2 * R(mu_{2,2,5})
    # First, we need a rank-minimal decomposition for mu_{2,2,5}.
    # mu_{2,2,5} can be seen as mu_{2,2,2} x mu_{1,1,3} roughly? No.
    # Let's use the known bound: R(mu_{2,2,5}) <= R(mu_{2,2,2}) + R(mu_{2,2,3})?
    # Actually, a better decomposition for 2x2x5:
    # Split k into {0,1} and {2,3,4}.
    # mu_{2,2,2} has rank 7 (Strassen).
    # mu_{2,2,3} has rank 12 (Schoolbook 2*2*3=12, can it be lower? 
    #    R(mu_{2,2,3}) is known to be 12? 
    #    Let's check: 2x2x3 tensor. 
    #    Lower bound: max(4, 4, 6) = 6. 
    #    Upper bound: 12. 
    #    Is there a rank 11 or 10? 
    #    According to literature, R(mu_{2,2,3}) = 12 is not necessarily minimal.
    #    However, for 2x2x3, we can use Strassen-like? 
    #    Actually, 2x2x3 is small. 
    #    Let's try to construct a rank 12 decomposition for 2x2x3 first (schoolbook) 
    #    and see if we can optimize.
    
    # Alternative: Use the "doubling" trick directly on the full 2x4x5.
    # If we have a decomposition for 2x2x5 of rank R, we can get 2x4x5 of rank 2R.
    # Let's find a good decomposition for 2x2x5.
    # 2x2x5: Split k into k1 (size 2) and k2 (size 3).
    # Term 1: mu_{2,2,2} (Rank 7)
    # Term 2: mu_{2,2,3} (Rank ?)
    
    # Let's assume R(mu_{2,2,3}) = 12 for now (schoolbook).
    # Then R(mu_{2,2,5}) <= 7 + 12 = 19.
    # Then R(mu_{2,4,5}) <= 2 * 19 = 38. This is the same as the current strategy!
    
    # To improve, we need R(mu_{2,2,3}) < 12 or a better coupling.
    # Is R(mu_{2,2,3}) < 12?
    # The tensor mu_{2,2,3} has dimensions 4x6x2.
    # Flattening to 4x12 gives rank <= 4. 
    # Flattening to 6x8 gives rank <= 6.
    # Flattening to 2x24 gives rank <= 2.
    # So rank is at least 6.
    # Can we do better than 12?
    # Let's try to construct a rank 10 or 11 decomposition for 2x2x3.
    # If we find rank 10, then R(mu_{2,2,5}) <= 17, and R(mu_{2,4,5}) <= 34.
    # If we find rank 9, then R(mu_{2,2,5}) <= 16, and R(mu_{2,4,5}) <= 32.
    
    # Let's attempt to find a rank 10 decomposition for 2x2x3.
    # This is a specific sub-problem.
    # Since I cannot run a solver here, I will implement a search for a small rank 
    # decomposition for 2x2x3 using random initialization and gradient descent 
    # or just try a few known structures.
    
    # However, the current program already achieves 38.
    # To beat 38, I need a total rank < 38.
    # If I use the 2+2 split on m, I get 38.
    # If I use the 1+3 split on m?
    # mu_{1,1,5} has rank 5 (schoolbook).
    # mu_{1,3,5} ? 
    # R(mu_{1,3,5}) <= R(mu_{1,3,2}) + R(mu_{1,3,3})?
    # This seems complicated.
    
    # Let's try a different structural change:
    # Use the "Kronecker product" structure more aggressively.
    # The tensor mu_{2,4,5} can be viewed as mu_{2,4,2} x mu_{1,1,1} + ... no.
    
    # Let's try to optimize the 2x2x3 block.
    # If I can get 2x2x3 to rank 11, total rank = 2 * (7 + 11) = 36.
    # If I can get 2x2x3 to rank 10, total rank = 2 * (7 + 10) = 34.
    
    # I will implement a simple local search to find a rank 10 or 11 
    # decomposition for 2x2x3.
    
    def find_rank_r_decomp_223(r_target):
        """
        Attempt to find a rank r decomposition for mu_{2,2,3}.
        Returns U, V, W if found, else None.
        """
        if time.time() > deadline - 10:
            return None
            
        # Random initialization
        np.random.seed(42)
        # Shapes: U (4, r), V (6, r), W (2, r)
        U = np.random.randn(4, r_target)
        V = np.random.randn(6, r_target)
        W = np.random.randn(2, r_target)
        
        # Target tensor T (4, 6, 2)
        T = np.zeros((4, 6, 2))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i*2+j, j*3+k, k*2+i] = 1.0
        
        # Objective: minimize sum (T - U*V*W)^2
        # Use simple gradient descent or Nelder-Mead?
        # Nelder-Mead is slow for high dimensions.
        # Let's try a few random restarts with a simple heuristic.
        
        # Actually, let's just try the schoolbook rank 12 for 2x2x3 
        # and see if we can manually find a rank 11.
        # 
        # Known result: R(mu_{2,2,3}) is 12? 
        # A paper by Landsberg says R(mu_{2,2,3}) <= 12. 
        # Is it 11? 
        # The border rank is 9. 
        # So rank 10 or 11 might be possible.
        
        # Given the time constraints, I will stick to the 38 rank 
        # unless I can quickly verify a 36 rank.
        
        # Let's try to construct a rank 11 for 2x2x3 manually.
        # If it fails, return None.
        
        # For now, I will return None to indicate "not found" 
        # and rely on the 38 rank fallback.
        # BUT, the directive asks for a STRUCTURAL change.
        # The current program IS the 38 rank.
        # I need to try something else.
        
        # What if I use a rank 7 for 2x2x2 and a rank 11 for 2x2x3?
        # Total 36.
        
        # Let's try to hard-code a rank 11 decomposition for 2x2x3 if I can find one.
        # Since I don't have one memorized, I will implement a quick search.
        
        # Search for rank 11
        for _ in range(5): # 5 restarts
            if time.time() > deadline - 5:
                break
            U = np.random.randn(4, 11)
            V = np.random.randn(6, 11)
            W = np.random.randn(2, 11)
            
            # Simple gradient step
            lr = 0.01
            for step in range(100):
                # Compute residual
                R = np.zeros((4, 6, 2))
                for c in range(11):
                    R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                
                err = R - T
                if np.linalg.norm(err) < 1e-5:
                    return U, V, W
                
                # Gradient w.r.t U[:, c]
                grad_U = np.zeros((4, 11))
                grad_V = np.zeros((6, 11))
                grad_W = np.zeros((2, 11))
                
                for c in range(11):
                    # d/dU_a err_abc = V_b W_c err_abc
                    grad_U[:, c] = np.einsum('abc,bc->a', err, np.outer(V[:, c], W[:, c]))
                    grad_V[:, c] = np.einsum('abc,ac->b', err, np.outer(U[:, c], W[:, c]))
                    grad_W[:, c] = np.einsum('abc,ab->c', err, np.outer(U[:, c], V[:, c]))
                    
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
        return None

    # Try to find a better 2x2x3 decomposition
    decomp_223 = find_rank_r_decomp_223(11)
    
    if decomp_223 is not None:
        U223, V223, W223 = decomp_223
        rank_223 = 11
    else:
        # Fallback to schoolbook 2x2x3 (Rank 12)
        U223 = np.zeros((4, 12))
        V223 = np.zeros((6, 12))
        W223 = np.zeros((2, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U223[i*2+j, col] = 1.0
                    V223[j*3+k, col] = 1.0
                    W223[k*2+i, col] = 1.0
                    col += 1
        rank_223 = 12

    # Now construct the full 2x4x5 decomposition
    # Split m into two blocks of 2.
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    
    # For each block, split k into {0, 1} (Strassen) and {2, 3, 4} (2x2x3)
    
    rank_strassen = 7
    total_rank = 2 * (rank_strassen + rank_223)
    
    U_new = np.zeros((n * m, total_rank))
    V_new = np.zeros((m * p, total_rank))
    W_new = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    def add_strassen(j_offset, k_offset):
        nonlocal col_offset
        # Strassen multipliers for 2x2x2
        # ... (same as before)
        
        # Term 1: M1
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[0*4 + (0+j_offset)] = 1.0; u1[1*4 + (1+j_offset)] = 1.0
        v1[(0+j_offset)*5 + (0+k_offset)] = 1.0; v1[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w1[(0+k_offset)*2 + 0] = 1.0; w1[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u1; V_new[:, col_offset] = v1; W_new[:, col_offset] = w1
        col_offset += 1
        
        # Term 2: M2
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[1*4 + (0+j_offset)] = 1.0; u2[1*4 + (1+j_offset)] = 1.0
        v2[(0+j_offset)*5 + (0+k_offset)] = 1.0
        w2[(0+k_offset)*2 + 1] = 1.0; w2[(1+k_offset)*2 + 1] = -1.0
        U_new[:, col_offset] = u2; V_new[:, col_offset] = v2; W_new[:, col_offset] = w2
        col_offset += 1
        
        # Term 3: M3
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[0*4 + (0+j_offset)] = 1.0
        v3[(0+j_offset)*5 + (1+k_offset)] = 1.0; v3[(1+j_offset)*5 + (1+k_offset)] = -1.0
        w3[(1+k_offset)*2 + 0] = 1.0; w3[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u3; V_new[:, col_offset] = v3; W_new[:, col_offset] = w3
        col_offset += 1
        
        # Term 4: M4
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[1*4 + (1+j_offset)] = 1.0
        v4[(1+j_offset)*5 + (0+k_offset)] = 1.0; v4[(0+j_offset)*5 + (0+k_offset)] = -1.0
        w4[(0+k_offset)*2 + 0] = 1.0; w4[(0+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u4; V_new[:, col_offset] = v4; W_new[:, col_offset] = w4
        col_offset += 1
        
        # Term 5: M5
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[0*4 + (0+j_offset)] = 1.0; u5[0*4 + (1+j_offset)] = 1.0
        v5[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w5[(0+k_offset)*2 + 0] = -1.0; w5[(1+k_offset)*2 + 0] = 1.0
        U_new[:, col_offset] = u5; V_new[:, col_offset] = v5; W_new[:, col_offset] = w5
        col_offset += 1
        
        # Term 6: M6
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[1*4 + (0+j_offset)] = 1.0; u6[0*4 + (0+j_offset)] = -1.0
        v6[(0+j_offset)*5 + (0+k_offset)] = 1.0; v6[(0+j_offset)*5 + (1+k_offset)] = 1.0
        w6[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u6; V_new[:, col_offset] = v6; W_new[:, col_offset] = w6
        col_offset += 1
        
        # Term 7: M7
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[0*4 + (1+j_offset)] = 1.0; u7[1*4 + (1+j_offset)] = -1.0
        v7[(1+j_offset)*5 + (0+k_offset)] = 1.0; v7[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w7[(0+k_offset)*2 + 0] = 1.0
        U_new[:, col_offset] = u7; V_new[:, col_offset] = v7; W_new[:, col_offset] = w7
        col_offset += 1

    def add_mu_223(j_offset, k_offset):
        nonlocal col_offset
        # Add the 2x2x3 decomposition found above
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1,2} (local k)
        # Global indices:
        # U: i*4 + (j + j_offset)
        # V: (j + j_offset)*5 + (k + k_offset)
        # W: (k + k_offset)*2 + i
        
        r_sub = U223.shape[1]
        for c in range(r_sub):
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # Map local U indices (4) to global
            for li in range(4):
                li_i, li_j = divmod(li, 2)
                global_u = li_i * 4 + (li_j + j_offset)
                u_vec[global_u] = U223[li, c]
                
            # Map local V indices (6) to global
            for lv in range(6):
                lv_j, lv_k = divmod(lv, 3)
                global_v = (lv_j + j_offset) * 5 + (lv_k + k_offset)
                v_vec[global_v] = V223[lv, c]
                
            # Map local W indices (2) to global
            for lw in range(2):
                lw_k, lw_i = divmod(lw, 2)
                global_w = (lw_k + k_offset) * 2 + lw_i
                w_vec[global_w] = W223[lw, c]
                
            U_new[:, col_offset] = u_vec
            V_new[:, col_offset] = v_vec
            W_new[:, col_offset] = w_vec
            col_offset += 1

    # Apply to Block 1 (j_offset=0)
    add_strassen(0, 0)       # k in {0,1}
    add_mu_223(0, 2)         # k in {2,3,4}
    
    # Apply to Block 2 (j_offset=2)
    add_strassen(2, 0)       # k in {0,1}
    add_mu_223(2, 2)         # k in {2,3,4}
    
    # Verify new decomposition
    if verify(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
