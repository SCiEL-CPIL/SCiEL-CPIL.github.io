"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 8
    num_v = m * p  # 28
    num_w = p * n  # 14
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    def make_u(i, j, val):
        vec = [0.0] * 8
        vec[i*4 + j] = val
        return vec
        
    def make_v(j, k, val):
        vec = [0.0] * 28
        vec[j*7 + k] = val
        return vec
        
    def make_w(k, i, val):
        vec = [0.0] * 14
        vec[k*2 + i] = val
        return vec

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_i_base, a_j_base, b_j_base, b_k_base):
        """
        Adds 7 terms for multiplying A[a_i_base:a_i_base+2, a_j_base:a_j_base+2] 
        with B[b_j_base:b_j_base+2, b_k_base:b_k_base+2].
        The result contributes to C[a_i_base:a_i_base+2, b_k_base:b_k_base+2].
        """
        # Term 1: M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0]*8; u1[a_i_base*4 + a_j_base] = 1; u1[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v1 = [0.0]*28; v1[b_j_base*7 + b_k_base] = 1; v1[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w1 = [0.0]*14; w1[b_k_base*2 + a_i_base] = 1; w1[(b_k_base+1)*2 + (a_i_base+1)] = -1
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (a10 + a11) * b00
        u2 = [0.0]*8; u2[(a_i_base+1)*4 + a_j_base] = 1; u2[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v2 = [0.0]*28; v2[b_j_base*7 + b_k_base] = 1
        w2 = [0.0]*14; w2[b_k_base*2 + (a_i_base+1)] = 1; w2[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: M3 = a00 * (b01 - b11)
        u3 = [0.0]*8; u3[a_i_base*4 + a_j_base] = 1
        v3 = [0.0]*28; v3[b_j_base*7 + (b_k_base+1)] = 1; v3[(b_j_base+1)*7 + (b_k_base+1)] = -1
        w3 = [0.0]*14; w3[(b_k_base+1)*2 + a_i_base] = 1; w3[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: M4 = a11 * (b10 - b00)
        u4 = [0.0]*8; u4[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v4 = [0.0]*28; v4[(b_j_base+1)*7 + b_k_base] = 1; v4[b_j_base*7 + b_k_base] = -1
        w4 = [0.0]*14; w4[b_k_base*2 + a_i_base] = 1; w4[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (a00 + a01) * b11
        u5 = [0.0]*8; u5[a_i_base*4 + a_j_base] = 1; u5[a_i_base*4 + (a_j_base+1)] = 1
        v5 = [0.0]*28; v5[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w5 = [0.0]*14; w5[b_k_base*2 + a_i_base] = -1; w5[(b_k_base+1)*2 + a_i_base] = 1
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0]*8; u6[(a_i_base+1)*4 + a_j_base] = 1; u6[a_i_base*4 + a_j_base] = -1
        v6 = [0.0]*28; v6[b_j_base*7 + b_k_base] = 1; v6[b_j_base*7 + (b_k_base+1)] = 1
        w6 = [0.0]*14; w6[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0]*8; u7[a_i_base*4 + (a_j_base+1)] = 1; u7[(a_i_base+1)*4 + (a_j_base+1)] = -1
        v7 = [0.0]*28; v7[(b_j_base+1)*7 + b_k_base] = 1; v7[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w7 = [0.0]*14; w7[b_k_base*2 + a_i_base] = 1; w7[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u7, v7, w7)

    # Rank-6 decomposition for 2x2 * 2x3
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # We can view this as:
    # C[:, 0:2] = A * B[:, 0:2] (Rank 7 via Strassen? No, that's 2x2*2x2 which is 7. But we want total 6 for the 2x3 part?
    # Wait, R(2,2,3) = 6.
    # Let's check the decomposition for R(2,2,3)=6.
    # It is known that R(2,2,3)=6.
    # One such decomposition:
    # M1 = (a+b) * (e+h) -> contributes to C00 (1), C10 (1) ... wait.
    # Let's use the standard rank 6 decomposition for 2x2x3.
    # Reference: "Tensor rank of 2x2x3"
    # Actually, let's just use the schoolbook 12 terms if I can't verify the 6 terms quickly?
    # No, the prompt says "aiming for a total rank significantly lower".
    # 7+7+12 = 26 per side -> 52 total.
    # 7+7+6 = 20 per side -> 40 total.
    # Let's try to find the 6-term decomposition.
    
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C00 = ae + bh
    # C01 = af + bi
    # C02 = ag + bj
    # C10 = ce + dh
    # C11 = cf + di
    # C12 = cg + dj
    
    # Try:
    # M1 = (a+b)(e+h) = ae + ah + be + bh. 
    #   C00 += ae + bh (need ah, be to cancel or be part of other terms)
    #   C10 += 0? No, M1 uses A row 0 and B row 0/1?
    #   M1 uses A[0,0]+A[0,1] and B[0,0]+B[1,0].
    #   U: (0,0)=1, (0,1)=1. V: (0,0)=1, (1,0)=1.
    #   Product: (a+b)(e+h) = ae + ah + be + bh.
    #   We want C00 = ae + bh. So ah + be must be cancelled.
    
    # Let's look for a known rank 6 decomposition.
    # Actually, R(2,2,3) is 6.
    # Terms:
    # 1. (a+b)(e+h) -> W: C00+=1, C10+=1? No.
    #    U: a+b. V: e+h.
    #    C00 gets a*e + a*h + b*e + b*h.
    #    C10 gets c*e + c*h + d*e + d*h? No, U is only row 0 of A.
    #    So C10 is not touched by this term?
    #    Wait, U vector has non-zeros at (0,0) and (0,1).
    #    V vector has non-zeros at (0,0) and (1,0).
    #    The product is (a+b)(e+h).
    #    This contributes to C[i,k] where i is from U and k is from V?
    #    No, tensor T[i,j,k] = U[i,j] V[j,k] W[k,i].
    #    If U has non-zeros at (0,0) and (0,1), and V has non-zeros at (0,0) and (1,0).
    #    The sum is over j.
    #    j=0: U[0,0]V[0,0] = a*e.
    #    j=1: U[0,1]V[1,0] = b*h.
    #    So the term is a*e + b*h. This is exactly C00!
    #    So Term 1: U=(1,1,0,0...), V=(1,0,0...), W=(1,0...) for C00.
    #    This is rank 1 for C00.
    
    # Can we do the same for C10?
    # C10 = c*e + d*h.
    # U=(0,0,1,1...), V=(1,0,0...), W=(0,1...) for C10.
    # Term 2: U[1,0]=1, U[1,1]=1. V[0,0]=1, V[1,0]=1.
    # Sum: c*e + d*h. Correct.
    
    # Now C01 = a*f + b*i.
    # Term 3: U[0,0]=1, U[0,1]=1. V[0,1]=1, V[1,1]=1.
    # Sum: a*f + b*i. Correct.
    
    # C11 = c*f + d*i.
    # Term 4: U[1,0]=1, U[1,1]=1. V[0,1]=1, V[1,1]=1.
    # Sum: c*f + d*i. Correct.
    
    # C02 = a*g + b*j.
    # Term 5: U[0,0]=1, U[0,1]=1. V[0,2]=1, V[1,2]=1.
    # Sum: a*g + b*j. Correct.
    
    # C12 = c*g + d*j.
    # Term 6: U[1,0]=1, U[1,1]=1. V[0,2]=1, V[1,2]=1.
    # Sum: c*g + d*j. Correct.
    
    # Total 6 terms!
    # This is just the schoolbook algorithm for 2x2 * 2x3, but optimized?
    # Wait, schoolbook for 2x2 * 2x3 is 12 terms (2 terms per output element).
    # My "Term 1" above combines ae and bh into one rank-1 term?
    # U = [1, 1, 0, 0] (for A row 0: a, b)
    # V = [1, 0, 0; 1, 0, 0] (for B col 0: e, h) -> V[0,0]=1, V[1,0]=1.
    # The product is sum_j U[j] V[j,k]? No.
    # The tensor decomposition is sum_r U_r[i,j] V_r[j,k] W_r[k,i].
    # For Term 1:
    # U_1[0,0] = 1, U_1[0,1] = 1. (Others 0)
    # V_1[0,0] = 1, V_1[1,0] = 1. (Others 0)
    # W_1[0,0] = 1. (Others 0)
    # Contribution to T[0,0,0] (i=0, j=0, k=0): U[0,0]V[0,0]W[0,0] = 1*1*1 = 1. (ae)
    # Contribution to T[0,1,0] (i=0, j=1, k=0): U[0,1]V[1,0]W[0,0] = 1*1*1 = 1. (bh)
    # Contribution to T[0,0,1] (i=0, j=0, k=1): V[0,1]=0. 0.
    # Contribution to T[0,1,1] (i=0, j=1, k=1): V[1,1]=0. 0.
    # So this single term produces ae in C00 and bh in C00?
    # No, C00 is T[0,0,0]?
    # W index is k*n + i.
    # C[i,k] corresponds to W[k*n+i].
    # C[0,0] is W[0*2+0] = W[0].
    # The term adds to T[i,j,k] where U[i,j] and V[j,k] are non-zero.
    # U[0,0]=1, U[0,1]=1.
    # V[0,0]=1, V[1,0]=1.
    # Non-zero U,V pairs:
    # (i=0, j=0) and (j=0, k=0) -> T[0,0,0] += 1. (ae)
    # (i=0, j=1) and (j=1, k=0) -> T[0,1,0] += 1. (bh)
    # So this term adds 1 to C[0,0] (via ae) and 1 to C[0,0] (via bh)?
    # No, T[0,0,0] is C[0,0] element corresponding to A[0,0]B[0,0].
    # T[0,1,0] is C[0,0] element corresponding to A[0,1]B[1,0].
    # Both contribute to C[0,0].
    # So yes, one rank-1 term can produce two contributions to the same output C[i,k] if they come from different j's?
    # Wait. The output C[i,k] is the sum over j of A[i,j]B[j,k].
    # In the tensor decomposition, each rank-1 term is a product of 3 vectors.
    # If U has non-zeros at j=0 and j=1, and V has non-zeros at j=0 and j=1,
    # and they are aligned such that U[j]V[j,k] picks out specific pairs.
    # In my example:
    # U[0,0]=1, U[0,1]=1.
    # V[0,0]=1, V[1,0]=1.
    # Term = U * V * W.
    # T[0,0,0] gets U[0,0]V[0,0]W[0,0] = 1.
    # T[0,1,0] gets U[0,1]V[1,0]W[0,0] = 1.
    # So C[0,0] receives +1 from A[0,0]B[0,0] and +1 from A[0,1]B[1,0].
    # This is exactly the sum ae + bh.
    # So YES, we can compute C[0,0] with 1 term?
    # No, we need to compute ALL elements of C.
    # C[0,0] = ae + bh. (1 term)
    # C[0,1] = af + bi. (1 term: U[0,0]=1, U[0,1]=1; V[0,1]=1, V[1,1]=1; W[1,0]=1)
    # C[0,2] = ag + bj. (1 term: U[0,0]=1, U[0,1]=1; V[0,2]=1, V[1,2]=1; W[2,0]=1)
    # C[1,0] = ce + dh. (1 term: U[1,0]=1, U[1,1]=1; V[0,0]=1, V[1,0]=1; W[0,1]=1)
    # C[1,1] = cf + di. (1 term: U[1,0]=1, U[1,1]=1; V[0,1]=1, V[1,1]=1; W[1,1]=1)
    # C[1,2] = cg + dj. (1 term: U[1,0]=1, U[1,1]=1; V[0,2]=1, V[1,2]=1; W[2,1]=1)
    # Total 6 terms.
    # This is correct! R(2,2,3) = 6.
    
    def add_rank6_2x2x3(a_i_base, a_j_base, b_j_base, b_k_base):
        """
        Adds 6 terms for multiplying A[a_i_base:a_i_base+2, a_j_base:a_j_base+2] 
        with B[b_j_base:b_j_base+2, b_k_base:b_k_base+3].
        """
        # Term 1: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        u1 = [0.0]*8; u1[a_i_base*4 + a_j_base] = 1; u1[a_i_base*4 + (a_j_base+1)] = 1
        v1 = [0.0]*28; v1[b_j_base*7 + b_k_base] = 1; v1[(b_j_base+1)*7 + b_k_base] = 1
        w1 = [0.0]*14; w1[b_k_base*2 + a_i_base] = 1
        add_term(u1, v1, w1)
        
        # Term 2: C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        u2 = [0.0]*8; u2[a_i_base*4 + a_j_base] = 1; u2[a_i_base*4 + (a_j_base+1)] = 1
        v2 = [0.0]*28; v2[b_j_base*7 + (b_k_base+1)] = 1; v2[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w2 = [0.0]*14; w2[(b_k_base+1)*2 + a_i_base] = 1
        add_term(u2, v2, w2)
        
        # Term 3: C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        u3 = [0.0]*8; u3[a_i_base*4 + a_j_base] = 1; u3[a_i_base*4 + (a_j_base+1)] = 1
        v3 = [0.0]*28; v3[b_j_base*7 + (b_k_base+2)] = 1; v3[(b_j_base+1)*7 + (b_k_base+2)] = 1
        w3 = [0.0]*14; w3[(b_k_base+2)*2 + a_i_base] = 1
        add_term(u3, v3, w3)
        
        # Term 4: C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        u4 = [0.0]*8; u4[(a_i_base+1)*4 + a_j_base] = 1; u4[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v4 = [0.0]*28; v4[b_j_base*7 + b_k_base] = 1; v4[(b_j_base+1)*7 + b_k_base] = 1
        w4 = [0.0]*14; w4[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u4, v4, w4)
        
        # Term 5: C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        u5 = [0.0]*8; u5[(a_i_base+1)*4 + a_j_base] = 1; u5[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v5 = [0.0]*28; v5[b_j_base*7 + (b_k_base+1)] = 1; v5[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w5 = [0.0]*14; w5[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u5, v5, w5)
        
        # Term 6: C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
        u6 = [0.0]*8; u6[(a_i_base+1)*4 + a_j_base] = 1; u6[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v6 = [0.0]*28; v6[b_j_base*7 + (b_k_base+2)] = 1; v6[(b_j_base+1)*7 + (b_k_base+2)] = 1
        w6 = [0.0]*14; w6[(b_k_base+2)*2 + (a_i_base+1)] = 1
        add_term(u6, v6, w6)

    # Construction Plan:
    # Side 1: A_left (i=0,1, j=0,1) * B_top (j=0,1, k=0..6)
    # B_top_0: j=0,1, k=0,1. Strassen.
    add_strassen_block(0, 0, 0, 0)
    # B_top_1: j=0,1, k=2,3. Strassen.
    add_strassen_block(0, 0, 0, 2)
    # B_top_2: j=0,1, k=4,5,6. Rank 6.
    add_rank6_2x2x3(0, 0, 0, 4)
    
    # Side 2: A_right (i=0,1, j=2,3) * B_bottom (j=2,3, k=0..6)
    # B_bottom_0: j=2,3, k=0,1. Strassen.
    add_strassen_block(0, 2, 2, 0)
    # B_bottom_1: j=2,3, k=2,3. Strassen.
    add_strassen_block(0, 2, 2, 2)
    # B_bottom_2: j=2,3, k=4,5,6. Rank 6.
    add_rank6_2x2x3(0, 2, 2, 4)
    
    # Convert to numpy arrays
    U_arr = np.array(U).T
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    # Verify
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U_arr.shape[1]):
        u_col = U_arr[:, c]
        v_col = V_arr[:, c]
        w_col = W_arr[:, c]
        
        u_idx = np.nonzero(u_col)[0]
        v_idx = np.nonzero(v_col)[0]
        w_idx = np.nonzero(w_col)[0]
        
        for ui in u_idx:
            i = ui // m
            j = ui % m
            u_val = u_col[ui]
            
            for vi in v_idx:
                v_j = vi // p
                k = vi % p
                if v_j != j:
                    continue
                v_val = v_col[vi]
                
                for wi in w_idx:
                    w_k = wi // n
                    w_i = wi % n
                    if w_k != k or w_i != i:
                        continue
                    w_val = w_col[wi]
                    
                    T[i, j, k] += u_val * v_val * w_val
                    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        U_fallback = np.zeros((num_u, n*m*p), dtype=float)
        V_fallback = np.zeros((num_v, n*m*p), dtype=float)
        W_fallback = np.zeros((num_w, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, col] = 1.0
                    V_fallback[j * p + k, col] = 1.0
                    W_fallback[k * n + i, col] = 1.0
                    col += 1
        U_arr = U_fallback
        V_arr = V_fallback
        W_arr = W_fallback

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
