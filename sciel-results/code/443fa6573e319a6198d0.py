import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # We expect rank 56
    # We generate the decomposition recursively
    terms = []
    
    # Helper to add a term
    # u: vector of length n*m (4)
    # v: vector of length m*p (32)
    # w: vector of length p*n (32)
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Recursive function to generate terms for M_{2, 2, p}
    # p must be power of 2
    # u, v, w are dictionaries or sparse representations?
    # Given small size, we can use lists of (index, value) or just dense arrays and sum them up later.
    # But we need to output U, V, W matrices.
    # Let's collect the linear forms.
    
    # We will build U, V, W as lists of columns.
    
    def gen_terms(p_val):
        # Returns list of (u_vec, v_vec, w_vec)
        # u_vec is length 4
        # v_vec is length 2 * p_val
        # w_vec is length p_val * 2
        
        if p_val == 2:
            # Base case: M_{2,2,2} using Strassen
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C = A B
            # Strassen:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A10 + A11) * B00
            # M3 = A00 * (B01 - B11)
            # M4 = A11 * (B10 - B00)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B01)
            # M7 = (A01 - A11) * (B10 + B11)
            
            # Indices for A (u): 00->0, 01->1, 10->2, 11->3
            # Indices for B (v): 00->0, 01->1, 10->2, 11->3 (for p=2, m=2, v len 4)
            # Wait, v index is j*p + k.
            # j=0, k=0 -> 0
            # j=0, k=1 -> 1
            # j=1, k=0 -> 2
            # j=1, k=1 -> 3
            
            # Indices for C (w): k*n + i
            # k=0, i=0 -> 0
            # k=0, i=1 -> 1
            # k=1, i=0 -> 2
            # k=1, i=1 -> 3
            
            # M1
            u1 = [1, 0, 0, 1]
            v1 = [1, 0, 0, 1]
            w1 = [1, 0, 0, 1] # C00 += M1, C11 += M1
            
            # M2
            u2 = [0, 0, 1, 1]
            v2 = [1, 0, 0, 0]
            w2 = [0, 1, 0, 0] # C10 += M2
            
            # M3
            u3 = [1, 0, 0, 0]
            v3 = [0, 1, 0, -1]
            w3 = [0, 0, 1, 0] # C01 += M3
            
            # M4
            u4 = [0, 0, 0, 1]
            v4 = [0, 0, 1, -1]
            w4 = [0, 1, 0, 0] # C10 += M4
            
            # M5
            u5 = [1, 1, 0, 0]
            v5 = [0, 0, 0, 1]
            w5 = [0, 0, 0, 1] # C11 += M5
            
            # M6
            u6 = [0, 0, 1, -1] # A10 - A00
            v6 = [1, 1, 0, 0]
            w6 = [0, 0, 1, 0] # C01 += M6 ? 
            # Let's check Strassen formulas carefully.
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 + M7
            
            # Re-evaluating w vectors based on standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 + M7
            
            # Let's construct the 7 terms properly.
            base_terms = [
                # u, v, w
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
                ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # M3
                ([0, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 0]), # M4
                ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]), # M5
                ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]), # M6 -> C00 += M6
                ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])  # M7 -> C00 -= M7, C11 += M7
            ]
            return base_terms
        else:
            # Recursive step
            # Split p into p/2
            half_p = p_val // 2
            # We compute A * (B_L + B_R) and A * (B_L - B_R)
            # B_L is cols 0..half_p-1, B_R is cols half_p..p-1
            
            # Get terms for M_{2,2,half_p}
            sub_terms = gen_terms(half_p)
            
            new_terms = []
            for u, v_sub, w_sub in sub_terms:
                # v_sub is length 2 * half_p
                # We need to map v_sub to v_left and v_right
                # v_left corresponds to B_L, v_right to B_R
                
                # For the sum case:
                # We want to compute A * (B_L + B_R).
                # This is equivalent to using v_sum where v_sum has B_L in first half and B_R in second half?
                # No, the multiplication is A * B.
                # If we call the sub-routine on B_sum = B_L + B_R, the v vector of the sub-routine
                # will have non-zero entries corresponding to B_sum.
                # B_sum[j, k] = B_L[j, k] + B_R[j, k].
                # In the full V vector (length 2*p), B_L is at indices [0, 2*half_p) and B_R at [2*half_p, 2*p).
                # Wait, V index is j*p + k.
                # For sub-problem with p'=half_p, V' index is j*p' + k'.
                # k' corresponds to local k.
                # For B_L, global k = k'. For B_R, global k = k' + half_p.
                
                # So for a term in sub-problem with v_sub[j*p' + k'],
                # In the sum case, we add to V at j*p + k' AND j*p + (k' + half_p).
                # In the diff case, we add to V at j*p + k' AND subtract at j*p + (k' + half_p).
                
                # Let's create v_sum and v_diff
                v_sum = [0] * (2 * p_val)
                v_diff = [0] * (2 * p_val)
                
                for idx, val in enumerate(v_sub):
                    if val != 0:
                        j_sub = idx // half_p
                        k_sub = idx % half_p
                        
                        # Global indices
                        idx_L = j_sub * p_val + k_sub
                        idx_R = j_sub * p_val + (k_sub + half_p)
                        
                        v_sum[idx_L] += val
                        v_sum[idx_R] += val
                        
                        v_diff[idx_L] += val
                        v_diff[idx_R] -= val
                        
                # w_sub is length half_p * 2
                # w_sub index is k_sub * 2 + i
                # In full W (length p_val * 2), index is k_global * 2 + i
                # For sum/diff, the output C is for the whole B.
                # But we are computing C_sum = A * (B_L + B_R) and C_diff = A * (B_L - B_R).
                # The w vectors from sub-routine correspond to the output of the sub-multiplication.
                # The output of sub-multiplication is C_sub (size 2 x half_p).
                # This C_sub is exactly the contribution to C_global for the respective k range?
                # No. C_global = A * B_global.
                # C_sum = A * (B_L + B_R) = C_L + C_R.
                # C_diff = A * (B_L - B_R) = C_L - C_R.
                # So C_L = (C_sum + C_diff) / 2
                # C_R = (C_sum - C_diff) / 2
                
                # The w_sub from the sub-routine represents the linear combination of C_sub entries.
                # We need to map w_sub to w_full.
                # w_sub[k_sub * 2 + i] corresponds to C_sub[k_sub, i].
                # In C_sum, this value is added to both C_L[k_sub, i] and C_R[k_sub, i]?
                # No. C_sum[k, i] = C_L[k, i] + C_R[k, i].
                # So if w_sub has coefficient $c$ for $C_{sub}[k', i]$,
                # then in the sum term, we add $c$ to $W[k' \cdot 2 + i]$ and $c$ to $W[(k' + half_p) \cdot 2 + i]$.
                # In the diff term, we add $c$ to $W[k' \cdot 2 + i]$ and $-c$ to $W[(k' + half_p) \cdot 2 + i]$.
                
                w_sum = [0] * (2 * p_val)
                w_diff = [0] * (2 * p_val)
                
                for idx, val in enumerate(w_sub):
                    if val != 0:
                        k_sub = idx // 2
                        i_sub = idx % 2
                        
                        idx_L = k_sub * 2 + i_sub
                        idx_R = (k_sub + half_p) * 2 + i_sub
                        
                        w_sum[idx_L] += val
                        w_sum[idx_R] += val
                        
                        w_diff[idx_L] += val
                        w_diff[idx_R] -= val
                        
                # Now add the two new terms
                # Term 1: u * v_sum -> w_sum
                # Term 2: u * v_diff -> w_diff
                # Note: u is the same.
                
                new_terms.append((u, v_sum, w_sum))
                new_terms.append((u, v_diff, w_diff))
                
            return new_terms

    all_terms = gen_terms(p)
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
            
    return {"U": U, "V": V, "W": W}
