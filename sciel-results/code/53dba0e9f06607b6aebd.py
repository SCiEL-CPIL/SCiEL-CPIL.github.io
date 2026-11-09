import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Trivial rank
    R_trivial = n * m * p
    U = np.zeros((n * m, R_trivial))
    V = np.zeros((m * p, R_trivial))
    W = np.zeros((p * n, R_trivial))
    
    # Fill trivial solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    # Identify blocks to apply Strassen
    # Blocks: (i_start, i_end, j_start, j_end, k_start, k_end)
    # We want disjoint 2x2x2 blocks.
    blocks = [
        (0, 2, 0, 2, 0, 2), # i:0-1, j:0-1, k:0-1
        (0, 2, 2, 4, 0, 2), # i:0-1, j:2-3, k:0-1
        (0, 2, 0, 2, 2, 4), # i:0-1, j:0-1, k:2-3
        (0, 2, 2, 4, 2, 4), # i:0-1, j:2-3, k:2-3
    ]
    
    # We will modify U, V, W in place.
    # We need to remove the trivial terms for these blocks and add Strassen terms.
    # Since we are adding columns, we can just append the new terms and zero out the old ones.
    # But appending changes the shape.
    # Better: Create new arrays with size R_trivial - 4*1 (savings) + 4*7 (new terms)?
    # No, each block replaces 8 terms with 7.
    # So we save 1 term per block.
    # Total new rank = 60 - 4 = 56.
    
    # Let's build the new matrices.
    # We'll keep the trivial terms that are NOT in the blocks.
    # And add the Strassen terms.
    
    U_new = []
    V_new = []
    W_new = []
    
    # Add trivial terms not in blocks
    # Create a set of blocked (i,j,k) tuples
    blocked = set()
    for b in blocks:
        i_s, i_e, j_s, j_e, k_s, k_e = b
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    blocked.add((i,j,k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in blocked:
                    col_U = np.zeros(n*m)
                    col_V = np.zeros(m*p)
                    col_W = np.zeros(p*n)
                    col_U[i*m+j] = 1
                    col_V[j*p+k] = 1
                    col_W[k*n+i] = 1
                    U_new.append(col_U)
                    V_new.append(col_V)
                    W_new.append(col_W)
                    
    # Add Strassen terms for each block
    for b in blocks:
        i_s, i_e, j_s, j_e, k_s, k_e = b
        # Local indices 0,1 map to global i_s+i', j_s+j', k_s+k'
        
        # Strassen coefficients for A, B, C
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We define the terms as (coeff_A, coeff_B, coeff_C)
        # coeff_A is a dict {(i,j): val}
        # coeff_B is a dict {(j,k): val}
        # coeff_C is a dict {(i,k): val}
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):-1, (1,0):-1, (1,1):1}), # C00=1, C01=-1, C10=-1, C11=1? 
            # Wait, C = M1 + M4 - M5 + M7 for C00.
            # The tensor decomposition sums over r.
            # So each term r contributes to C_ik with weight W_ik,r.
            # The term is U_r * V_r * W_r.
            # So if M1 contributes to C00 with +1, C01 with -1, etc, we set W accordingly.
            
            # Let's list the contribution of each M to each C_ik.
            # C00: M1(+), M4(+), M5(-), M7(+)
            # C01: M3(+), M5(+)
            # C10: M2(+), M4(+)
            # C11: M1(+), M2(-), M3(+), M6(+)
            
            # So for each M, we define its U, V, and the W vector.
        ]
        
        # M1
        u1 = {(0,0):1, (1,1):1}
        v1 = {(0,0):1, (1,1):1}
        w1 = {(0,0):1, (0,1):-1, (1,0):-1, (1,1):1} # This is wrong. W is fixed for the term.
        # The term is u1 . v1 . w.
        # It adds to C_ik: u1[i,j] * v1[j,k] * w[i,k].
        # We want the sum over all terms to equal the delta functions.
        
        # Let's just define the 7 vectors explicitly.
        
        # Term 1 (M1)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i_s*m + j_s] = 1; u[i_s*m + j_s+1] = 0 # A00 is (i_s, j_s)
        # A00: i=i_s, j=j_s. A11: i=i_s+1, j=j_s+1.
        u[i_s*m + j_s] = 1
        u[(i_s+1)*m + (j_s+1)] = 1
        v[j_s*p + k_s] = 1
        v[(j_s+1)*p + (k_s+1)] = 1
        # W for M1:
        # C00 += 1, C01 += -1, C10 += -1, C11 += 1
        # Local (0,0) -> Global (i_s, k_s)
        # Local (0,1) -> Global (i_s, k_s+1)
        # Local (1,0) -> Global (i_s+1, k_s)
        # Local (1,1) -> Global (i_s+1, k_s+1)
        w[k_s*n + i_s] = 1
        w[(k_s+1)*n + i_s] = -1
        w[k_s*n + (i_s+1)] = -1
        w[(k_s+1)*n + (i_s+1)] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 2 (M2)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A10+A11: i=i_s+1, j=j_s and i=i_s+1, j=j_s+1
        u[(i_s+1)*m + j_s] = 1
        u[(i_s+1)*m + (j_s+1)] = 1
        # B00: j=j_s, k=k_s
        v[j_s*p + k_s] = 1
        # C10 += 1, C11 += -1
        w[k_s*n + (i_s+1)] = 1
        w[(k_s+1)*n + (i_s+1)] = -1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 3 (M3)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A00: i=i_s, j=j_s
        u[i_s*m + j_s] = 1
        # B01-B11: j=j_s, k=k_s+1 and j=j_s+1, k=k_s+1
        v[j_s*p + (k_s+1)] = 1
        v[(j_s+1)*p + (k_s+1)] = -1
        # C01 += 1, C11 += 1
        w[(k_s+1)*n + i_s] = 1
        w[(k_s+1)*n + (i_s+1)] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 4 (M4)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A11: i=i_s+1, j=j_s+1
        u[(i_s+1)*m + (j_s+1)] = 1
        # B10-B00: j=j_s+1, k=k_s and j=j_s, k=k_s
        v[(j_s+1)*p + k_s] = 1
        v[j_s*p + k_s] = -1
        # C00 += 1, C10 += 1
        w[k_s*n + i_s] = 1
        w[k_s*n + (i_s+1)] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 5 (M5)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A00+A01: i=i_s, j=j_s and i=i_s, j=j_s+1
        u[i_s*m + j_s] = 1
        u[i_s*m + (j_s+1)] = 1
        # B11: j=j_s+1, k=k_s+1
        v[(j_s+1)*p + (k_s+1)] = 1
        # C00 += -1, C01 += 1
        w[k_s*n + i_s] = -1
        w[(k_s+1)*n + i_s] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 6 (M6)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A10-A00: i=i_s+1, j=j_s and i=i_s, j=j_s
        u[(i_s+1)*m + j_s] = 1
        u[i_s*m + j_s] = -1
        # B00+B01: j=j_s, k=k_s and j=j_s, k=k_s+1
        v[j_s*p + k_s] = 1
        v[j_s*p + (k_s+1)] = 1
        # C11 += 1
        w[(k_s+1)*n + (i_s+1)] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)
        
        # Term 7 (M7)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # A01-A11: i=i_s, j=j_s+1 and i=i_s+1, j=j_s+1
        u[i_s*m + (j_s+1)] = 1
        u[(i_s+1)*m + (j_s+1)] = -1
        # B10+B11: j=j_s+1, k=k_s and j=j_s+1, k=k_s+1
        v[(j_s+1)*p + k_s] = 1
        v[(j_s+1)*p + (k_s+1)] = 1
        # C00 += 1
        w[k_s*n + i_s] = 1
        U_new.append(u); V_new.append(v); W_new.append(w)

    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    # T = sum U_r V_r^T W_r^T ?
    # C_ik = sum_j A_ij B_jk
    # Decomposition: sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    
    # Quick check with random A, B
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_calc = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pass # This is slow.
                
    # Vectorized check
    # U shape (nm, R), V shape (mp, R), W shape (pn, R)
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_reshaped = U_final.reshape(n, m, -1)
    V_reshaped = V_final.reshape(m, p, -1)
    W_reshaped = W_final.reshape(p, n, -1)
    
    # C_ik = sum_j sum_r U_{ijr} V_{jkr} W_{ikr}
    # This is a tensor contraction.
    # C = np.einsum('ijr,jkr,ikr->ik', U_reshaped, V_reshaped, W_reshaped)
    # But we have A and B.
    # The decomposition is for the TENSOR of multiplication.
    # So if we plug in A and B, we should get C.
    # A_ij B_jk * (sum_r U_... V_... W_...) ? No.
    # The tensor T is defined such that T_{ij,jk,ik} = 1.
    # So sum_r U_{ij,r} V_{jk,r} W_{ik,r} = delta_{j,j'} delta_{k,k'} ?
    # No, the tensor entry is 1 if the indices match the multiplication pattern.
    # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i).
    # So sum_r U_{a,r} V_{b,r} W_{c,r} = 1 if b's j matches a's j AND c's k matches b's k AND c's i matches a's i.
    
    # Let's check:
    # sum_r U_{ij,r} V_{j'k',r} W_{k''i'',r}
    # This should be 1 if i=i'', j=j', k=k', else 0.
    
    # My construction:
    # For a trivial term (i,j,k), U_{ij,r}=1, V_{jk,r}=1, W_{ki,r}=1.
    # Sum is 1 if indices match.
    
    # For Strassen terms, we defined U, V, W such that the sum reproduces the multiplication.
    
    # Let's verify one block manually.
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1}.
    # We added 7 terms.
    # The sum of these 7 terms should equal the sum of the 8 trivial terms for these indices.
    
    # If the verification fails, return trivial.
    
    # Efficient verification
    # Compute T_calc = sum_r U_r V_r^T (flattened) ... no.
    # We want to check if the tensor decomposes correctly.
    # We can check if for random A, B, the result matches.
    # But the decomposition is for the TENSOR, not the specific A, B.
    # The tensor T satisfies T_{a,b,c} = 1 if valid.
    # So we can check if sum_r U_{a,r} V_{b,r} W_{c,r} == T_{a,b,c} for all a,b,c.
    # This is 12*20*15 = 3600 checks.
    # R is ~56.
    # 3600 * 56 operations. Very fast.
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
                
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(U_final.shape[1]):
        T_calc += np.einsum('a,b,c->abc', U_final[:,r], V_final[:,r], W_final[:,r])
        
    if not np.allclose(T_calc, T_expected):
        # Fallback to trivial
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
