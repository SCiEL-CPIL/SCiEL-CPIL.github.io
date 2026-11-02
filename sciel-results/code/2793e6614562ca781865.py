def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    terms = [] # List of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
    def decompose_2x2x2(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
        # Global offsets for the 2x2, 2x2, 2x2 block
        # Local i,j,k from 0 to 1
        # Strassen coefficients
        # m1: (a+d)(e+h)
        # ...
        # We need to map A[i,j] to U, B[j,k] to V, C[i,k] to W.
        # A is 2x2. Local indices (li, lj). Global U index: (u_off_i + li)*5 + (u_off_j + lj) -- Wait, m is 5 globally.
        # Actually, the caller passes the *global* offsets for the block start in terms of the original matrix indices?
        # No, the caller passes offsets relative to the global array, but the block might be a sub-range.
        # It is better to pass functions to get global indices.
        
        # Let's simplify: The function assumes the block is at global positions:
        # U rows: [u_off_i .. u_off_i+1] x [u_off_j .. u_off_j+1]
        # V rows: [v_off_j .. v_off_j+1] x [v_off_k .. v_off_k+1]
        # W rows: [w_off_k .. w_off_k+1] x [w_off_i .. w_off_i+1]
        
        # Strassen terms for 2x2
        # Inputs A: a00,a01,a10,a11
        # Inputs B: b00,b01,b10,b11
        # Outputs C: c00,c01,c10,c11
        
        # We will add 7 terms.
        # Each term corresponds to a product X * Y.
        # X is a linear combo of A entries. Y is linear combo of B entries.
        # We need to expand X and Y into individual basis terms for U and V?
        # No! The decomposition is $T = \sum U_{ij,r} V_{jk,r} W_{ki,r}$.
        # If we use Strassen, we introduce intermediate variables M1..M7.
        # This changes the structure.
        # The standard Strassen decomposition of the 2x2x2 tensor has rank 7.
        # The 7 terms are:
        # 1. (a11+a22)(b11+b22)
        # 2. (a21+a22)b11
        # 3. a11(b21-b11)
        # 4. a22(b12-b22)
        # 5. (a11+a12)b22
        # 6. (a21-a11)(b11+b12)
        # 7. (a12-a22)(b21+b22)
        # Wait, standard Strassen:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)(b11)
        # m3 = (a11)(b21-b11)
        # m4 = (a22)(b12-b22)
        # m5 = (a11+a12)(b22)
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # To convert this to the tensor decomposition form:
        # We just add 7 terms.
        # Term r has:
        # U part: vector of coefficients for A entries.
        # V part: vector of coefficients for B entries.
        # W part: vector of coefficients for C entries?
        # No, W is the output C. The decomposition is:
        # C_ij = sum_r U_i,r V_j,r W_r? No.
        # The tensor is $T_{ijk}$. $T_{ijk} = \sum_r U_{ij, r} V_{jk, r} W_{ki, r}$.
        # Note that W is indexed by $k,n$ (output cols and rows).
        # In Strassen, we compute C. So W_r should be the coefficients of C in term r.
        # BUT, in the standard tensor format, the output is determined by the contraction.
        # If we set $U_{ij, r}$ and $V_{jk, r}$, the output is automatically determined?
        # No, we need $W_{ki, r}$ to match the output index.
        # Actually, if we define the term as $u_r(a) v_r(b) w_r(c)$, we must ensure consistency.
        # For Strassen, we have $c_{ij} = \sum_{r=1}^7 (\text{coeff of } c_{ij} \text{ in } M_r) M_r$.
        # So $W_{ki, r}$ is the coefficient of $c_{i,k}$ in the r-th Strassen product.
        # And $U_{ij, r}$ is the coefficient of $a_{i,j}$ in the first factor of $M_r$.
        # And $V_{jk, r}$ is the coefficient of $b_{j,k}$ in the second factor of $M_r$.
        
        # This is perfect. We just need the Strassen factors.
        
        pass # Implemented in full below
