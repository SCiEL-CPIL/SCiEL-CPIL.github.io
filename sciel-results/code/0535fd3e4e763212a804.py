import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 40 decomposition.
    # Split p=10 into 5 blocks of size 2.
    # For each block, use Strassen for the 2x2x2 part and 1 rank for the 2x1x2 part.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices in A (2x2): (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Indices in B (2x2): (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    
    # We map these to the global indices.
    # Global A index: i * m + j. Here m=3.
    # Block A uses j=0,1. So indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
    # Global B index: j * p + k. Here p=10.
    # Block B uses j=0,1. k depends on block.
    # Global W index: k * n + i. Here n=2.
    
    # Strassen factors for U (A side)
    # u1: a+d -> [1, 0, 1, 0] at indices 0,1,3,4
    # u2: c+d -> [0, 0, 1, 1]
    # u3: a   -> [1, 0, 0, 0]
    # u4: d   -> [0, 0, 0, 1]
    # u5: a+b -> [1, 1, 0, 0]
    # u6: c-a -> [-1, 0, 1, 0]
    # u7: b-d -> [0, 1, 0, -1]
    
    # Strassen factors for V (B side)
    # v1: e+h -> [1, 0, 0, 1] at indices 0,1,3,4 (relative to block)
    # v2: e   -> [1, 0, 0, 0]
    # v3: f-h -> [0, 1, 0, -1]
    # v4: g-e -> [-1, 0, 1, 0]
    # v5: h   -> [0, 0, 0, 1]
    # v6: e+f -> [1, 1, 0, 0]
    # v7: g+h -> [0, 0, 1, 1]
    
    # Strassen factors for W (C side)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W indices for block: (k,i). k=0,1; i=0,1.
    # w0: C11 -> [1, 0, 0, 1] ? No, W is (k*n+i).
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    # C11 is at (0,0) -> idx 0.
    # C12 is at (0,1) -> idx 1.
    # C21 is at (1,0) -> idx 2.
    # C22 is at (1,1) -> idx 3.
    
    # w1 (for M1): [1, 0, 0, 1] (contributes to C11 and C22)
    # w2 (for M2): [0, 0, 1, -1] (contributes to C21 and C22)
    # w3 (for M3): [0, 1, 0, 1] (contributes to C12 and C22)
    # w4 (for M4): [1, 0, 1, 0] (contributes to C11 and C21)
    # w5 (for M5): [-1, 1, 0, 0] (contributes to C11 and C12)
    # w6 (for M6): [0, 0, 0, 1] (contributes to C22)
    # w7 (for M7): [1, 0, 0, 0] (contributes to C11)
    
    # Let's verify W coefficients:
    # M1: C11(+1), C22(+1) -> [1, 0, 0, 1]
    # M2: C21(+1), C22(-1) -> [0, 0, 1, -1]
    # M3: C12(+1), C22(+1) -> [0, 1, 0, 1]
    # M4: C11(+1), C21(+1) -> [1, 0, 1, 0]
    # M5: C11(-1), C12(+1) -> [-1, 1, 0, 0]
    # M6: C22(+1)           -> [0, 0, 0, 1]
    # M7: C11(+1)           -> [1, 0, 0, 0]
    
    # Check C11: M1+M4-M5+M7 -> 1+1-(-1)+1 = 4? No.
    # C11 = M1 + M4 - M5 + M7.
    # Coeffs: M1:1, M4:1, M5:-1, M7:1.
    # My w5 was for M5. If w5 is [-1, 1, 0, 0], then M5 contributes -1 to C11. Correct.
    # My w7 is [1, 0, 0, 0]. M7 contributes 1 to C11. Correct.
    # Sum for C11: 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2?
    # Wait. Strassen formula:
    # C11 = M1 + M4 - M5 + M7.
    # If I define the factors such that the product is M_r, then the W factor should be the coefficient.
    # So if W_r is the vector of coefficients for C, then sum_r V_r * U_r * W_r.
    # So W_r should be exactly the coefficient vector.
    # W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, -1]
    # W3 = [0, 1, 0, 1]
    # W4 = [1, 0, 1, 0]
    # W5 = [-1, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    # W7 = [1, 0, 0, 0]
    
    # Let's re-verify C11:
    # 1*W1 + 1*W4 + (-1)*W5 + 1*W7
    # = [1,0,0,1] + [1,0,1,0] - [-1,1,0,0] + [1,0,0,0]
    # = [1,0,0,1] + [1,0,1,0] + [1,-1,0,0] + [1,0,0,0]
    # = [4, -1, 1, 1]. This is not [1,0,0,0].
    
    # Ah, the Strassen formula is:
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # So the coefficient of M1 in C11 is 1.
    # The coefficient of M4 in C11 is 1.
    # The coefficient of M5 in C11 is -1.
    # The coefficient of M7 in C11 is 1.
    # So W1 should have 1 at C11. W4 should have 1 at C11. W5 should have -1 at C11. W7 should have 1 at C11.
    # My W vectors are correct for the *contribution* of that M to the C matrix.
    # But I need to sum them up to get the identity? No, I need the sum to be the correct C.
    # The decomposition is $\sum_r U_r V_r W_r$.
    # This means for a specific output element $C_{ik}$, the value is $\sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
    # So $W_{ki,r}$ is just a scalar.
    # My W vectors above represent the entire row of W for a given r.
    # So for r=1, W[0,1]=1, W[3,1]=1.
    # This means M1 contributes to C11 and C22.
    # This is correct.
    
    # Now, let's assemble the full U, V, W for rank 40.
    
    # U rows: 6 (2x3). Indices 0..5.
    # V rows: 30 (3x10). Indices 0..29.
    # W rows: 20 (10x2). Indices 0..19.
    
    # Block t=0..4.
    # k_offset = t * 2.
    
    for t in range(5):
        k_off = t * 2
        # Strassen part
        # U factors (7 columns)
        # Indices in U: (i,j) -> i*3+j.
        # a: (0,0)=0, b: (0,1)=1, c: (1,0)=3, d: (1,1)=4.
        
        U_vals = [
            [1, 0, 1, 0], # M1: a+d
            [0, 0, 1, 1], # M2: c+d
            [1, 0, 0, 0], # M3: a
            [0, 0, 0, 1], # M4: d
            [1, 1, 0, 0], # M5: a+b
            [-1, 0, 1, 0],# M6: c-a
            [0, 1, 0, -1] # M7: b-d
        ]
        
        # V factors (7 columns)
        # Indices in V: (j,k) -> j*10+k.
        # e: (0, k_off), f: (0, k_off+1), g: (1, k_off), h: (1, k_off+1).
        # Indices: 10*k_off, 10*k_off+1, 10*k_off+10, 10*k_off+11.
        
        V_vals = [
            [1, 0, 0, 1], # M1: e+h
            [1, 0, 0, 0], # M2: e
            [0, 1, 0, -1],# M3: f-h
            [-1, 0, 1, 0],# M4: g-e
            [0, 0, 0, 1], # M5: h
            [1, 1, 0, 0], # M6: e+f
            [0, 0, 1, 1]  # M7: g+h
        ]
        
        # W factors (7 columns)
        # Indices in W: (k,i) -> k*2+i.
        # C11: (k_off, 0) -> 2*k_off
        # C12: (k_off, 1) -> 2*k_off+1
        # C21: (k_off+1, 0) -> 2*k_off+2
        # C22: (k_off+1, 1) -> 2*k_off+3
        
        W_vals = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        base_col = t * 8
        
        for r in range(7):
            col = base_col + r
            # U
            for i in range(2):
                for j in range(2):
                    idx = i * 3 + j
                    U[idx][col] = U_vals[r][i*2+j] # Wait, U_vals is 2x2?
                    # My U_vals list above is 7 vectors of length 4.
                    # The 4 values correspond to a,b,c,d.
                    # a is (0,0), b is (0,1), c is (1,0), d is (1,1).
                    # So index in vector is i*2+j?
                    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
                    # Yes.
            # V
            for j in range(2):
                for kk in range(2):
                    idx = j * 10 + (k_off + kk)
                    V[idx][col] = V_vals[r][j*2+kk]
            # W
            for kk in range(2):
                for i in range(2):
                    idx = (k_off + kk) * 2 + i
                    W[idx][col] = W_vals[r][kk*2+i]
                    
        # Outer product part (1 column)
        col = base_col + 7
        # A part: A_{i,2}. Indices: (0,2)=2, (1,2)=5.
        U[2][col] = 1
        U[5][col] = 1
        
        # B part: B_{2,k}. Indices: (2, k_off), (2, k_off+1).
        V[2 * 10 + k_off][col] = 1
        V[2 * 10 + k_off + 1][col] = 1
        
        # W part: C_{i,k}.
        # C_{0,k} += A_{0,2} B_{2,k}
        # C_{1,k} += A_{1,2} B_{2,k}
        # So W should be 1 for all (k,i) in this block.
        for kk in range(2):
            for i in range(2):
                idx = (k_off + kk) * 2 + i
                W[idx][col] = 1

    return {"U": U, "V": V, "W": W}
