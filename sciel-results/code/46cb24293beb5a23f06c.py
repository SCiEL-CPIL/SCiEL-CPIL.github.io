import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block-Strassen approach.
    # A is 2x2. B is 2x13. C is 2x13.
    # We split B into blocks of 2 columns, plus a tail of 1 column.
    # Number of full 2-column blocks: q = p // 2 = 6
    # Tail size: r = p % 2 = 1
    
    # Each 2x2 block multiplication can be done with rank 7 (Strassen).
    # The tail 2x1 multiplication can be done with rank 3 (naive: 2*1=2? No, 2x1 * 1x1? 
    # Wait, B block is 2x1. A is 2x2. Result is 2x1.
    # C_block = A * B_block.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This requires 4 multiplications if naive?
    # M1 = A00*B0, M2 = A01*B1, M3 = A10*B0, M4 = A11*B1.
    # Can we do it with rank 3?
    # Tensor for 2x2 * 2x1.
    # Dimensions: n=2, m=2, p=1.
    # Naive rank = 4.
    # Is rank 3 possible?
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # It is known that the rank of 2x2 * 2x1 is 4. (Since it contains 2x1 * 2x1 twice? No.)
    # Let's check if rank 3 is possible.
    # If rank 3, then we have 3 bilinear forms.
    # The space of 2x2 matrices has dim 4. The space of 2x1 vectors has dim 2.
    # The tensor is in U* x V* x W*.
    # Actually, just use rank 4 for the tail to be safe, or check if 3 works.
    # For 2x2 * 2x1, the minimal rank is 4.
    # So total rank = 6 * 7 + 1 * 4 = 42 + 4 = 46.
    
    # Let's verify Strassen rank for 2x2 * 2x2 is 7.
    # Yes, standard Strassen.
    
    # Constructing the decomposition:
    # We will build U, V, W matrices.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (26, rank)
    # W has shape (p*n, rank) = (26, rank)
    
    # We will fill columns of U, V, W.
    # Let R = 46.
    R = 46
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen factors for 2x2 * 2x2 block
    # A indices in U:
    # a = A[0,0] -> U[0]
    # b = A[0,1] -> U[1]
    # c = A[1,0] -> U[2]
    # d = A[1,1] -> U[3]
    
    # B indices in V for block starting at column k:
    # e = B[0,k]   -> V[k]
    # f = B[0,k+1] -> V[k+1]
    # g = B[1,k]   -> V[m*p + k] = V[13+k]
    # h = B[1,k+1] -> V[m*p + k+1] = V[13+k+1]
    
    # C indices in W for block starting at column k:
    # p = C[0,k]   -> W[k*n + 0] = W[2k]
    # q = C[0,k+1] -> W[(k+1)*n + 0] = W[2k+2]
    # r = C[1,k]   -> W[k*n + 1] = W[2k+1]
    # s = C[1,k+1] -> W[(k+1)*n + 1] = W[2k+3]
    
    # Strassen Multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-e)
    # M4 = (d)(h-f)
    # M5 = (a+c)(h)
    # M6 = (b-d)(g+h)
    # M7 = (b-c)(e+g)
    
    # Reconstruction:
    # p = M1 + M4 - M5 + M7
    # q = M3 + M1 + M5 - M6
    # r = M2 + M4 - M1 + M6
    # s = M3 + M2 + M7
    
    # We define the coefficients for U, V, W for each Strassen term.
    # Term 1: M1
    # U: a=1, d=1 -> U[0]=1, U[3]=1
    # V: e=1, h=1 -> V[k]=1, V[13+k+1]=1
    # W: p=1, q=1, r=-1, s=0 -> W[2k]=1, W[2k+2]=1, W[2k+1]=-1, W[2k+3]=0
    
    # Term 2: M2
    # U: c=1, d=1 -> U[2]=1, U[3]=1
    # V: e=1 -> V[k]=1
    # W: p=0, q=0, r=1, s=1 -> W[2k+1]=1, W[2k+3]=1
    
    # Term 3: M3
    # U: a=1 -> U[0]=1
    # V: f=1, e=-1 -> V[k+1]=1, V[k]=-1
    # W: p=0, q=1, r=0, s=1 -> W[2k+2]=1, W[2k+3]=1
    
    # Term 4: M4
    # U: d=1 -> U[3]=1
    # V: h=1, f=-1 -> V[13+k+1]=1, V[k+1]=-1
    # W: p=1, q=0, r=1, s=0 -> W[2k]=1, W[2k+1]=1
    
    # Term 5: M5
    # U: a=1, c=1 -> U[0]=1, U[2]=1
    # V: h=1 -> V[13+k+1]=1
    # W: p=-1, q=1, r=0, s=0 -> W[2k]=-1, W[2k+2]=1
    
    # Term 6: M6
    # U: b=1, d=-1 -> U[1]=1, U[3]=-1
    # V: g=1, h=1 -> V[13+k]=1, V[13+k+1]=1
    # W: p=0, q=-1, r=1, s=0 -> W[2k+2]=-1, W[2k+1]=1
    
    # Term 7: M7
    # U: b=1, c=-1 -> U[1]=1, U[2]=-1
    # V: e=1, g=1 -> V[k]=1, V[13+k]=1
    # W: p=1, q=0, r=0, s=1 -> W[2k]=1, W[2k+3]=1
    
    # Define the Strassen block data
    # Each entry is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {row: val}
    # V_coeffs: dict {row: val}
    # W_coeffs: dict {row: val}
    
    strassen_terms = [
        # M1
        (
            {0: 1, 3: 1},
            {0: 1, 1: 1}, # relative to k: e at 0, h at 1 (offset +13 in V)
            {0: 1, 2: 1, 1: -1, 3: 0} # relative to 2k: p at 0, q at 2, r at 1, s at 3
        ),
        # M2
        (
            {2: 1, 3: 1},
            {0: 1, 1: 0}, # e at 0
            {0: 0, 2: 0, 1: 1, 3: 1}
        ),
        # M3
        (
            {0: 1, 3: 0},
            {0: -1, 1: 1}, # e=-1, f=1
            {0: 0, 2: 1, 1: 0, 3: 1}
        ),
        # M4
        (
            {0: 0, 3: 1},
            {0: 0, 1: -1}, # f=-1, h=1 (offset +13)
            {0: 1, 2: 0, 1: 1, 3: 0}
        ),
        # M5
        (
            {0: 1, 2: 1},
            {0: 0, 1: 1}, # h=1 (offset +13)
            {0: -1, 2: 1, 1: 0, 3: 0}
        ),
        # M6
        (
            {1: 1, 3: -1},
            {0: 0, 1: 1}, # g=1, h=1 (offset +13)
            {0: 0, 2: -1, 1: 1, 3: 0}
        ),
        # M7
        (
            {1: 1, 2: -1},
            {0: 1, 1: 1}, # e=1, g=1 (offset +13)
            {0: 1, 2: 0, 1: 0, 3: 1}
        )
    ]
    
    # Process 6 full blocks
    for block in range(6):
        k = block * 2
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # U
            for row, val in u_coeffs.items():
                if val != 0:
                    U[row][col_idx] = val
            
            # V
            # v_coeffs keys are relative to the block's B rows.
            # Key 0 corresponds to B[0, k] -> V[k]
            # Key 1 corresponds to B[1, k+1]? No.
            # Let's re-map carefully.
            # In my definitions above:
            # M1 V: e=1 (B[0,k]), h=1 (B[1,k+1])
            # My dict {0:1, 1:1} meant e and h.
            # e is B[0,k] -> V[k]
            # h is B[1,k+1] -> V[13+k+1]
            # So I need to map the logical "e,f,g,h" to actual V indices.
            
            # Let's redefine the term structure to be explicit about e,f,g,h coefficients.
            # This is safer.
            
            col_idx += 1
    
    # The previous loop was a dummy. Let's restart the filling logic properly.
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col_idx = 0
    
    # Define Strassen terms as (U_vec, (c_e, c_f, c_g, c_h), (c_p, c_q, c_r, c_s))
    # U_vec: list of 4 values [a, b, c, d]
    # V_coeffs: (e, f, g, h)
    # W_coeffs: (p, q, r, s)
    
    S = [
        ([1, 0, 0, 1], (1, 0, 0, 1), (1, 1, -1, 0)), # M1
        ([0, 0, 1, 1], (1, 0, 0, 0), (0, 0, 1, 1)),   # M2
        ([1, 0, 0, 0], (0, 1, 0, -1), (0, 1, 0, 1)),  # M3: a*(f-e) -> V: e=-1, f=1. Wait.
        # M3 = a * (f - e).
        # U: a=1.
        # V: f=1, e=-1. So (e=-1, f=1, g=0, h=0).
        # My tuple was (1, 0, 0, -1) for (e,f,g,h)? No, I wrote (0, 1, 0, -1).
        # Let's stick to (e, f, g, h).
        # M3: e=-1, f=1.
        # M4: d * (h - f). U: d=1. V: h=1, f=-1. (e=0, f=-1, g=0, h=1).
        # M5: (a+c) * h. U: a=1, c=1. V: h=1. (e=0, f=0, g=0, h=1).
        # M6: (b-d) * (g+h). U: b=1, d=-1. V: g=1, h=1. (e=0, f=0, g=1, h=1).
        # M7: (b-c) * (e+g). U: b=1, c=-1. V: e=1, g=1. (e=1, f=0, g=1, h=0).
        
        # W coeffs (p, q, r, s):
        # M1: p=1, q=1, r=-1, s=0
        # M2: p=0, q=0, r=1, s=1
        # M3: p=0, q=1, r=0, s=1
        # M4: p=1, q=0, r=1, s=0
        # M5: p=-1, q=1, r=0, s=0
        # M6: p=0, q=-1, r=1, s=0
        # M7: p=1, q=0, r=0, s=1
    ]
    
    # Corrected Strassen Table:
    Strassen = [
        # M1
        ([1, 0, 0, 1], (1, 0, 0, 1), (1, 1, -1, 0)),
        # M2
        ([0, 0, 1, 1], (1, 0, 0, 0), (0, 0, 1, 1)),
        # M3
        ([1, 0, 0, 0], (-1, 1, 0, 0), (0, 1, 0, 1)),
        # M4
        ([0, 0, 0, 1], (0, -1, 0, 1), (1, 0, 1, 0)),
        # M5
        ([1, 0, 1, 0], (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6
        ([0, 1, 0, -1], (0, 0, 1, 1), (0, -1, 1, 0)),
        # M7
        ([0, 1, -1, 0], (1, 0, 1, 0), (1, 0, 0, 1))
    ]
    
    # Fill 6 blocks
    for block in range(6):
        k = block * 2
        for u_vec, v_vec, w_vec in Strassen:
            # U
            for i in range(4):
                if u_vec[i] != 0:
                    U[i][col_idx] = u_vec[i]
            
            # V
            # e -> V[k]
            # f -> V[k+1]
            # g -> V[13+k]
            # h -> V[13+k+1]
            ce, cf, cg, ch = v_vec
            if ce != 0: V[k][col_idx] = ce
            if cf != 0: V[k+1][col_idx] = cf
            if cg != 0: V[13+k][col_idx] = cg
            if ch != 0: V[13+k+1][col_idx] = ch
            
            # W
            # p -> W[2k]
            # q -> W[2k+2]
            # r -> W[2k+1]
            # s -> W[2k+3]
            cp, cq, cr, cs = w_vec
            if cp != 0: W[2*k][col_idx] = cp
            if cq != 0: W[2*k+2][col_idx] = cq
            if cr != 0: W[2*k+1][col_idx] = cr
            if cs != 0: W[2*k+3][col_idx] = cs
            
            col_idx += 1
            
    # Tail block: 2x2 * 2x1
    # k = 12
    # B[0, 12] -> V[12]
    # B[1, 12] -> V[13+12] = V[25]
    # C[0, 12] -> W[24]
    # C[1, 12] -> W[25]
    
    # Naive decomposition for 2x2 * 2x1:
    # M1 = A00 * B0
    # M2 = A01 * B1
    # M3 = A10 * B0
    # M4 = A11 * B1
    
    # C0 = M1 + M2
    # C1 = M3 + M4
    
    Tail = [
        # M1
        ([1, 0, 0, 0], (1, 0), (1, 0)), # U: a, V: e, W: p
        # M2
        ([0, 1, 0, 0], (0, 1), (1, 0)), # U: b, V: f, W: p
        # M3
        ([0, 0, 1, 0], (1, 0), (0, 1)), # U: c, V: e, W: r
        # M4
        ([0, 0, 0, 1], (0, 1), (0, 1))  # U: d, V: f, W: r
    ]
    
    k = 12
    for u_vec, v_vec, w_vec in Tail:
        # U
        for i in range(4):
            if u_vec[i] != 0:
                U[i][col_idx] = u_vec[i]
        
        # V
        # e -> V[12]
        # f -> V[25]
        ce, cf = v_vec
        if ce != 0: V[12][col_idx] = ce
        if cf != 0: V[25][col_idx] = cf
        
        # W
        # p -> W[24]
        # r -> W[25]
        cp, cr = w_vec
        if cp != 0: W[24][col_idx] = cp
        if cr != 0: W[25][col_idx] = cr
        
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
