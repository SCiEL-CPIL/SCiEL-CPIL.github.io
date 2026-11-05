"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-34 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Uses Strassen (rank 7) for two 2x2x2 subproblems and schoolbook for the rest.
    """
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # (9, 34)
    V = [[0] * rank for _ in range(m * p)]  # (12, 34)
    W = [[0] * rank for _ in range(p * n)]  # (12, 34)
    
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals: dict (i,j)->val, v_vals: dict (j,k)->val, w_vals: dict (k,i)->val"""
        nonlocal col
        for (i, j), val in u_vals.items():
            U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            V[j * p + k][col] += val
        for (k, i), val in w_vals.items():
            W[k * n + i][col] += val
        col += 1
    
    # Strassen for 2x2x2: [[a,b],[c,d]] * [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(e+f), M7 = (b-d)(g+h)
    # p11 = M1 - M4 + M5 - M7
    # p12 = M3 + M4
    # p21 = M2 + M7
    # p22 = M1 - M2 + M3 + M6
    
    def strassen_2x2(a_pos, b_pos, c_pos, d_pos, e_pos, f_pos, g_pos, h_pos, 
                     p11_pos, p12_pos, p21_pos, p22_pos):
        """
        a_pos: (i,j) for A[0,0], b_pos: (i,j) for A[0,1], etc.
        e_pos: (j,k) for B[0,0], f_pos: (j,k) for B[0,1], etc.
        p11_pos: (k,i) for C[0,0], etc.
        """
        nonlocal col
        
        # M1 = (a+d)(e+h)
        u1 = {a_pos: 1, d_pos: 1}
        v1 = {e_pos: 1, h_pos: 1}
        w1 = {p11_pos: 1, p12_pos: 1, p21_pos: 1, p22_pos: 1}
        # M1 contributes +1 to p11, +1 to p12, +1 to p21, +1 to p22? No!
        # M1 is a single product, it contributes to multiple outputs with specific coefficients.
        
        # Actually, each M is a single rank-1 term. The W matrix entries determine
        # how much each M contributes to each output.
        
        # Let me redefine: for each M, I specify U, V, and W coefficients.
        
        # M1 = (a+d)(e+h): U has 1 at a and d, V has 1 at e and h
        # This M1 contributes to p11 with coeff +1, p12 with coeff +1, p21 with coeff +1, p22 with coeff +1
        # Wait no. M1 is ONE product. In the tensor decomposition, each rank-1 term has
        # specific U, V, W vectors. The contribution to output (k,i) is U[i,j]*V[j,k]*W[k,i]
        # summed over j. But for a rank-1 term, it's just one (i,j,k) triple... no.
        
        # Actually, in the tensor decomposition, a rank-1 term is:
        # U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # where U[:,t] is a vector of length n*m, V[:,t] of length m*p, W[:,t] of length p*n.
        
        # The contribution to C[i,k] is sum_j U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # So for a rank-1 term, I specify which (i,j), (j,k), (k,i) entries are non-zero.
        
        # For Strassen M1 = (a+d)(e+h):
        # U: 1 at (0,0) [a] and (1,1) [d]  -- assuming a=A[0,0], d=A[1,1]
        # V: 1 at (0,0) [e] and (1,1) [h]  -- assuming e=B[0,0], h=B[1,1]
        # W: needs to be set so that M1 contributes correctly to all 4 outputs.
        
        # M1 contributes: +1 to p11, +1 to p12, +1 to p21, +1 to p22? No.
        # From the formulas:
        # p11 = M1 - M4 + M5 - M7  → M1 coeff +1
        # p12 = M3 + M4            → M1 coeff 0
        # p21 = M2 + M7            → M1 coeff 0
        # p22 = M1 - M2 + M3 + M6  → M1 coeff +1
        
        # So M1 contributes to p11 and p22 only.
        # W for M1: 1 at p11_pos and p22_pos.
        
        # But wait, the U and V for M1 involve multiple (i,j) and (j,k) pairs.
        # U: (0,0)->1, (1,1)->1
        # V: (0,0)->1, (1,1)->1
        # The product U[i,j]*V[j,k] for M1 is non-zero when:
        #   (i,j)=(0,0) and (j,k)=(0,0) → i=0, k=0 → C[0,0]
        #   (i,j)=(0,0) and (j,k)=(1,1) → i=0, k=1 → C[0,1]
        #   (i,j)=(1,1) and (j,k)=(0,0) → i=1, k=0 → C[1,0]
        #   (i,j)=(1,1) and (j,k)=(1,1) → i=1, k=1 → C[1,1]
        
        # So M1's U*V product is non-zero for all 4 (i,k) pairs!
        # Therefore W must be set to give the correct coefficient for each.
        
        # M1 contribution to C[i,k] = sum over the (i,j,k) combos of U[i,j]*V[j,k]*W[k,i]
        # For M1: U has entries at (0,0) and (1,1), V has entries at (0,0) and (1,1).
        # The (i,k) pairs that get non-zero contribution: (0,0), (0,1), (1,0), (1,1).
        
        # For (0,0): comes from (i,j)=(0,0), (j,k)=(0,0): U=1, V=1, so W[0,0] * 1 * 1
        # For (0,1): comes from (i,j)=(0,0), (j,k)=(1,1): U=1, V=1, so W[1,0] * 1 * 1
        # For (1,0): comes from (i,j)=(1,1), (j,k)=(0,0): U=1, V=1, so W[0,1] * 1 * 1
        # For (1,1): comes from (i,j)=(1,1), (j,k)=(1,1): U=1, V=1, so W[1,1] * 1 * 1
        
        # We want M1 to contribute +1 to p11=C[0,0] and +1 to p22=C[1,1], and 0 to p12 and p21.
        # So W[0,0]=1, W[1,0]=0, W[0,1]=0, W[1,1]=1.
        
        # This is getting complex. Let me just carefully set up each M.
        
        # I'll define a helper that takes the U, V, W dictionaries and adds the term.
        
        # M1 = (a+d)(e+h)
        add_term(
            {a_pos: 1, d_pos: 1},
            {e_pos: 1, h_pos: 1},
            {p11_pos: 1, p22_pos: 1}
        )
        
        # M2 = (c+d)e
        add_term(
            {c_pos: 1, d_pos: 1},
            {e_pos: 1},
            {p21_pos: 1}
        )
        
        # M3 = a(f-h)
        add_term(
            {a_pos: 1},
            {f_pos: 1, h_pos: -1},
            {p12_pos: 1, p22_pos: 1}
        )
        
        # M4 = d(g-e)
        add_term(
            {d_pos: 1},
            {g_pos: 1, e_pos: -1},
            {p12_pos: 1, p11_pos: -1}
        )
        
        # M5 = (a+b)h
        add_term(
            {a_pos: 1, b_pos: 1},
            {h_pos: 1},
            {p11_pos: 1}
        )
        
        # M6 = (c-d)(e+f)
        add_term(
            {c_pos: 1, d_pos: -1},
            {e_pos: 1, f_pos: 1},
            {p22_pos: 1}
        )
        
        # M7 = (b-d)(g+h)
        add_term(
            {b_pos: 1, d_pos: -1},
            {g_pos: 1, h_pos: 1},
            {p21_pos: 1, p11_pos: -1}
        )
        
        return  # 7 terms added
    
    # Now build the rank-34 decomposition:
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    strassen_2x2(
        a_pos=(0,0), b_pos=(0,1), c_pos=(1,0), d_pos=(1,1),
        e_pos=(0,0), f_pos=(0,1), g_pos=(1,0), h_pos=(1,1),
        p11_pos=(0,0), p12_pos=(1,0), p21_pos=(0,1), p22_pos=(1,1)
    )
    
    # Block 2: A[0:2,2] * B[2,0:2] -> C[0:2,0:2] (schoolbook, rank 4)
    # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    add_term({(0,2): 1}, {(2,0): 1}, {(0,0): 1})
    add_term({(0,2): 1}, {(2,1): 1}, {(1,0): 1})
    add_term({(1,2): 1}, {(2,0): 1}, {(0,1): 1})
    add_term({(1,2): 1}, {(2,1): 1}, {(1,1): 1})
    
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    strassen_2x2(
        a_pos=(0,0), b_pos=(0,1), c_pos=(1,0), d_pos=(1,1),
        e_pos=(0,2), f_pos=(0,3), g_pos=(1,2), h_pos=(1,3),
        p11_pos=(2,0), p12_pos=(3,0), p21_pos=(2,1), p22_pos=(3,1)
    )
    
    # Block 4: A[0:2,2] * B[2,2:4] -> C[0:2,2:4] (schoolbook, rank 4)
    add_term({(0,2): 1}, {(2,2): 1}, {(2,0): 1})
    add_term({(0,2): 1}, {(2,3): 1}, {(3,0): 1})
    add_term({(1,2): 1}, {(2,2): 1}, {(2,1): 1})
    add_term({(1,2): 1}, {(2,3): 1}, {(3,1): 1})
    
    # Block 5: A[2,0:2] * B[0:2,0:2] -> C[2,0:2] (schoolbook, rank 4)
    add_term({(2,0): 1}, {(0,0): 1}, {(0,2): 1})
    add_term({(2,0): 1}, {(0,1): 1}, {(1,2): 1})
    add_term({(2,1): 1}, {(0,0): 1}, {(0,2): 1})
    add_term({(2,1): 1}, {(0,1): 1}, {(1,2): 1})
    
    # Block 6: A[2,2] * B[2,0:2] -> C[2,0:2] (schoolbook, rank 2)
    add_term({(2,2): 1}, {(2,0): 1}, {(0,2): 1})
    add_term({(2,2): 1}, {(2,1): 1}, {(1,2): 1})
    
    # Block 7: A[2,0:2] * B[0:2,2:4] -> C[2,2:4] (schoolbook, rank 4)
    add_term({(2,0): 1}, {(0,2): 1}, {(2,2): 1})
    add_term({(2,0): 1}, {(0,3): 1}, {(3,2): 1})
    add_term({(2,1): 1}, {(0,2): 1}, {(2,2): 1})
    add_term({(2,1): 1}, {(0,3): 1}, {(3,2): 1})
    
    # Block 8: A[2,2] * B[2,2:4] -> C[2,2:4] (schoolbook, rank 2)
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1})
    add_term({(2,2): 1}, {(2,3): 1}, {(3,2): 1})
    
    # Total: 7+4+7+4+4+2+4+2 = 34
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness with random test
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, size=(n, m))
    B_test = np.random.randint(-5, 6, size=(m, p))
    C_expected = A_test @ B_test
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        u_t = U_np[:, t].reshape(n, m)
        v_t = V_np[:, t].reshape(m, p)
        w_t = W_np[:, t].reshape(p, n)
        # C[i,k] += sum_j u_t[i,j] * v_t[j,k] * w_t[k,i]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += np.sum(u_t * v_t * w_t[k, i])
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
