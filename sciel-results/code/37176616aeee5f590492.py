import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor is M_{2,2,2} \otimes M_{1,1,6}
    # Rank of M_{2,2,2} is 7 (Strassen)
    # Rank of M_{1,1,6} is 6 (Identity)
    # Total rank = 42
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[i, j], [k, l]]
    
    # U (for A)
    # S0: a+d
    # S1: c+d
    # S2: a
    # S3: d
    # S4: a+b
    # S5: c-a
    # S6: b-d
    
    U_strassen = np.array([
        [1, 0, 0, 1],  # S0: a+d
        [0, 0, 1, 1],  # S1: c+d
        [1, 0, 0, 0],  # S2: a
        [0, 0, 0, 1],  # S3: d
        [1, 1, 0, 0],  # S4: a+b
        [-1, 0, 1, 0], # S5: c-a
        [0, 1, 0, -1]  # S6: b-d
    ])
    
    # V (for B)
    # T0: e+h -> V0: e, V1: h
    # T1: e -> V0: e
    # T2: f-h -> V0: f, V1: -h
    # T3: g-e -> V0: -e, V1: g
    # T4: h -> V1: h
    # T5: e+f -> V0: e, V0: f (wait, indices)
    # Standard Strassen:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)(e)
    # m2 = (a)(f-h)
    # m3 = (d)(g-e)
    # m4 = (a+b)(h)
    # m5 = (c-a)(e+f)
    # m6 = (b-d)(g+h)
    
    # Let's map these to V forms.
    # B indices: 0:e, 1:f, 2:g, 3:h
    V_strassen = np.array([
        [1, 0, 0, 1],  # T0: e+h
        [1, 0, 0, 0],  # T1: e
        [0, 1, 0, -1], # T2: f-h
        [-1, 0, 1, 0], # T3: g-e
        [0, 0, 0, 1],  # T4: h
        [1, 1, 0, 0],  # T5: e+f
        [0, 0, 1, 1]   # T6: g+h
    ])
    
    # W (for C)
    # C indices: 0:i, 1:j, 2:k, 3:l
    # i = m0 + m3 - m4 + m6
    # j = m2 + m4
    # k = m1 + m3
    # l = m0 - m1 + m2 + m5
    
    W_strassen = np.array([
        [1, 0, 0, 0],  # m0
        [0, 0, 1, -1], # m1
        [0, 1, 0, 1],  # m2
        [0, 0, 1, 0],  # m3
        [-1, 1, 0, 0], # m4
        [0, 0, 0, 1],  # m5
        [1, 0, 0, 0]   # m6
    ])
    
    # Now construct the full U, V, W for M_{2,2,12}
    # U_full: (n*m, rank) = (4, 42)
    # V_full: (m*p, rank) = (24, 42)
    # W_full: (p*n, rank) = (24, 42)
    
    rank = 42
    U = np.zeros((4, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # The tensor is T = sum_{k=0}^5 sum_{t=0}^6 u_t \otimes v_{t,k} \otimes w_{t,k}
    # where u_t is the t-th row of U_strassen
    # v_{t,k} is the t-th row of V_strassen, placed in the k-th block of B
    # w_{t,k} is the t-th row of W_strassen, placed in the k-th block of C
    
    # B is 2x12. We split it into 6 blocks of 2x2.
    # Block k corresponds to columns 2k, 2k+1 of B.
    # B indices: 0..23.
    # Block k: rows 0,1; cols 2k, 2k+1.
    # In the V matrix (size 24), the indices for block k are:
    # B_{0, 2k} is index 2k
    # B_{0, 2k+1} is index 2k+1
    # B_{1, 2k} is index 12 + 2k
    # B_{1, 2k+1} is index 12 + 2k + 1
    
    # C is 2x12. We split it into 6 blocks of 2x2.
    # Block k corresponds to columns 2k, 2k+1 of C.
    # C indices: 0..23.
    # C_{0, 2k} is index 2k
    # C_{0, 2k+1} is index 2k+1
    # C_{1, 2k} is index 12 + 2k
    # C_{1, 2k+1} is index 12 + 2k + 1
    
    for k in range(6):
        for t in range(7):
            col = k * 7 + t
            
            # Fill U
            U[:, col] = U_strassen[t]
            
            # Fill V
            # V_strassen[t] has 4 entries: e, f, g, h
            # Map to B indices for block k
            v_vec = V_strassen[t]
            V[2k][col] = v_vec[0]       # e
            V[2k+1][col] = v_vec[1]     # f
            V[12+2k][col] = v_vec[2]    # g
            V[12+2k+1][col] = v_vec[3]  # h
            
            # Fill W
            # W_strassen[t] has 4 entries: i, j, k, l
            # Map to C indices for block k
            w_vec = W_strassen[t]
            W[2k][col] = w_vec[0]       # i
            W[2k+1][col] = w_vec[1]     # j
            W[12+2k][col] = w_vec[2]    # k
            W[12+2k+1][col] = w_vec[3]  # l
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
