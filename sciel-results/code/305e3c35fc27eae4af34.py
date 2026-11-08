"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition:
    # 1. 16 mults for A[:,0] * B[0,:] (Outer product)
    # 2. 4 blocks of A[:,1:] * B[1:, 2q:2q+2] using Strassen (7 mults each)
    # Total: 16 + 4*7 = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 B_1 (A col 0, B row 0)
    # A indices: i in [0,1], j=0 -> row i*m + 0
    # B indices: j=0, k in [0,7] -> row j*p + k
    # C indices: i in [0,1], k in [0,7] -> W row k*n + i
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_2 B_2 (A cols 1,2; B rows 1,2)
    # A_2 is 2x2, B_2 is 2x8.
    # Split B_2 into 4 blocks of 2x2.
    # Block q corresponds to B rows 1,2 and cols 2q, 2q+1.
    
    # Strassen 7 products for 2x2x2:
    # Inputs P (2x2), Q (2x2)
    # m1 = (p00+p11)(q00+q11)
    # m2 = (p10+p11)q00
    # m3 = p00(q01-q11)
    # m4 = p11(q10-q00)
    # m5 = (p00+p01)q11
    # m6 = (p10-p00)(q00+q01)
    # m7 = (p01-p11)(q10+q11)
    
    # Outputs C (2x2)
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m4 - m6 + m7
    # c10 = m2 - m3 + m6 - m7
    # c11 = m1 - m2 + m4 + m7
    
    # We need to add coefficients to U, V, W for each term.
    # For each Strassen product mk:
    # U entries are sums of P elements.
    # V entries are sums of Q elements.
    # W entries are single C element with coefficient from output formula.
    
    for q in range(4):
        k_start = 2 * q
        
        # Map local indices 0,1 to global indices
        # P (A_2) rows: A[0, 1..2], A[1, 1..2]
        # p00 -> A[0, 1+m? No, j=1] -> U[0*m+1]
        # p01 -> A[0, 2] -> U[0*m+2]
        # p10 -> A[1, 1] -> U[1*m+1]
        # p11 -> A[1, 2] -> U[1*m+2]
        
        # Q (B_2 block) rows: B[1, k_start..k_start+1], B[2, k_start..k_start+1]
        # q00 -> B[1, k_start] -> V[1*p+k_start]
        # q01 -> B[1, k_start+1] -> V[1*p+k_start+1]
        # q10 -> B[2, k_start] -> V[2*p+k_start]
        # q11 -> B[2, k_start+1] -> V[2*p+k_start+1]
        
        # C outputs:
        # c00 -> C[0, k_start] -> W[k_start*n + 0]
        # c01 -> C[0, k_start+1] -> W[(k_start+1)*n + 0]
        # c10 -> C[1, k_start] -> W[k_start*n + 1]
        # c11 -> C[1, k_start+1] -> W[(k_start+1)*n + 1]
        
        u_idx = {
            0: 0 * m + 1, # p00
            1: 0 * m + 2, # p01
            2: 1 * m + 1, # p10
            3: 1 * m + 2  # p11
        }
        v_idx = {
            0: 1 * p + k_start,     # q00
            1: 1 * p + k_start + 1, # q01
            2: 2 * p + k_start,     # q10
            3: 2 * p + k_start + 1  # q11
        }
        w_idx = {
            0: k_start * n + 0,       # c00
            1: (k_start + 1) * n + 0, # c01
            2: k_start * n + 1,       # c10
            3: (k_start + 1) * n + 1  # c11
        }
        
        # Define the 7 Strassen terms
        # Each term is (U_combinations, V_combinations, W_combinations)
        # U_combinations: list of (local_u_index, coeff)
        # V_combinations: list of (local_v_index, coeff)
        # W_combinations: list of (local_w_index, coeff)
        
        strassen_terms = [
            # m1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, -1), (2, -1), (3, 1)]),
            # m2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # m3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (2, -1)]),
            # m4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (1, 1), (3, 1)]),
            # m5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, 1)]),
            # m6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, -1), (2, 1)]),
            # m7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1), (1, 1), (2, -1)])
        ]
        
        for u_comb, v_comb, w_comb in strassen_terms:
            for u_l, coeff in u_comb:
                U[u_idx[u_l]][col] += coeff
            for v_l, coeff in v_comb:
                V[v_idx[v_l]][col] += coeff
            for w_l, coeff in w_comb:
                W[w_idx[w_l]][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
