import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (columns 0-3 of B)
    # Rank 48 for 4x4x3 part (columns 4-6 of B)
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: 4x4x4 using Recursive Strassen ---
    # Base Strassen coefficients for 2x2x2
    # u_base[p, q, t], v_base[q, r, t], w_base[p, r, t]
    # t is 0..6
    
    u_base = np.zeros((2, 2, 7))
    v_base = np.zeros((2, 2, 7))
    w_base = np.zeros((2, 2, 7))
    
    # M1 = (A00+A11)(B00+B11)
    u_base[0,0,0] = 1; u_base[1,1,0] = 1
    v_base[0,0,0] = 1; v_base[1,1,0] = 1
    # C00 += M1, C11 += M1
    w_base[0,0,0] = 1; w_base[1,1,0] = 1
    
    # M2 = (A10+A11)B00
    u_base[1,0,1] = 1; u_base[1,1,1] = 1
    v_base[0,0,1] = 1
    # C10 += M2, C11 -= M2
    w_base[1,0,1] = 1; w_base[1,1,1] = -1
    
    # M3 = A00(B01-B11)
    u_base[0,0,2] = 1
    v_base[0,1,2] = 1; v_base[1,1,2] = -1
    # C01 += M3, C11 += M3
    w_base[0,1,2] = 1; w_base[1,1,2] = 1
    
    # M4 = A11(B10-B00)
    u_base[1,1,3] = 1
    v_base[1,0,3] = 1; v_base[0,0,3] = -1
    # C00 += M4, C10 += M4
    w_base[0,0,3] = 1; w_base[1,0,3] = 1
    
    # M5 = (A00+A01)B11
    u_base[0,0,4] = 1; u_base[0,1,4] = 1
    v_base[1,1,4] = 1
    # C00 -= M5, C01 += M5
    w_base[0,0,4] = -1; w_base[0,1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    u_base[1,0,5] = 1; u_base[0,0,5] = -1
    v_base[0,0,5] = 1; v_base[0,1,5] = 1
    # C11 += M6
    w_base[1,1,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    u_base[0,1,6] = 1; u_base[1,1,6] = -1
    v_base[1,0,6] = 1; v_base[1,1,6] = 1
    # C00 += M7
    w_base[0,0,6] = 1

    # Top-level Strassen for 4x4x4
    # Blocks A00, A01, A10, A11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks in M_s
    # a_s[I, J, p, q] where I,J are block indices (0,1) and p,q are local indices (0,1)
    # Actually, we can just define the linear combinations of A entries directly.
    
    # We will build U, V, W for the 49 terms.
    # Term index r = s * 7 + t
    
    for s in range(7):
        # Determine linear combination of A blocks for M_s
        # A_s is a 2x2 matrix of coefficients for A's 4x4 entries?
        # No, A_s is a 2x2 matrix where each entry is a linear combo of A's 2x2 blocks.
        
        # Let's define the coefficients for A_s[p, q] (p,q in 0..1)
        # A_s[p, q] = sum_{I,J} coeff_A[s, p, q, I, J] * A_block[I, J]
        # A_block[I, J] is 2x2.
        
        coeff_A = np.zeros((2, 2, 2, 2)) # [p, q, I, J]
        coeff_B = np.zeros((2, 2, 2, 2)) # [q, r, I, J]
        
        if s == 0: # M1
            coeff_A[0,0,0,0] = 1; coeff_A[1,1,1,1] = 1
            coeff_B[0,0,0,0] = 1; coeff_B[1,1,1,1] = 1
        elif s == 1: # M2
            coeff_A[1,0,1,0] = 1; coeff_A[1,1,1,1] = 1
            coeff_B[0,0,0,0] = 1
        elif s == 2: # M3
            coeff_A[0,0,0,0] = 1
            coeff_B[0,1,0,1] = 1; coeff_B[1,1,1,1] = -1
        elif s == 3: # M4
            coeff_A[1,1,1,1] = 1
            coeff_B[1,0,1,0] = 1; coeff_B[0,0,0,0] = -1
        elif s == 4: # M5
            coeff_A[0,0,0,0] = 1; coeff_A[0,1,0,1] = 1
            coeff_B[1,1,1,1] = 1
        elif s == 5: # M6
            coeff_A[1,0,1,0] = 1; coeff_A[0,0,0,0] = -1
            coeff_B[0,0,0,0] = 1; coeff_B[0,1,0,1] = 1
        elif s == 6: # M7
            coeff_A[0,1,0,1] = 1; coeff_A[1,1,1,1] = -1
            coeff_B[1,0,1,0] = 1; coeff_B[1,1,1,1] = 1

        for t in range(7):
            r = s * 7 + t
            # Compute U, V, W for this term
            # U[i, j, r]
            # i = 2*I + p, j = 2*J + q
            # U[i, j, r] = sum_{p', q'} u_base[p', q', t] * coeff_A[p', q', I, J]
            
            for I in range(2):
                for J in range(2):
                    for p in range(2):
                        for q in range(2):
                            i = 2*I + p
                            j = 2*J + q
                            val = 0
                            for p_prime in range(2):
                                for q_prime in range(2):
                                    val += u_base[p_prime, q_prime, t] * coeff_A[p_prime, q_prime, I, J]
                            U[i*m + j, r] = val
                            
            # V[j, k, r]
            # j = 2*J + q, k = 2*K + r_local
            # V[j, k, r] = sum_{q', r'} v_base[q', r', t] * coeff_B[q', r', J, K]
            # Note: coeff_B is indexed by [q, r, I, J] in my definition above?
            # My coeff_B definition: coeff_B[q, r, I, J] where I,J are block indices of B.
            # B block indices are I (row), J (col).
            # So for B, row block is I, col block is J.
            # In M_s, the "row" of B is determined by the "col" of A.
            # A is 4x4, B is 4x4.
            # A block [I, J] multiplies B block [J, K].
            # So for a fixed s, the B blocks involved are determined by the A blocks?
            # No, Strassen's M_s uses specific combinations.
            # M1 = (A00+A11)(B00+B11).
            # A00 is block [0,0], A11 is [1,1].
            # B00 is block [0,0], B11 is [1,1].
            # So the B blocks used are [0,0] and [1,1].
            # In my coeff_B definition, I used I,J as block indices.
            # So for M1, coeff_B[0,0,0,0]=1 and coeff_B[1,1,1,1]=1.
            # This means B block [0,0] and [1,1] are used.
            
            for J in range(2): # Row block of B (matches Col block of A)
                for K in range(2): # Col block of B
                    for q in range(2):
                        for r_local in range(2):
                            j = 2*J + q
                            k = 2*K + r_local
                            val = 0
                            for q_prime in range(2):
                                for r_prime in range(2):
                                    val += v_base[q_prime, r_prime, t] * coeff_B[q_prime, r_prime, J, K]
                            V[j*p + k, r] = val
                            
            # W[k, i, r]
            # k = 2*K + r_local, i = 2*I + p
            # W[k, i, r] = w_top[I, K, s] * w_base[p, r_local, t]
            # w_top[I, K, s] is the coefficient of M_s in C_block[I, K]
            
            w_top = np.zeros((2, 2))
            if s == 0: # M1
                w_top[0,0] = 1; w_top[1,1] = 1
            elif s == 1: # M2
                w_top[1,0] = 1; w_top[1,1] = -1
            elif s == 2: # M3
                w_top[0,1] = 1; w_top[1,1] = 1
            elif s == 3: # M4
                w_top[0,0] = 1; w_top[1,0] = 1
            elif s == 4: # M5
                w_top[0,0] = -1; w_top[0,1] = 1
            elif s == 5: # M6
                w_top[1,1] = 1
            elif s == 6: # M7
                w_top[0,0] = 1

            for I in range(2):
                for K in range(2):
                    for p in range(2):
                        for r_local in range(2):
                            i = 2*I + p
                            k = 2*K + r_local
                            W[k*n + i, r] = w_top[I, K] * w_base[p, r_local, t]

    # --- Part 2: 4x4x3 using Trivial Decomposition ---
    start_rank = rank_444
    for i in range(n):
        for j in range(m):
            for k in range(4, 7): # k in 4,5,6
                r = start_rank + (i*m + j)*3 + (k-4)
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
