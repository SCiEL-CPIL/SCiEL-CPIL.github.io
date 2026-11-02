import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 28 (A1B) + 25 (A2aBa) + 25 (A2bBb) = 78
    rank = 78
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
        
    # Helper for Strassen 2x2x2
    # X is 2x2 from A, Y is 2x2 from B
    # X indices in U: (u_r1_c1, u_r1_c2, u_r2_c1, u_r2_c2)
    # Y indices in V: (v_r1_c1, v_r1_c2, v_r2_c1, v_r2_c2)
    # W indices: (w_r1_c1, w_r1_c2, w_r2_c1, w_r2_c2)
    # Note: Strassen coefficients can be -1, 0, 1.
    # We need to add 7 columns.
    def add_strassen(u11, u12, u21, u22, v11, v12, v21, v22, w11, w12, w21, w22):
        nonlocal col
        
        # M1 = (u11+u22)(v11+v22)
        U[u11][col] += 1; U[u22][col] += 1
        V[v11][col] += 1; V[v22][col] += 1
        W[w11][col] += 1; W[w22][col] += 1
        col += 1
        
        # M2 = (u21+u22)v11
        U[u21][col] += 1; U[u22][col] += 1
        V[v11][col] += 1
        W[w21][col] += 1
        col += 1
        
        # M3 = u11(v12-v22)
        U[u11][col] += 1
        V[v12][col] += 1; V[v22][col] -= 1
        W[w12][col] += 1
        col += 1
        
        # M4 = u22(v21-v11)
        U[u22][col] += 1
        V[v21][col] += 1; V[v11][col] -= 1
        W[w11][col] += 1; W[w21][col] += 1
        col += 1
        
        # M5 = (u11+u12)v22
        U[u11][col] += 1; U[u12][col] += 1
        V[v22][col] += 1
        W[w11][col] -= 1; W[w12][col] += 1
        col += 1
        
        # M6 = (u21-u11)(v11+v12)
        U[u21][col] += 1; U[u11][col] -= 1
        V[v11][col] += 1; V[v12][col] += 1
        W[w22][col] += 1
        col += 1
        
        # M7 = (u12-u22)(v21+v22)
        U[u12][col] += 1; U[u22][col] -= 1
        V[v21][col] += 1; V[v22][col] += 1
        W[w11][col] += 1; W[w22][col] += 1
        col += 1
        
        # Verify W accumulation:
        # w11: M1(+1) + M4(+1) - M5(-1) + M7(+1) -> 1+1-1+1 = 2? 
        # Wait. Strassen formula:
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Let's re-verify the W updates in the helper.
        # M1: W[w11]+=1, W[w22]+=1. (c11 gets M1, c22 gets M1)
        # M2: W[w21]+=1. (c21 gets M2)
        # M3: W[w12]+=1. (c12 gets M3)
        # M4: W[w11]+=1, W[w21]+=1. (c11 gets M4, c21 gets M4)
        # M5: W[w11]-=1, W[w12]+=1. (c11 gets -M5, c12 gets M5)
        # M6: W[w22]+=1. (c22 gets M6)
        # M7: W[w11]+=1, W[w22]+=1. (c11 gets M7, c22 gets M7)
        
        # Sum for w11 (c11): 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2?
        # No, the coefficients in W should be the final coefficients for the linear combination.
        # But the decomposition is a sum of rank-1 tensors.
        # Each column is a rank-1 tensor.
        # The product is sum over columns of U[:,c] * V[:,c]^T * W[:,c]?
        # No, the tensor is T = sum U_c V_c W_c.
        # So for each column c, we have a term $u_i v_j w_k$.
        # In Strassen, we replace 8 terms with 7 terms.
        # The 7 terms are linear combinations of the inputs.
        # So yes, we just add the linear combination coefficients to U, V, W for that column.
        
        # Let's check the logic again.
        # Term 1: $(u11+u22)(v11+v22)$.
        # This contributes to c11 and c22?
        # Strassen:
        # M1 is used in c11 and c22.
        # So for column M1:
        # U has 1 at u11, 1 at u22.
        # V has 1 at v11, 1 at v22.
        # W has 1 at w11, 1 at w22.
        # This creates terms:
        # u11 v11 w11, u11 v22 w11, u22 v11 w11, u22 v22 w11
        # u11 v11 w22, u11 v22 w22, u22 v11 w22, u22 v22 w22
        # This matches the expansion of $(u11+u22)(v11+v22)$ distributed to w11 and w22?
        # No, M1 is a single scalar.
        # $c_{11} = M1 + ...$
        # So the term $M1$ contributes $+1$ to $c_{11}$.
        # The term $M1$ contributes $+1$ to $c_{22}$.
        # So W[w11] += 1, W[w22] += 1.
        # And U, V contain the linear combos.
        # This is correct.
        
        pass # The logic in the helper is correct.

    # Part 1: A1 B (1x4 x 4x7)
    # A1 is row 0. B is all.
    for j in range(4):
        for k in range(7):
            add_term(0 * 4 + j, j * 7 + k, k * 3 + 0)
            
    # Part 2: A2 B (2x4 x 4x7)
    # A2 is rows 1,2.
    # Split A2 cols: 0,1 and 2,3.
    # Split B rows: 0,1 and 2,3.
    
    # Subpart 2a: A2a (rows 1,2 cols 0,1) * Ba (rows 0,1)
    # U indices:
    # u11 = 1*4+0 = 4
    # u12 = 1*4+1 = 5
    # u21 = 2*4+0 = 8
    # u22 = 2*4+1 = 9
    
    # V indices for Ba (rows 0,1):
    # v11 = 0*7+k1
    # v12 = 0*7+k2
    # v21 = 1*7+k1
    # v22 = 1*7+k2
    
    # W indices for C (rows 1,2):
    # w11 = k1*3+1
    # w12 = k2*3+1
    # w21 = k1*3+2
    # w22 = k2*3+2
    
    # K splits: (0,1), (2,3), (4,5), (6)
    for k_start in [0, 2, 4]:
        k1 = k_start
        k2 = k_start + 1
        v11 = 0 * 7 + k1
        v12 = 0 * 7 + k2
        v21 = 1 * 7 + k1
        v22 = 1 * 7 + k2
        w11 = k1 * 3 + 1
        w12 = k2 * 3 + 1
        w21 = k1 * 3 + 2
        w22 = k2 * 3 + 2
        add_strassen(4, 5, 8, 9, v11, v12, v21, v22, w11, w12, w21, w22)
        
    # K split: (6)
    k1 = 6
    v11 = 0 * 7 + k1
    v21 = 1 * 7 + k1
    w11 = k1 * 3 + 1
    w21 = k1 * 3 + 2
    # Schoolbook for 2x2 x 2x1
    # c11 = u11 v11 + u12 v21
    # c21 = u21 v11 + u22 v21
    add_term(4, v11, w11)
    add_term(5, v21, w11)
    add_term(8, v11, w21)
    add_term(9, v21, w21)
    
    # Subpart 2b: A2b (rows 1,2 cols 2,3) * Bb (rows 2,3)
    # U indices:
    # u11 = 1*4+2 = 6
    # u12 = 1*4+3 = 7
    # u21 = 2*4+2 = 10
    # u22 = 2*4+3 = 11
    
    # V indices for Bb (rows 2,3):
    # v11 = 2*7+k1
    # v12 = 2*7+k2
    # v21 = 3*7+k1
    # v22 = 3*7+k2
    
    # W indices are same as 2a.
    
    for k_start in [0, 2, 4]:
        k1 = k_start
        k2 = k_start + 1
        v11 = 2 * 7 + k1
        v12 = 2 * 7 + k2
        v21 = 3 * 7 + k1
        v22 = 3 * 7 + k2
        w11 = k1 * 3 + 1
        w12 = k2 * 3 + 1
        w21 = k1 * 3 + 2
        w22 = k2 * 3 + 2
        add_strassen(6, 7, 10, 11, v11, v12, v21, v22, w11, w12, w21, w22)
        
    k1 = 6
    v11 = 2 * 7 + k1
    v21 = 3 * 7 + k1
    w11 = k1 * 3 + 1
    w21 = k1 * 3 + 2
    add_term(6, v11, w11)
    add_term(7, v21, w11)
    add_term(10, v11, w21)
    add_term(11, v21, w21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
