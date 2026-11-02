import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (12, rank)
    # V: (m*p, rank) -> (16, rank)
    # W: (p*n, rank) -> (12, rank)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    def add_strassen(a_11, a_12, a_21, a_22, b_11, b_12, b_21, b_22, c_11, c_12, c_21, c_22):
        # Strassen 2x2x2
        # M1 = (a11+a22)(b11+b22)
        add_term(a_11, b_11, c_11, 1, 1, 1)
        add_term(a_11, b_22, c_11, 1, 1, 1)
        add_term(a_22, b_11, c_11, 1, 1, 1)
        add_term(a_22, b_22, c_11, 1, 1, 1)
        
        # M2 = (a21+a22)b11
        add_term(a_21, b_11, c_21, 1, 1, 1)
        add_term(a_22, b_11, c_21, 1, 1, 1)
        
        # M3 = a11(b12-b22)
        add_term(a_11, b_12, c_12, 1, 1, 1)
        add_term(a_11, b_22, c_12, 1, -1, 1)
        
        # M4 = a22(b21-b11)
        add_term(a_22, b_21, c_21, 1, 1, 1)
        add_term(a_22, b_11, c_21, 1, -1, 1)
        
        # M5 = (a11+a12)b22
        add_term(a_11, b_22, c_11, 1, 1, -1) # C11 gets -M5
        add_term(a_12, b_22, c_11, 1, 1, -1)
        add_term(a_11, b_22, c_12, 1, 1, 1) # C12 gets M5
        add_term(a_12, b_22, c_12, 1, 1, 1)
        
        # M6 = (a21-a11)(b11+b12)
        add_term(a_21, b_11, c_22, 1, 1, 1)
        add_term(a_21, b_12, c_22, 1, 1, 1)
        add_term(a_11, b_11, c_22, -1, 1, 1)
        add_term(a_11, b_12, c_22, -1, 1, 1)
        
        # M7 = (a12-a22)(b21+b22)
        add_term(a_12, b_21, c_11, 1, 1, 1)
        add_term(a_12, b_22, c_11, 1, 1, 1)
        add_term(a_22, b_21, c_11, -1, 1, 1)
        add_term(a_22, b_22, c_11, -1, 1, 1)
        
        # Wait, the standard Strassen formulas are:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My add_term calls above mixed the coefficients into the W matrix.
        # This is correct because W entries can be any number.
        # However, I need to be careful with the indices.
        
        # Let's re-verify the mapping.
        # The function add_term adds to U, V, W at the current column.
        # It does not handle multiple contributions to the same C entry in one call unless I pass different W indices.
        # But Strassen combines M's into C's.
        # The tensor decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is $u \otimes v \otimes w$.
        # If I have $C_{11} = M_1 + M_4 - M_5 + M_7$, this means the output $C_{11}$ is a linear combination of the intermediate products.
        # In the tensor decomposition, the $W$ vector for a term corresponding to $M_1$ should have a 1 at the $C_{11}$ position if $M_1$ contributes to $C_{11}$.
        # But $M_1$ is an intermediate scalar.
        # The standard way to express Strassen as a tensor decomposition is to have 7 terms, where the W vectors are the linear combinations.
        # Term 1 (M1): U has coeffs for a11,a22. V has coeffs for b11,b22. W has 1 at c11 and 1 at c22.
        # Term 2 (M2): U has coeffs for a21,a22. V has coeff for b11. W has 1 at c21 and -1 at c22.
        # ...
        
        # My previous implementation added terms to U, V, W directly.
        # This is valid if I treat each "multiplication" as a rank-1 tensor.
        # But Strassen reuses the multiplications.
        # The rank-1 tensors in the decomposition are the 7 multiplications $M_k$.
        # So we should add 7 terms to the global U, V, W.
        # For each $M_k$, we define the U, V, W vectors.
        
        # Let's restart the add_strassen logic to add 7 terms.
        
        # I will delete the previous add_term calls in the actual code and use this logic.
        
        # For M1:
        # U: a11=1, a22=1
        # V: b11=1, b22=1
        # W: c11=1, c22=1
        # For M2:
        # U: a21=1, a22=1
        # V: b11=1
        # W: c21=1, c22=-1
        # For M3:
        # U: a11=1
        # V: b12=1, b22=-1
        # W: c12=1, c22=1
        # For M4:
        # U: a22=1
        # V: b21=1, b11=-1
        # W: c11=1, c21=1
        # For M5:
        # U: a11=1, a12=1
        # V: b22=1
        # W: c11=-1, c12=1
        # For M6:
        # U: a21=1, a11=-1
        # V: b11=1, b12=1
        # W: c22=1
        # For M7:
        # U: a12=1, a22=-1
        # V: b21=1, b22=1
        # W: c11=1
        
        # This is the correct way.
        
        pass

    # Correct add_strassen
    def add_strassen_correct(a_11, a_12, a_21, a_22, b_11, b_12, b_21, b_22, c_11, c_12, c_21, c_22):
        nonlocal col
        
        # M1
        U[a_11][col] += 1; U[a_22][col] += 1
        V[b_11][col] += 1; V[b_22][col] += 1
        W[c_11][col] += 1; W[c_22][col] += 1
        col += 1
        
        # M2
        U[a_21][col] += 1; U[a_22][col] += 1
        V[b_11][col] += 1
        W[c_21][col] += 1; W[c_22][col] += -1
        col += 1
        
        # M3
        U[a_11][col] += 1
        V[b_12][col] += 1; V[b_22][col] += -1
        W[c_12][col] += 1; W[c_22][col] += 1
        col += 1
        
        # M4
        U[a_22][col] += 1
        V[b_21][col] += 1; V[b_11][col] += -1
        W[c_11][col] += 1; W[c_21][col] += 1
        col += 1
        
        # M5
        U[a_11][col] += 1; U[a_12][col] += 1
        V[b_22][col] += 1
        W[c_11][col] += -1; W[c_12][col] += 1
        col += 1
        
        # M6
        U[a_21][col] += 1; U[a_11][col] += -1
        V[b_11][col] += 1; V[b_12][col] += 1
        W[c_22][col] += 1
        col += 1
        
        # M7
        U[a_12][col] += 1; U[a_22][col] += -1
        V[b_21][col] += 1; V[b_22][col] += 1
        W[c_11][col] += 1
        col += 1

    # 1. C_top_L (1x4 by 4x2)
    # A: 0,1,2,3
    # B: 0,1,2,3 (k=0) and 4,5,6,7 (k=1)
    # C: 0 (k=0,i=0), 3 (k=1,i=0)
    
    # C_00
    for j in range(4):
        U[j][col] = 1
        V[j][col] = 1
        W[0][col] = 1
        col += 1
        
    # C_10 (index 3)
    for j in range(4):
        U[j][col] = 1
        V[j+4][col] = 1
        W[3][col] = 1
        col += 1

    # 2. C_top_R (1x4 by 4x2)
    # B: 8..15
    # C: 6 (k=2,i=0), 9 (k=3,i=0)
    
    # C_20 (index 6)
    for j in range(4):
        U[j][col] = 1
        V[j+8][col] = 1
        W[6][col] = 1
        col += 1
        
    # C_30 (index 9)
    for j in range(4):
        U[j][col] = 1
        V[j+12][col] = 1
        W[9][col] = 1
        col += 1

    # 3. C_bot_L (2x4 by 4x2)
    # A: 4..11
    # B: 0..7
    # C: 1,2 (k=0), 4,5 (k=1)
    
    # Split A: 4,5,8,9 (L) and 6,7,10,11 (R)
    # Split B: 0,1,4,5 (T) and 2,3,6,7 (B)
    
    # A_L B_T
    add_strassen_correct(4, 5, 8, 9, 0, 1, 4, 5, 1, 4, 2, 5)
    
    # A_R B_B
    add_strassen_correct(6, 7, 10, 11, 2, 3, 6, 7, 1, 4, 2, 5)

    # 4. C_bot_R (2x4 by 4x2)
    # B: 8..15
    # C: 7,8 (k=2), 10,11 (k=3)
    
    # Split A: 4..11
    # Split B: 8,9,12,13 (T) and 10,11,14,15 (B)
    
    # A_L B_T
    add_strassen_correct(4, 5, 8, 9, 8, 9, 12, 13, 7, 10, 8, 11)
    
    # A_R B_B
    add_strassen_correct(6, 7, 10, 11, 10, 11, 14, 15, 7, 10, 8, 11)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
