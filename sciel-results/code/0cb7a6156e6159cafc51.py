import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # 1. M_2,2,2 for A_L (rows 0,1 of A), B_T1 (rows 0,1 of B), C_11 (row 0, cols 0,1 of C)
    # A rows: 0, 1. B rows: 0, 1. C row: 0, C cols: 0, 1.
    # Strassen
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C11, C01? No.
    # C00 is at W[0*2+0]=0. C01 is at W[1*2+0]=2.
    # C10 is at W[0*2+1]=1. C11 is at W[1*2+1]=3.
    
    # Let's define indices clearly.
    # A indices: i*m + j. i in [0,1], j in [0,1] for A_L.
    # B indices: j*p + k. j in [0,1], k in [0,1] for B_T1.
    # C indices: k*n + i. k in [0,1], i in [0,0] for C_11.
    
    # Strassen M1
    add_term(0, 0, 0) # A00 B00 -> C00
    add_term(0, 1, 1) # A00 B10? No.
    # Strassen formulas are linear combinations.
    # We add basis terms.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # This expands to A00B00 + A00B11 + A11B00 + A11B11
    # But we need to group them into the 7 multiplications.
    # The "multiplication" is a single rank-1 term in the decomposition.
    # So for M1, U has 1s at A00, A11. V has 1s at B00, B11. W has 1s at C00, C10, C01, C11?
    # No, W gets the result of the multiplication.
    # The term is $u \cdot v \cdot w$.
    # $u$ is a vector in A-space. $v$ in B-space. $w$ in C-space.
    # For M1: $u = e_{A00} + e_{A11}$. $v = e_{B00} + e_{B11}$.
    # The product is $C_{00} + C_{10} + C_{01} + C_{11}$?
    # No, $C = A B$.
    # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$
    # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$
    # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$
    # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$
    
    # Strassen M1 contributes to all 4 C entries?
    # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
    # These terms appear in:
    # $C_{00}$: $A_{00}B_{00}$
    # $C_{01}$: $A_{00}B_{11}$
    # $C_{10}$: $A_{11}B_{00}$
    # $C_{11}$: $A_{11}B_{11}$
    # So $W$ for M1 has 1s at $C_{00}, C_{01}, C_{10}, C_{11}$.
    
    # Let's implement a helper for Strassen block.
    def add_strassen(a_i, a_j, b_j, b_k, c_k, c_i):
        # a_i, a_j: row indices in A (2 rows)
        # b_j, b_k: row indices in B (2 rows) -- wait, B is m x p.
        # In our local 2x2, B rows are j, k? No.
        # B is 2x2. Rows 0,1. Cols 0,1.
        # c_k, c_i: col index in C, row index in C.
        
        # M1
        U_idx = [a_i*5+0, a_i*5+1, (a_i+1)*5+0, (a_i+1)*5+1] # Wrong indexing.
        # A is 2x5. Rows 0,1. Cols 0,1 for A_L.
        # So A indices: 0*5+0, 0*5+1, 1*5+0, 1*5+1.
        
        pass

    # This manual construction is error prone.
    # I will write the code to generate the blocks programmatically.
    
    return {"U": U, "V": V, "W": W}
