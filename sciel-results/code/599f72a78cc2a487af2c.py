"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen for first 2 cols + Schoolbook for 3rd col
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        for i in u_idx:
            U[i][0] += val * u_idx[i] # This is wrong, u_idx is list of indices, val is scalar? 
        # Let's just do it manually or with a clearer helper
        pass

    # Strassen for 2x2x2 (cols 0,1 of B)
    # M1 = (a11 + a22)(b11 + b22)
    # U: a11(0), a22(3)
    # V: b11(0), b22(4)
    # W: C11(0)+, C22(3)+
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (a21 + a22)b11
    # U: a21(2), a22(3)
    # V: b11(0)
    # W: C21(1)+, C22(3)-
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    # M3 = a11(b12 - b22)
    # U: a11(0)
    # V: b12(1), b22(4)
    # W: C12(2)+, C22(3)+
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    # M4 = a22(b21 - b11)
    # U: a22(3)
    # V: b21(3), b11(0)
    # W: C11(0)+, C21(1)+
    U[3][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5 = (a11 + a12)b22
    # U: a11(0), a12(1)
    # V: b22(4)
    # W: C11(0)-, C12(2)+
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # M6 = (a21 - a11)(b11 + b12)
    # U: a21(2), a11(0)
    # V: b11(0), b12(1)
    # W: C22(3)+
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (a12 - a22)(b21 + b22)
    # U: a12(1), a22(3)
    # V: b21(3), b22(4)
    # W: C11(0)+
    U[1][6] += 1; U[3][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Schoolbook for 3rd col (col 2 of B)
    # M8 = a11*b13
    # U: a11(0)
    # V: b13(2)
    # W: C13(4)+
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # M9 = a12*b23
    # U: a12(1)
    # V: b23(5)
    # W: C13(4)+
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # M10 = a21*b13
    # U: a21(2)
    # V: b13(2)
    # W: C23(5)+
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # M11 = a22*b23
    # U: a22(3)
    # V: b23(5)
    # W: C23(5)+
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
