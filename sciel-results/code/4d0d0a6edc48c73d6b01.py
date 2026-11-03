"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def strassen_2x2_2x2(col, u_rows, v_rows, w_rows):
        ua, ub, uc, ud = u_rows
        ve, vf, vg, vh = v_rows
        w00, w01, w10, w11 = w_rows
        
        U[ua][col] += 1; U[ud][col] += 1; V[ve][col] += 1; V[vh][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        U[uc][col] += 1; U[ud][col] += 1; V[ve][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        
        U[ua][col] += 1; V[vf][col] += 1; V[vh][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        U[ud][col] += 1; V[vg][col] += 1; V[ve][col] += -1
        W[w00][col] += -1; W[w10][col] += 1
        col += 1
        
        U[ua][col] += 1; U[ub][col] += 1; V[vh][col] += 1
        W[w00][col] += 1; W[w01][col] += 1
        col += 1
        
        U[uc][col] += 1; U[ua][col] += -1; V[ve][col] += 1; V[vf][col] += 1
        W[w11][col] += 1
        col += 1
        
        U[ub][col] += 1; U[ud][col] += -1; V[vg][col] += 1; V[vh][col] += 1
        W[w00][col] += -1
        col += 1
        
        return col
    
    def direct_2x2_2x1(col, u_rows, v_rows, w_rows):
        ua, ub, uc, ud = u_rows
        ve, vg = v_rows
        w0, w1 = w_rows
        
        U[ua][col] += 1; V[ve][col] += 1; W[w0][col] += 1; col += 1
        U[ub][col] += 1; V[vg][col] += 1; W[w0][col] += 1; col += 1
        U[uc][col] += 1; V[ve][col] += 1; W[w1][col] += 1; col += 1
        U[ud][col] += 1; V[vg][col] += 1; W[w1][col] += 1; col += 1
        return col
    
    col = 0
    a1_u = [0, 1, 4, 5]
    a2_u = [2, 3, 6, 7]
    b1_01_v = [0, 1, 5, 6]
    b1_23_v = [2, 3, 7, 8]
    b1_4_v = [4, 9]
    b2_01_v = [10, 11, 15, 16]
    b2_23_v = [12, 13, 17, 18]
    b2_4_v = [14, 19]
    w_01 = [0, 1, 2, 3]
    w_23 = [4, 5, 6, 7]
    w_4 = [8, 9]
    
    col = strassen_2x2_2x2(col, a1_u, b1_01_v, w_01)
    col = strassen_2x2_2x2(col, a1_u, b1_23_v, w_23)
    col = direct_2x2_2x1(col, a1_u, b1_4_v, w_4)
    col = strassen_2x2_2x2(col, a2_u, b2_01_v, w_01)
    col = strassen_2x2_2x2(col, a2_u, b2_23_v, w_23)
    col = direct_2x2_2x1(col, a2_u, b2_4_v, w_4)
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
