import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 6 Strassen blocks (2x2x2) + 6 Schoolbook blocks (1x2x2)
    # 6 * 7 + 6 * 4 = 42 + 24 = 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for i in {0, 1}
    # i_0 = 0
    # j_0 in {0, 2}
    # k_0 in {0, 2, 4}
    
    for j_0 in [0, 2]:
        for k_0 in [0, 2, 4]:
            # 7 terms
            # M1
            U[0*4 + 0][col] += 1
            U[1*4 + 1][col] += 1
            V[0*6 + 0][col] += 1
            V[1*6 + 1][col] += 1
            W[0*3 + 0][col] += 1
            W[1*3 + 0][col] += 0 # Wait, W indices are k*n+i
            # C00 = M1... W[0,0] is k=0, i=0. Index 0*3+0=0.
            # C01 = ... W[1,0] is k=1, i=0. Index 1*3+0=3.
            # C10 = ... W[0,1] is k=0, i=1. Index 0*3+1=1.
            # C11 = ... W[1,1] is k=1, i=1. Index 1*3+1=4.
            
            # Let's map local (i_loc, j_loc, k_loc) to global
            # i_global = i_0 + i_loc
            # j_global = j_0 + j_loc
            # k_global = k_0 + k_loc
            
            # M1: U(0,0), U(1,1); V(0,0), V(1,1); W(0,0)
            # Note: U, V, W are accumulated, so we just add to the specific indices.
            
            # We need to be careful with the offsets.
            pass
