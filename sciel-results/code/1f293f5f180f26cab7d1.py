"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    def make_rank8():
        rank = 8
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.ones((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j, col] = 1
                for k in range(p):
                    V[j*p+k, col] = 1
                W[:, col] = 1
                col += 1
        return U, V, W
    
    U, V, W = make_rank8()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
