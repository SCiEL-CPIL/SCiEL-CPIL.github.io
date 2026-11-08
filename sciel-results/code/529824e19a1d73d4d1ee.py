import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def dot_2x2x1():
        U = np.zeros((4, 1))
        V = np.zeros((2, 1))
        W = np.zeros((2, 1))
        U[:, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W
    
    def dot_1x2x2():
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        U[1, 1] = 1
        V[3, 1] = 1
        W[1, 1] = 1
        return U, V, W
    
    def dot_1x2x1():
        U = np.zeros((2, 1))
        V = np.zeros((2, 1))
        W = np.zeros((1, 1))
        U[:, 0] = 1
        V[:, 0] = 1
        W[0, 0] = 1
        return U, V, W

    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_b, V_b, W_b)
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_b, V_b, W_b)
    add_block([0, 1, 4, 5], [4, 9], [12, 13], *dot_2x2x1())
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_b, V_b, W_b)
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_b, V_b, W_b)
    add_block([2, 3, 6, 7], [14, 19], [12, 13], *dot_2x2x1())
    add_block([8, 9], [0, 1, 5, 6], [2, 5], *dot_1x2x2())
    add_block([8, 9], [2, 3, 7, 8], [8, 11], *dot_1x2x2())
    add_block([8, 9], [4, 9], [14], *dot_1x2x1())
    add_block([10, 11], [10, 11, 15, 16], [2, 5], *dot_1x2x2())
    add_block([10, 11], [12, 13, 17, 18], [8, 11], *dot_1x2x2())
    add_block([10, 11], [14, 19], [14], *dot_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
