import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Trivial rank = n*m*p = 60
    rank_trivial = n * m * p
    
    # Try to find a better rank via integer programming / linear algebra
    # For 2x5x6 matrix multiplication, we know:
    # - Trivial: 60
    # - Block Strassen (6 blocks of 2x2x2): 54
    # - Potentially better with more clever decompositions
    
    # Let's try to construct a rank-54 decomposition using Strassen on 2x2x2 blocks
    # and see if we can do better.
    
    # Actually, let's try a different approach: use the known fact that
    # for 2xm x mxp, there are known bounds. Let's try to find the minimum
    # via a search over possible ranks.
    
    # For now, let's use the Strassen-based rank-54 decomposition which is verified.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # j pairs: (0,1), (2,3); k pairs: (0,1), (2,3), (4,5)
    # But wait, m=5, so j goes 0..4. Let's use j pairs (0,1), (2,3) and leave j=4 as standard.
    # p=6, k goes 0..5. k pairs (0,1), (2,3), (4,5)
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    strassen_cells = set()
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for i in range(n):
                for j in (j1, j2):
                    for k in (k1, k2):
                        strassen_cells.add((i, j, k))
    
    col = 0
    
    # Standard terms for non-Strassen cells
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_cells:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Strassen terms
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2
            
            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2
            
            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1
            
            # P1 = (a00+a11)(b00+b11)
            U[u00][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1
            
            # P2 = (a10+a11)b00
            U[u10][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1
            W[w11][col] += -1
            col += 1
            
            # P3 = a00(b01-b11)
            U[u00][col] += 1
            V[v01][col] += 1
            V[v11][col] += -1
            W[w01][col] += 1
            W[w11][col] += 1
            col += 1
            
            # P4 = a11(b10-b00)
            U[u11][col] += 1
            V[v10][col] += 1
            V[v00][col] += -1
            W[w00][col] += 1
            W[w10][col] += 1
            col += 1
            
            # P5 = (a00+a01)b11
            U[u00][col] += 1
            U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1
            W[w01][col] += 1
            col += 1
            
            # P6 = (a10-a00)(b00+b01)
            U[u10][col] += 1
            U[u00][col] += -1
            V[v00][col] += 1
            V[v01][col] += 1
            W[w11][col] += 1
            col += 1
            
            # P7 = (a01-a11)(b10+b11)
            U[u01][col] += 1
            U[u11][col] += -1
            V[v10][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        # Check for linear dependencies to potentially reduce rank
        # We can check if any column is a linear combination of others.
        # This is equivalent to checking the rank of the matrix formed by flattening U, V, W.
        # But a simpler check: try removing each column and see if the decomposition still holds?
        # That is O(r * nmp) which is 54 * 60 = 3240 ops. Very fast.
        
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        
        for c in range(rank):
            if time.time() > deadline - 10:
                break
                
            # Create copies without column c
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            
            if verify(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = rank - 1
                # If we found a reduction, we should restart the loop with the new matrices
                # to see if we can reduce further
                U, V, W = U_test, V_test, W_test
                rank = best_rank
                # Restart loop from 0
                c = -1 
                continue
        
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to trivial
        rank_f = n * m * p
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}
