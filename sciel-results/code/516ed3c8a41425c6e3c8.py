import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank_known = 28
    
    # Build the target tensor T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1
    
    # Build the known rank-28 Strassen solution
    def build_strassen_28():
        rank = 28
        U = np.zeros((8, rank))
        V = np.zeros((16, rank))
        W = np.zeros((8, rank))
        
        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            # M1
            U[X_rows[0], col_offset + 0] += 1
            U[X_rows[3], col_offset + 0] += 1
            V[Y_rows[0], col_offset + 0] += 1
            V[Y_rows[3], col_offset + 0] += 1
            W[Z_rows[0], col_offset + 0] += 1
            W[Z_rows[3], col_offset + 0] += 1
            # M2
            U[X_rows[2], col_offset + 1] += 1
            U[X_rows[3], col_offset + 1] += 1
            V[Y_rows[0], col_offset + 1] += 1
            W[Z_rows[2], col_offset + 1] += 1
            W[Z_rows[3], col_offset + 1] -= 1
            # M3
            U[X_rows[0], col_offset + 2] += 1
            V[Y_rows[1], col_offset + 2] += 1
            V[Y_rows[3], col_offset + 2] -= 1
            W[Z_rows[1], col_offset + 2] += 1
            W[Z_rows[3], col_offset + 2] += 1
            # M4
            U[X_rows[3], col_offset + 3] += 1
            V[Y_rows[2], col_offset + 3] += 1
            V[Y_rows[0], col_offset + 3] -= 1
            W[Z_rows[0], col_offset + 3] += 1
            W[Z_rows[2], col_offset + 3] += 1
            # M5
            U[X_rows[0], col_offset + 4] += 1
            U[X_rows[1], col_offset + 4] += 1
            V[Y_rows[3], col_offset + 4] += 1
            W[Z_rows[0], col_offset + 4] -= 1
            W[Z_rows[1], col_offset + 4] += 1
            # M6
            U[X_rows[2], col_offset + 5] += 1
            U[X_rows[0], col_offset + 5] -= 1
            V[Y_rows[0], col_offset + 5] += 1
            V[Y_rows[1], col_offset + 5] += 1
            W[Z_rows[3], col_offset + 5] += 1
            # M7
            U[X_rows[1], col_offset + 6] += 1
            U[X_rows[3], col_offset + 6] -= 1
            V[Y_rows[2], col_offset + 6] += 1
            V[Y_rows[3], col_offset + 6] += 1
            W[Z_rows[0], col_offset + 6] += 1
        
        X1 = [0, 1, 4, 5]; Y1 = [0, 1, 4, 5]; Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)
        X2 = [2, 3, 6, 7]; Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)
        X3 = [0, 1, 4, 5]; Y3 = [2, 3, 6, 7]; Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)
        X4 = [2, 3, 6, 7]; Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)
        
        return U, V, W
    
    U28, V28, W28 = build_strassen_28()
    
    def check_solution(U, V, W, rank, tol=1e-9):
        # T_est[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
        # Use einsum for efficiency
        T_est = np.einsum('xr,yr,zr->xyz', U, V, W)
        return np.allclose(T_est, T, atol=tol)
    
    # Verify the known solution
    assert check_solution(U28, V28, W28, 28), "Known solution failed verification"
    
    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28
    
    # Convert to lists of lists for the return interface
    def to_list(arr):
        return [[float(v) for v in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
