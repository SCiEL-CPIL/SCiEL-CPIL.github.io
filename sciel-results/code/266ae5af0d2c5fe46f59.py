import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # Try ranks from 20 to 24
    for rank in range(20, 25):
        # Initialize
        U = np.random.rand(n * m, rank)
        V = np.random.rand(m * p, rank)
        W = np.random.rand(p * n, rank)
        
        # Iterative refinement
        for it in range(100):
            # Solve for U
            # A_U vec(U) = 1
            # A_U is (n*m*p) x (n*m*rank)
            # Row index corresponds to (i, j, k)
            # Col index corresponds to (i, j, r)
            # Entry A_U[(i,j,k), (i,j,r)] = V[j*p+k, r] * W[k*n+i, r]
            
            A_U = np.zeros((n * m * p, n * m * rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        for r in range(rank):
                            col = (i * m + j) * rank + r
                            A_U[row, col] = V[j * p + k, r] * W[k * n + i, r]
            
            # Solve least squares
            U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(n * m * p), rcond=None)
            U = U_flat.reshape(n * m, rank)
            
            # Solve for V
            # A_V vec(V) = 1
            # Row (i, j, k), Col (j, k, r)
            # Entry A_V[(i,j,k), (j,k,r)] = U[i*m+j, r] * W[k*n+i, r]
            A_V = np.zeros((n * m * p, m * p * rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        for r in range(rank):
                            col = (j * p + k) * rank + r
                            A_V[row, col] = U[i * m + j, r] * W[k * n + i, r]
            
            V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(n * m * p), rcond=None)
            V = V_flat.reshape(m * p, rank)
            
            # Solve for W
            # A_W vec(W) = 1
            # Row (i, j, k), Col (k, i, r)
            # Entry A_W[(i,j,k), (k,i,r)] = U[i*m+j, r] * V[j*p+k, r]
            A_W = np.zeros((n * m * p, p * n * rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        for r in range(rank):
                            col = (k * n + i) * rank + r
                            A_W[row, col] = U[i * m + j, r] * V[j * p + k, r]
            
            W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(n * m * p), rcond=None)
            W = W_flat.reshape(p * n, rank)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1.0) ** 2
            if error < 1e-6:
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Convert to list of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        # If rank 20 fails, try next rank
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
