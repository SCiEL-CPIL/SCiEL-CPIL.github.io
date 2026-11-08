import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # 1. Construct the target tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    def validate(U, V, W, atol=1e-6):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Need to map indices correctly for einsum
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # T index: i, j, k
        
        # Reshape to 3D
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Sum over rank dimension
        # U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def build_rank46_fallback():
        # Re-implement the known valid rank 46 from the original code
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    U46, V46, W46 = build_rank46_fallback()

    # 2. Attempt to find lower rank (45, 44...) using Randomized Gradient Descent
    # We try ranks from 45 down to 35
    for rank in range(45, 34, -1):
        # Try multiple random seeds
        for seed in range(100):
            np.random.seed(seed)
            # Initialize with small random values
            # Using a mix of integers and half-integers in initialization can help
            U = np.random.randn(U_dim, rank) * 0.5
            V = np.random.randn(V_dim, rank) * 0.5
            W = np.random.randn(W_dim, rank) * 0.5
            
            # Gradient Descent Parameters
            lr = 0.05
            steps = 300
            
            for step in range(steps):
                # Compute current product
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
                err = prod - T
                loss = np.sum(err * err)
                
                if loss < 1e-12:
                    break
                
                # Compute Gradients
                # dL/dU_ijr = sum_{k} err_ijk * V_jkr * W_kir
                # dL/dV_jkr = sum_{i} err_ijk * U_ijr * W_kir
                # dL/dW_kir = sum_{j} err_ijk * U_ijr * V_jkr
                
                dU_3d = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d, optimize=True)
                dV_3d = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d, optimize=True)
                dW_3d = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d, optimize=True)
                
                dU = dU_3d.reshape(U_dim, rank)
                dV = dV_3d.reshape(V_dim, rank)
                dW = dW_3d.reshape(W_dim, rank)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodically snap and validate
                if step % 10 == 0:
                    U_s = snap_to_half(U)
                    V_s = snap_to_half(V)
                    W_s = snap_to_half(W)
                    if validate(U_s, V_s, W_s):
                        return {
                            "U": [[float(x) for x in row] for row in U_s],
                            "V": [[float(x) for x in row] for row in V_s],
                            "W": [[float(x) for x in row] for row in W_s]
                        }
            
            # Final snap and validate after GD loop
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            if validate(U_s, V_s, W_s):
                return {
                    "U": [[float(x) for x in row] for row in U_s],
                    "V": [[float(x) for x in row] for row in V_s],
                    "W": [[float(x) for x in row] for row in W_s]
                }

    # 3. Fallback to Rank 46
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
