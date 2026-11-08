"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe (evaluator kills at 150s)
    
    def rank54_decomp():
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P2
                U[1*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                W[k1*n + 1, col] -= 1
                col += 1
                # P3
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] -= 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P4
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] -= 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                # P5
                U[0*m + j0, col] += 1
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] -= 1
                W[k1*n + 0, col] += 1
                col += 1
                # P6
                U[1*m + j0, col] += 1
                U[0*m + j0, col] -= 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P7
                U[0*m + j1, col] += 1
                U[1*m + j1, col] -= 1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-6):
        # Vectorized verification
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        
        # Reshape for broadcasting
        # U_ijk[t] = U[i*m+j, t]
        # V_ijk[t] = V[j*p+k, t]
        # W_ijk[t] = W[k*n+i, t]
        
        # This is tricky to do in one shot without huge memory.
        # Loop over i, j, k is 2*5*6 = 60 iterations. Fast enough.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    best_result = rank54_decomp()
    best_rank = 54
    
    # Attempt rank reduction
    for target_rank in [53, 52, 51, 50]:
        if time.time() > deadline - 30:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            try:
                rng = np.random.RandomState(42 + attempt * 13)
                U0, V0, W0 = best_result['U'], best_result['V'], best_result['W']
                
                # Remove random columns
                cols_to_keep = list(range(54))
                cols_to_remove = rng.choice(54, size=54 - target_rank, replace=False)
                cols_to_keep = [c for c in cols_to_keep if c not in cols_to_remove]
                
                U = np.array(U0[:, cols_to_keep], dtype=float)
                V = np.array(V0[:, cols_to_keep], dtype=float)
                W = np.array(W0[:, cols_to_keep], dtype=float)
                
                # Perturb
                U += rng.randn(n*m, target_rank) * 0.1
                V += rng.randn(m*p, target_rank) * 0.1
                W += rng.randn(p*n, target_rank) * 0.1
                
                # Gradient Descent
                lr = 0.005
                for it in range(1000):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute Residual R_ijk
                    R = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
                    
                    resid = np.sum(R**2)
                    if resid < 1e-10:
                        found = True
                        break
                    
                    # Compute Gradients
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                r = R[i,j,k]
                                if r == 0: continue
                                dU[i*m+j, :] += r * V[j*p+k, :] * W[k*n+i, :]
                                dV[j*p+k, :] += r * U[i*m+j, :] * W[k*n+i, :]
                                dW[k*n+i, :] += r * U[i*m+j, :] * V[j*p+k, :]
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                if found:
                    best_result = {"U": U, "V": W, "W": W} # Wait, W is W
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            except Exception:
                continue
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
