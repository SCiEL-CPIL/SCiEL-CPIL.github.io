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
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank
    best_result = rank54_decomp()
    best_rank = 54
    
    # Attempt rank 53, 52, 51 through optimization
    for target_rank in [53, 52, 51]:
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            try:
                rng = np.random.RandomState(42 + attempt * 7)
                # Initialize with rank54, remove random columns, then perturb
                U0, V0, W0 = rank54_decomp()
                U = np.array(U0, dtype=float)
                V = np.array(V0, dtype=float)
                W = np.array(W0, dtype=float)
                
                # Remove random (54 - target_rank) columns
                cols_to_keep = list(range(54))
                cols_to_remove = rng.choice(54, size=54 - target_rank, replace=False)
                cols_to_keep = [c for c in cols_to_keep if c not in cols_to_remove]
                U = U[:, cols_to_keep]
                V = V[:, cols_to_keep]
                W = W[:, cols_to_keep]
                
                # Perturb
                U += rng.randn(n*m, target_rank) * 0.05
                V += rng.randn(m*p, target_rank) * 0.05
                W += rng.randn(p*n, target_rank) * 0.05
                
                # Iterative refinement with adaptive learning rate
                lr = 0.01
                for it in range(500):
                    if time.time() > deadline - 10:
                        break
                    # Compute residual and gradients efficiently
                    resid = 0
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for t in range(target_rank):
                                    val = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                                    err = val - 1.0
                                    resid += err * err
                                    dU[i*m+j, t] += err * V[j*p+k, t] * W[k*n+i, t]
                                    dV[j*p+k, t] += err * U[i*m+j, t] * W[k*n+i, t]
                                    dW[k*n+i, t] += err * U[i*m+j, t] * V[j*p+k, t]
                    
                    if resid < 1e-10:
                        found = True
                        break
                    
                    # Adaptive learning rate
                    if resid < 1e-6:
                        lr = 0.001
                    elif resid < 1e-3:
                        lr = 0.005
                    else:
                        lr = 0.01
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                if found:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
            except Exception:
                continue
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
