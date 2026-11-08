import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe under 150s limit

    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def compute_residual(U, V, W):
        rank = U.shape[1]
        resid = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    resid += (val - 1.0)**2
        return resid

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
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                U[1*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                W[k1*n + 1, col] -= 1
                col += 1
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] -= 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] -= 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                U[0*m + j0, col] += 1
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] -= 1
                W[k1*n + 0, col] += 1
                col += 1
                U[1*m + j0, col] += 1
                U[0*m + j0, col] -= 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
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

    best_result = rank54_decomp()
    
    # Try to find lower rank solutions
    found_lower = False
    # Prioritize rank 53 as it is the most likely candidate for improvement
    for target_rank in [53, 52, 51]:
        if time.time() > deadline - 10:
            break
        
        base = rank54_decomp()
        U_base = base["U"]
        V_base = base["V"]
        W_base = base["W"]
        
        U_init = U_base[:, :target_rank].copy()
        V_init = V_base[:, :target_rank].copy()
        W_init = W_base[:, :target_rank].copy()
        
        # Try multiple perturbations
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
            
            try:
                rng = np.random.RandomState(42 + attempt * 13 + target_rank * 100)
                # Try different noise scales
                noise_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
                scale = noise_scales[attempt % len(noise_scales)]
                
                U = U_init + rng.randn(n*m, target_rank) * scale
                V = V_init + rng.randn(m*p, target_rank) * scale
                W = W_init + rng.randn(p*n, target_rank) * scale
                
                def objective(x):
                    U_flat = x[:n*m*target_rank]
                    V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = x[n*m*target_rank + m*p*target_rank:]
                    U_r = U_flat.reshape(n*m, target_rank)
                    V_r = V_flat.reshape(m*p, target_rank)
                    W_r = W_flat.reshape(p*n, target_rank)
                    return compute_residual(U_r, V_r, W_r)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round):
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        found_lower = True
                        break
            except Exception:
                continue
        
        if found_lower:
            break

    if not found_lower:
        best_result = rank54_decomp()
        best_result = {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
        
    return best_result
# EVOLVE-BLOCK-END
