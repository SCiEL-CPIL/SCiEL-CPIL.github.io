"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Precompute indices
    # U: (n*m, R) -> indices i*m+j
    # V: (m*p, R) -> indices j*p+k
    # W: (p*n, R) -> indices k*n+i
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from 50 down to 1
    # We'll start from 50 and try to reduce.
    # Actually, let's try a few specific ranks that might be optimal.
    # R(2, 5, 5) is likely around 30-40.
    
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1

    def verify(U, V, W, R):
        # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        # U, V, W are lists of lists
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half(x):
        return round(x * 2) / 2.0

    def optimize(R, max_iters=1000, lr=0.01, seed=0):
        np.random.seed(seed)
        # Random initialization
        U = np.random.uniform(-1, 1, (n * m, R))
        V = np.random.uniform(-1, 1, (m * p, R))
        W = np.random.uniform(-1, 1, (p * n, R))
        
        # Gradient descent on E = sum (sum_c ... - 1)^2
        # We can vectorize the computation of the tensor values
        # Tensor T_val[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute T_val
            T_val = np.zeros((n, m, p))
            for c in range(R):
                # U_col: (n*m) -> reshape to (n, m)
                # V_col: (m*p) -> reshape to (m, p)
                # W_col: (p*n) -> reshape to (p, n)
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                
                # We need to add Uc[i,j] * Vc[j,k] * Wc[k,i] to T_val[i,j,k]
                # This is an outer product of Uc, Vc, Wc with index coupling.
                # T_val[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
                
                # Efficient way:
                # For each i, j:
                #   T_val[i, j, :] += Uc[i, j] * (Vc[j, :] * Wc[:, i].T)
                
                for i in range(n):
                    for j in range(m):
                        T_val[i, j, :] += Uc[i, j] * (Vc[j, :] * Wc[:, i])
                        
            # Error
            error = np.sum((T_val - 1.0)**2)
            if error < 1e-8:
                # Round and verify
                Ur = np.array([[round_to_half(x) for x in row] for row in U])
                Vr = np.array([[round_to_half(x) for x in row] for row in V])
                Wr = np.array([[round_to_half(x) for x in row] for row in W])
                
                if verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), R):
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
            
            # Gradient
            # dE/dT_val = 2 * (T_val - 1)
            dE_dT = 2.0 * (T_val - 1.0)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(R):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                
                # dU[i*m+j, c] = sum_k dE_dT[i,j,k] * Vc[j,k] * Wc[k,i]
                # dV[j*p+k, c] = sum_i dE_dT[i,j,k] * Uc[i,j] * Wc[k,i]
                # dW[k*n+i, c] = sum_j dE_dT[i,j,k] * Uc[i,j] * Vc[j,k]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dT = dE_dT[i, j, k]
                            dU[i * m + j, c] += dT * Vc[j, k] * Wc[k, i]
                            dV[j * p + k, c] += dT * Uc[i, j] * Wc[k, i]
                            dW[k * n + i, c] += dT * Uc[i, j] * Vc[j, k]
                            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return None

    # Try to find a better rank
    # Start from 50 down to 20?
    # We have ~145s.
    # Let's try a few ranks.
    
    # R(2, 5, 5) might be 35?
    # Let's try 40, 35, 30...
    
    ranks_to_try = [45, 40, 35, 30, 25]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            res = optimize(R, max_iters=200, lr=0.05, seed=seed)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
