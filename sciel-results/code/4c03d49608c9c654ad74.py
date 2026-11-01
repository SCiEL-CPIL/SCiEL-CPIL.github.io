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

    def compute_T_val(U, V, W, R):
        """Compute tensor values T_val[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T_val = np.zeros((n, m, p))
        for c in range(R):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T_val[i, j, :] += Uc[i, j] * (Vc[j, :] * Wc[:, i])
        return T_val

    def compute_grads(T_val, U, V, W, R):
        """Compute gradients of loss = sum(T_val - 1)^2 w.r.t. U, V, W"""
        dE_dT = 2.0 * (T_val - 1.0)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(R):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dT = dE_dT[i, j, k]
                        dU[i * m + j, c] += dT * Vc[j, k] * Wc[k, i]
                        dV[j * p + k, c] += dT * Uc[i, j] * Wc[k, i]
                        dW[k * n + i, c] += dT * Uc[i, j] * Vc[j, k]
        
        return dU, dV, dW

    def optimize(R, max_iters=500, lr=0.05, seed=0, init_U=None, init_V=None, init_W=None):
        """Optimize factor matrices for given rank R using gradient descent."""
        np.random.seed(seed)
        
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            U = np.random.uniform(-1, 1, (n * m, R))
            V = np.random.uniform(-1, 1, (m * p, R))
            W = np.random.uniform(-1, 1, (p * n, R))
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute T_val
            T_val = compute_T_val(U, V, W, R)
            
            # Error
            loss = np.sum((T_val - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                # Round and verify
                Ur = np.array([[round_to_half(x) for x in row] for row in U])
                Vr = np.array([[round_to_half(x) for x in row] for row in V])
                Wr = np.array([[round_to_half(x) for x in row] for row in W])
                
                if verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), R):
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
            
            # Gradient
            dU, dV, dW = compute_grads(T_val, U, V, W, R)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                eff_lr = lr * (1.0 + 0.1 * np.exp(-it / 50.0))
                U -= eff_lr * dU / (1.0 + grad_norm * 0.1)
                V -= eff_lr * dV / (1.0 + grad_norm * 0.1)
                W -= eff_lr * dW / (1.0 + grad_norm * 0.1)
            
            # Periodic restart with different seed if stuck
            if it % 100 == 0 and it > 0:
                if loss > best_loss * 1.1:
                    U = best_U.copy()
                    V = best_V.copy()
                    W = best_W.copy()
        
        # Final attempt to round and verify
        Ur = np.array([[round_to_half(x) for x in row] for row in best_U])
        Vr = np.array([[round_to_half(x) for x in row] for row in best_V])
        Wr = np.array([[round_to_half(x) for x in row] for row in best_W])
        
        if verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), R):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        
        return None

    def strassen_init(R):
        """Initialize using Strassen-like structure for 2x2x2 blocks."""
        # For 2x5x5, we can use Strassen on 2x2x2 sub-problems
        # and standard multiplication elsewhere
        
        # Strassen constants for 2x2x2 (7 terms)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For our case, we'll create an initial guess based on 
        # combining Strassen patterns across the 5x5 dimension
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Use 7 Strassen terms for each pair of 2x2 blocks
        # We have floor(5/2)=2 full 2x2 blocks and 1 remaining row/col
        
        strassen_patterns = [
            # (U_pattern, V_pattern, W_pattern) for each of 7 Strassen terms
            # Each pattern is a dict mapping (i,j) -> coefficient
        ]
        
        # Simple initialization: random with some structure
        np.random.seed(42)
        
        # Create structured initialization
        for c in range(R):
            # Random weights
            U[:, c] = np.random.uniform(-0.5, 0.5, n * m)
            V[:, c] = np.random.uniform(-0.5, 0.5, m * p)
            W[:, c] = np.random.uniform(-0.5, 0.5, p * n)
        
        return U, V, W

    # Try to find a better rank
    ranks_to_try = [45, 42, 40, 38, 35, 32, 30, 28, 25]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds and initialization strategies
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            
            # Use different initialization for different seeds
            if seed < 4:
                # Random initialization
                res = optimize(R, max_iters=300, lr=0.05, seed=seed)
            else:
                # Structured initialization
                init_U, init_V, init_W = strassen_init(R)
                np.random.seed(seed)
                # Add small noise
                init_U += np.random.uniform(-0.1, 0.1, init_U.shape)
                init_V += np.random.uniform(-0.1, 0.1, init_V.shape)
                init_W += np.random.uniform(-0.1, 0.1, init_W.shape)
                res = optimize(R, max_iters=300, lr=0.05, seed=seed, 
                              init_U=init_U, init_V=init_V, init_W=init_W)
            
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        
        # If no success, try with more iterations for promising ranks
        if R in [40, 35, 30] and time.time() < deadline - 20:
            for seed in range(4):
                if time.time() > deadline - 10:
                    break
                res = optimize(R, max_iters=800, lr=0.02, seed=100+seed)
                if res:
                    U, V, W = res
                    return {"U": U, "V": V, "W": W}
                
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
