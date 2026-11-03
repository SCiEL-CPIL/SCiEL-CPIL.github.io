"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target tensor as a dictionary of nonzero entries.
    T[(i,j), (j',k), (k'',i'')] = 1 if j==j' and i==i' and k==k''
    """
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                target[(idx1, idx2, idx3)] = 1.0
    return target

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # seconds, with margin before 150s kill
    
    target = _build_target_tensor(n, m, p)
    rank_schoolbook = n * m * p
    
    # Start with schoolbook decomposition as fallback
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook
    
    # Precompute all target keys and non-target keys for efficient checking
    all_keys = set(target.keys())
    
    def _full_check(U, V, W, rank, tol=1e-8):
        """Check if U,V,W decompose the target tensor exactly, including zero constraints."""
        # Check target entries
        for (a, b, c), expected in target.items():
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - expected) > tol:
                return False
        
        # Check a sample of non-target entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            if (a, b, c) not in all_keys:
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > tol:
                    return False
        return True

    def _compute_full_error(U, V, W, rank):
        """Compute the total error including zero constraints."""
        error = 0.0
        # Target entries
        for (a, b, c), expected in target.items():
            val = np.dot(U[a], V[b] * W[c])
            error += (val - expected) ** 2
        
        # Sample non-target entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            if (a, b, c) not in all_keys:
                val = np.dot(U[a], V[b] * W[c])
                error += val ** 2
        return error

    np_rng = np.random.RandomState(12345)
    
    # Try ranks from rank_schoolbook-1 down
    for rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE:
                break
            
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Initialize with schoolbook pattern where possible
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        if c < rank:
                            U[i * m + j, c] = 1.0
                            V[j * p + k, c] = 1.0
                            W[k * n + i, c] = 1.0
            
            # Add random noise to break symmetry
            U += np_rng.randn(n * m, rank) * 0.1
            V += np_rng.randn(m * p, rank) * 0.1
            W += np_rng.randn(p * n, rank) * 0.1
            
            # Gradient Descent with momentum and adaptive learning rate
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(200):
                if time.time() - start_time > DEADLINE:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Compute gradients for target entries
                for (a, b, c), expected in target.items():
                    T_val = np.dot(U[a], V[b] * W[c])
                    E = T_val - expected
                    
                    dU[a] += 2 * E * (V[b] * W[c])
                    dV[b] += 2 * E * (U[a] * W[c])
                    dW[c] += 2 * E * (U[a] * V[b])
                
                # Add penalty for non-zero non-target entries
                rng_iter = np.random.RandomState(iteration + 12345)
                for _ in range(50):
                    a = rng_iter.randint(0, n*m)
                    b = rng_iter.randint(0, m*p)
                    c = rng_iter.randint(0, p*n)
                    if (a, b, c) not in all_keys:
                        T_val = np.dot(U[a], V[b] * W[c])
                        # Gradient of T_val^2 w.r.t. U[a,t] is 2*T_val*V[b,t]*W[c,t]
                        dU[a] += 2 * T_val * (V[b] * W[c])
                        dV[b] += 2 * T_val * (U[a] * W[c])
                        dW[c] += 2 * T_val * (U[a] * V[b])
                
                # Update with momentum
                vU = momentum * vU + (1 - momentum) * dU
                vV = momentum * vV + (1 - momentum) * dV
                vW = momentum * vW + (1 - momentum) * dW
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                # Track best solution
                if iteration % 10 == 0:
                    err = _compute_full_error(U, V, W, rank)
                    if err < best_error:
                        best_error = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Check convergence
                if _full_check(U, V, W, rank):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
            
            if best_rank == rank:
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    return best_result
# EVOLVE-BLOCK-END
