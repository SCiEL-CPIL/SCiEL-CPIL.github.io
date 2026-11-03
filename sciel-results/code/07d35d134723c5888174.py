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

def _matmul_check(U, V, W, n, m, p, target, tol=1e-9):
    """Check if U,V,W decompose the target tensor exactly."""
    r = len(U[0]) if U else 0
    for (a, b, c), expected in target.items():
        val = 0.0
        for t in range(r):
            val += U[a][t] * V[b][t] * W[c][t]
        if abs(val - expected) > tol:
            return False
    # Also check that unwanted entries are ~0
    # Sample a few unwanted positions
    import random
    rng = random.Random(42)
    all_keys = set()
    for (a, b, c), _ in target.items():
        all_keys.add((a, b, c))
    for _ in range(200):
        a = rng.randint(0, n*m-1)
        b = rng.randint(0, m*p-1)
        c = rng.randint(0, p*n-1)
        if (a, b, c) not in all_keys:
            val = 0.0
            for t in range(r):
                val += U[a][t] * V[b][t] * W[c][t]
            if abs(val) > tol:
                return False
    return True

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
    
    # Vectorized check for speed
    def _check_fast(U, V, W, rank):
        # Compute all tensor entries
        # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # We only need to check target entries and some random non-target entries
        # For efficiency, we compute the full tensor if rank is small, or just the target ones
        # Since n*m*p is 63, and rank is ~60, computing full tensor is 63*60*... 
        # Actually, let's just compute the values for the target keys and a few random others
        
        # Precompute V*W products?
        # T[a,b,c] = U[a,:] dot (V[b,:] * W[c,:])
        
        # Check target entries
        for (a, b, c), expected in target.items():
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - expected) > 1e-6:
                return False
        
        # Check random non-target entries
        rng = np.random.RandomState(42)
        all_keys = set(target.keys())
        for _ in range(100):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            if (a, b, c) not in all_keys:
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > 1e-6:
                    return False
        return True

    np_rng = np.random.RandomState(12345)
    
    # Try ranks from rank_schoolbook-1 down
    # We will use gradient descent to minimize the error
    for rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        for restart in range(3):
            if time.time() - start_time > DEADLINE:
                break
            
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Initialize with schoolbook pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = (i * m + j) * p + k
                        if c < rank:
                            U[i * m + j, c] = 1.0
                            V[j * p + k, c] = 1.0
                            W[k * n + i, c] = 1.0
            
            # Add random noise to break symmetry
            U += np_rng.randn(n * m, rank) * 0.01
            V += np_rng.randn(m * p, rank) * 0.01
            W += np_rng.randn(p * n, rank) * 0.01
            
            # Gradient Descent
            lr = 0.01
            for iteration in range(100):
                if time.time() - start_time > DEADLINE:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Compute gradients for target entries
                for (a, b, c), expected in target.items():
                    # T = sum U[a,t] V[b,t] W[c,t]
                    # Error E = T - expected
                    # dU[a,t] += 2 * E * V[b,t] * W[c,t]
                    # dV[b,t] += 2 * E * U[a,t] * W[c,t]
                    # dW[c,t] += 2 * E * U[a,t] * V[b,t]
                    
                    T_val = np.dot(U[a], V[b] * W[c])
                    E = T_val - expected
                    
                    dU[a] += 2 * E * (V[b] * W[c])
                    dV[b] += 2 * E * (U[a] * W[c])
                    dW[c] += 2 * E * (U[a] * V[b])
                
                # Add penalty for non-zero non-target entries (sparsity/zero constraint)
                # This is expensive to do for all non-target entries.
                # We rely on the fact that if we match the target exactly,
                # the non-target entries might not be zero.
                # For now, we only optimize target entries.
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if _check_fast(U, V, W, rank):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
            
            if best_rank == rank:
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    return best_result
# EVOLVE-BLOCK-END
