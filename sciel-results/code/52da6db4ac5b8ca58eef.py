"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-100 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition is exact for the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                prod_sum = np.dot(np.dot(u_val, v_val), w_val)
                if abs(prod_sum - 1.0) > tol:
                    return False
    return True

def _random_decomposition(rank, seed=0):
    """Generate a random rank-r decomposition attempt with integer/half-integer entries."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Use small integer/half-integer values
    def rand_entry():
        r = rng.random()
        if r < 0.1:
            return 0
        elif r < 0.3:
            return rng.choice([-1, 1])
        elif r < 0.5:
            return rng.choice([-0.5, 0.5])
        elif r < 0.7:
            return rng.choice([-2, 2])
        elif r < 0.85:
            return rng.choice([-1.5, 1.5])
        else:
            return rng.choice([-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3])
    
    U = [[rand_entry() for _ in range(rank)] for _ in range(n * m)]
    V = [[rand_entry() for _ in range(rank)] for _ in range(m * p)]
    W = [[rand_entry() for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def _strassen_based_decomposition():
    """
    Try to construct a decomposition using Strassen-like block structure.
    For 4x5x5, we can try to exploit the fact that 5 = 4+1 and use
    a hierarchical decomposition. This is a heuristic attempt.
    
    The key idea: if we can decompose the 5x5 multiplication using fewer
    than 25 scalar multiplications by exploiting algebraic structure,
    we can reduce the overall rank.
    """
    n, m, p = DIMENSIONS
    # For now, this is a placeholder for a structured approach.
    # We'll try to find a low-rank decomposition via optimization.
    return None

def _optimize_decomposition(rank, time_budget=10.0, seed=42):
    """
    Try to optimize a rank-r decomposition using gradient descent on the
    tensor completion problem. We minimize the error of the decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    scale = 0.5
    U = rng.randn(n * m, rank) * scale
    V = rng.randn(m * p, rank) * scale
    W = rng.randn(p * n, rank) * scale
    
    lr = 0.01
    patience = 50
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    no_improve = 0
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradient of loss = sum_{i,j,k} (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        # This is expensive, so we'll use a simplified approach.
        
        # Sample a subset of (i,j,k) triples for gradient estimation
        num_samples = min(100, n * m * p)
        samples = []
        for _ in range(num_samples):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            samples.append((i, j, k))
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k) in samples:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Compute the current value
            prod_sum = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
            error = prod_sum - 1.0
            
            # Gradients
            # d/dU[u_idx,c] = error * V[v_idx,c] * W[w_idx,c]
            grad_U[u_idx, :] += error * np.dot(V[v_idx, :], W[w_idx, :])
            
            # d/dV[v_idx,c] = error * U[u_idx,c] * W[w_idx,c]
            grad_V[v_idx, :] += error * np.dot(U[u_idx, :], W[w_idx, :])
            
            # d/dW[w_idx,c] = error * U[u_idx,c] * V[v_idx,c]
            grad_W[w_idx, :] += error * np.dot(U[u_idx, :], V[v_idx, :])
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Compute loss periodically
        if step % 100 == 0:
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        prod_sum = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                        loss += (prod_sum - 1.0) ** 2
            loss /= (n * m * p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # Decay learning rate
        lr *= 0.995
    
    # Round to nearest half-integer
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    # Verify
    U_list = U_rounded.tolist()
    V_list = V_rounded.tolist()
    W_list = W_rounded.tolist()
    
    if _verify(U_list, V_list, W_list, tol=1e-6):
        return U_list, V_list, W_list
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix
    by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with schoolbook (rank 100)
    best_result = _schoolbook()
    best_rank = 100
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Upper bound: schoolbook = 100
    
    # Try various ranks from low to high
    for rank in range(25, 100):
        if time.time() > deadline - 15.0:
            break
        
        # Try a few random seeds for this rank
        for seed in range(3):
            if time.time() > deadline - 15.0:
                break
            
            result = _optimize_decomposition(rank, time_budget=8.0, seed=seed + rank * 100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found a valid decomposition for this rank, try lower
    
        if best_rank < 100:
            break  # Found something better than schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
