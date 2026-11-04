"""Search for a lower-rank exact tensor decomposition of 3x5 * 5x7 multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
# Tensor shape: (n, m, p) with T[i,j,k] = 1 if i,j,k correspond to the multiplication term
# Actually the matrix multiplication tensor M[i,j,k] = 1 (scalar) for each (i,j,k)
# The rank-r decomposition is: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
# Wait, the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
# So M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

# We need to find rank r and matrices U, V, W such that for all i in [0,n), j in [0,m), k in [0,p):
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

# This is a system of nonlinear equations. We can use random initialization and gradient descent,
# or we can try to find structured solutions.

# Known results: For 2x2 matrix multiplication, rank is 7 (Strassen). For larger matrices,
# the rank is generally unknown. For 3x5 * 5x7, the schoolbook rank is 3*5*7 = 105.
# There might be lower ranks, but finding them exactly is hard.

# Let's try a random search with gradient descent for small ranks.

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_schoolbook
    
    # Try to find lower ranks
    # For each rank r from some minimum up to rank_schoolbook - 1, try random initialization and optimization
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks in descending order from rank_schoolbook - 1 down to min_rank
    # But this could be slow. Let's try a few random attempts for each rank.
    
    for r in range(rank_schoolbook - 1, min_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(seed=42 + r * 1000 + attempt)
            
            # Random initialization
            U = rng.standard_normal((n * m, r)) * 0.1
            V = rng.standard_normal((m * p, r)) * 0.1
            W = rng.standard_normal((p * n, r)) * 0.1
            
            # Define the loss function: sum of (M[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            def loss_and_grad(U, V, W):
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute the sum
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = val - 1.0
                            loss += err * err
                            
                            # Gradient with respect to U[i*m+j, c]
                            # d/dU[i*m+j,c] (err^2) = 2*err * V[j*p+k,c] * W[k*n+i,c]
                            for c in range(r):
                                g = 2 * err * V[j * p + k, c] * W[k * n + i, c]
                                grad_U[i * m + j, c] += g
                                grad_V[j * p + k, c] += 2 * err * U[i * m + j, c] * W[k * n + i, c]
                                grad_W[k * n + i, c] += 2 * err * U[i * m + j, c] * V[j * p + k, c]
                
                return loss, grad_U, grad_V, grad_W
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                
                loss, gU, gV, gW = loss_and_grad(U, V, W)
                
                if loss < 1e-10:
                    # Found a solution!
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = r
                    return _to_dict(U, V, W)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
            
            # Check if we got close
            loss, _, _, _ = loss_and_grad(U, V, W)
            if loss < 1e-8:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
    
    return _to_dict(best_U, best_V, best_W)


def _to_dict(U, V, W):
    """Convert numpy arrays to lists of lists for the required format."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
