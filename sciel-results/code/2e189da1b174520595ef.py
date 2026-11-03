"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor: T[i,j,k] = 1 if the output index is (i,k)
    # T is of shape (n*m, m*p, p*n) in a flattened sense.
    # Actually, the tensor T is indexed by (input_A, input_B, output_C)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    # We check a sample of entries to be fast, but for correctness we need all.
    # For efficiency, we can use tensor contraction.
    # Let's build the tensor T and compare with the reconstruction.
    
    # T has shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Reconstruct: R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We can do this efficiently with einsum or loops.
    # R = np.einsum('ar,br,cr->abc', U, V, W)
    # But this might be slow for large rank.
    # Let's use a loop over r for stability and speed for small n, m, p.
    R = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check if R equals T
    # We only care about the non-zero entries of T, which are the diagonal-like structure.
    # But to be safe, we check the whole tensor or just the relevant entries.
    # The relevant entries are those where the indices correspond to valid (i,j,k).
    # Let's check the max absolute difference.
    diff = np.max(np.abs(R - T))
    return diff < tol

def _rank_255_solution(n, m, p):
    """
    Construct a rank 28 solution for 2x5x5 matrix multiplication.
    This is based on the known result that rank(2x5x5) = 28.
    We use a combination of Strassen-like blocks and direct terms.
    Since deriving this from scratch is complex, we use a known construction
    if possible, or fall back to a higher rank.
    
    For now, we will implement a search-based approach for the 2x5x6 case directly,
    or use a known lower rank if we can verify it.
    
    Actually, let's try to construct a rank 30 solution for 2x5x6 by extending
    a rank 28 solution for 2x5x5 with 2 additional terms for the 6th column/row.
    But we don't have the 2x5x5 rank 28 matrices hardcoded.
    
    Let's try a different strategy: use a numerical search for rank 30 or 32.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try to find a low-rank decomposition using numerical optimization.
    # We'll use a simple gradient descent on the rank-30 hypothesis.
    # If it fails, we fall back to a known higher rank.
    
    # Target rank: 30 (lower bound for 2x5x6)
    target_rank = 30
    
    # Initialize U, V, W with random values
    # We use a random initialization scaled appropriately.
    np.random.seed(42)
    
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U_shape = (n * m, target_rank)
    V_shape = (m * p, target_rank)
    W_shape = (p * n, target_rank)
    
    # We'll try a few different initializations and steps.
    best_solution = None
    best_error = float('inf')
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Gradient descent loop
    # We minimize ||R - T||_F^2 where R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a non-convex optimization. We use a simple SGD-like approach.
    
    # For speed, we'll use a batch approach where we compute the full error and gradient.
    # But this might be slow. Let's try a simpler approach: random restarts with local search.
    
    # Alternative: use the known fact that rank(2x5x6) >= 30 and try to find rank 30.
    # If we can't find it in time, we return a schoolbook solution of rank 60.
    
    # Let's try a few random initializations and see if we can reduce the error to 0.
    num_restarts = 100
    max_iters = 500
    lr = 0.01
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(*U_shape) * 0.1
        V = np.random.randn(*V_shape) * 0.1
        W = np.random.randn(*W_shape) * 0.1
        
        # Scale to match the target magnitude
        # The target tensor has entries of 1.
        # We want sum_r U[a,r] V[b,r] W[c,r] ~ 1 for the right indices.
        # Let's normalize after each step.
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute R
            R = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for r in range(target_rank):
                R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.sum((R - T) ** 2)
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-6:
                break
            
            # Compute gradients
            # dR/du = v*w, etc.
            # dU[a,r] = sum_{b,c} (R[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # This is expensive. Let's use a simpler update.
            # We'll use a stochastic gradient with a few random samples.
            
            # Sample a few (a,b,c) entries
            num_samples = 100
            indices_a = np.random.randint(0, n * m, num_samples)
            indices_b = np.random.randint(0, m * p, num_samples)
            indices_c = np.random.randint(0, p * n, num_samples)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for s in range(num_samples):
                a, b, c = indices_a[s], indices_b[s], indices_c[s]
                err = R[a, b, c] - T[a, b, c]
                for r in range(target_rank):
                    grad_U[a, r] += err * V[b, r] * W[c, r]
                    grad_V[b, r] += err * U[a, r] * W[c, r]
                    grad_W[c, r] += err * U[a, r] * V[b, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent divergence
            # Keep the scale consistent
            norm_U = np.linalg.norm(U)
            norm_V = np.linalg.norm(V)
            norm_W = np.linalg.norm(W)
            if norm_U > 0 and norm_V > 0 and norm_W > 0:
                scale = 1.0 / (norm_U * norm_V * norm_W) ** (1.0 / 3.0)
                U *= scale
                V *= scale
                W *= scale
        
        # After local search, check if we have a solution
        if error < 1e-4:
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-6):
                # Found a valid solution
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no low-rank solution found, return schoolbook rank 60
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
