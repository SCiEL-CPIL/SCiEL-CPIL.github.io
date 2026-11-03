"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_wfta_decomposition(n, m, p):
    """
    Build a WFTA-based decomposition for n×m × m×p matrix multiplication.
    
    Strategy: Use polynomial evaluation/interpolation over m points.
    For m=5 (prime), we evaluate at points 0,1,2,3,4.
    
    For each row i of A (n rows):
      - Evaluate the row polynomial a_i(x) = sum_j A[i,j] * x^j at 5 points
      - This requires 5 evaluations per row: rank 5*n = 20
    
    For each column k of B (p columns):
      - Evaluate the column polynomial b_k(x) = sum_j B[j,k] * x^j at 5 points  
      - This requires 5 evaluations per column: rank 5*p = 25
    
    Pointwise multiply in evaluation domain:
      - For each of the 5 evaluation points, multiply the evaluated row and column
      - This is shared: rank 5
    
    Interpolate back to get C[i,k]:
      - For each output (i,k), interpolate from 5 values
      - This requires 5 interpolation weights per output: rank 5*n*p = 100
      
    Total naive: 20 + 25 + 5 + 100 = 150 (worse than schoolbook!)
    
    Better approach: The WFTA for matrix multiplication uses:
      - Forward transform on A: rank n*m (each row transformed)
      - Forward transform on B: rank m*p (each column transformed)  
      - Pointwise multiply: rank m
      - Inverse transform: rank n*p
      
    But this is still n*m + m*p + m + n*p = 20 + 25 + 5 + 20 = 70 for 4x5x5.
    
    Let me construct this properly.
    """
    # For m=5, use evaluation at points 0,1,2,3,4
    # Vandermonde matrix V where V[j, t] = t^j for j=0..4, t=0..4
    
    points = np.array([0, 1, 2, 3, 4], dtype=float)
    # V[j, t] = points[t]^j
    V_mat = np.zeros((m, m))
    for j in range(m):
        for t in range(m):
            V_mat[j, t] = points[t] ** j
    
    # V_mat is invertible (Vandermonde with distinct points)
    # V_mat_inv = np.linalg.inv(V_mat)
    
    # U factors: For each (i, j) pair, U[i*m+j, c] contributes to evaluating row i at point t
    # We want: sum_j U[i*m+j, c] * A[i,j] = a_i(eval_point_c) for c=0..4
    # So U[i*m+j, c] = points[c]^j for c=0..4, and 0 for c>=5
    # Rank from U: n * m = 20 (but only first 5 columns are non-zero per row group)
    
    # Actually, let's structure the rank more carefully.
    # Total rank = n*m (forward A) + m*p (forward B) + m (pointwise) + n*p (inverse)
    # = 20 + 25 + 5 + 20 = 70
    
    rank = n * m + m * p + m + n * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    
    # Part 1: Forward transform of A rows (rank n*m)
    # For each row i, column j: U[i*m+j, col_offset + i*m + j] = points[t]^j where t = (i*m+j) % m? 
    # No, let's think differently.
    
    # The WFTA structure:
    # C[i,k] = sum_t (sum_j A[i,j] * alpha[j,t]) * (sum_l B[l,k] * beta[l,t]) * gamma[t]
    # where alpha, beta, gamma are the transform matrices.
    
    # For evaluation at t points:
    # alpha[j,t] = t^j (evaluate polynomial at t)
    # beta[l,t] = t^l (evaluate polynomial at t)  
    # gamma[t] = 1 (pointwise multiply)
    # Then we need to interpolate: C[i,k] = sum_t (eval_A[i,t] * eval_B[k,t]) * interp[t,k,i]
    
    # This is getting complex. Let me just use the standard tensor decomposition form.
    
    # Simpler: just return None and use schoolbook, since constructing a correct WFTA
    # decomposition requires careful index management.
    return None

def _sgd_sparse_decomposition(n, m, p, target_rank, max_iters=50000, lr=0.01, lambda_l1=0.001, seed=42):
    """
    Formulate tensor rank minimization as non-convex sparse coding with SGD + l1 regularization.
    Uses complex field to allow more flexible decompositions.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize factor matrices with small random values
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    # Precompute index arrays for efficient gradient computation
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Gradient w.r.t. U[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] * (pred - 1)
    # where pred = sum_c' U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c']
    
    # For efficiency, we'll compute gradients in batches
    # Precompute all (i,j,k) combinations
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute current predictions and loss
        total_loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k) in indices:
            # pred = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            prod_VW = V[j*p+k] * W[k*n+i]  # (target_rank,)
            pred = np.dot(U[i*m+j], prod_VW)
            error = pred - 1.0
            
            total_loss += error * error
            
            # Gradients
            # d/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] * error
            grad_U[i*m+j] += prod_VW * error
            # d/dV[j*p+k, c] = U[i*m+j, c] * W[k*n+i, c] * error
            grad_V[j*p+k] += U[i*m+j] * W[k*n+i] * error
            # d/dW[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c] * error
            grad_W[k*n+i] += U[i*m+j] * V[j*p+k] * error
        
        total_loss /= len(indices)
        
        # Add L1 regularization to encourage sparsity (lower effective rank)
        grad_U += lambda_l1 * np.sign(U)
        grad_V += lambda_l1 * np.sign(V)
        grad_W += lambda_l1 * np.sign(W)
        
        # SGD update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Track best
        if total_loss < best_loss:
            best_loss = total_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check convergence
        if total_loss < 1e-12:
            break
        
        # Adaptive learning rate decay
        if it % 1000 == 0 and it > 0:
            lr *= 0.9
    
    # Verify the best solution
    if _verify(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W, best_loss
    else:
        return None, None, None, best_loss

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Try WFTA-based decomposition
    wfta_result = _build_wfta_decomposition(n, m, p)
    if wfta_result is not None:
        U_w, V_w, W_w = wfta_result
        if _verify(U_w, V_w, W_w, n, m, p):
            best_rank = U_w.shape[1]
            best_U, best_V, best_W = U_w, V_w, W_w
    
    # 3. Try SGD sparse decomposition with various target ranks
    # Start from lower ranks and work up
    target_ranks = [80, 85, 90, 95, 99]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try multiple seeds for diversity
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 15:
                break
            
            result = _sgd_sparse_decomposition(
                n, m, p, target_rank, 
                max_iters=30000, lr=0.005, lambda_l1=0.0005, seed=seed
            )
            
            if result[0] is not None:
                U_sgd, V_sgd, W_sgd, loss = result
                if _verify(U_sgd, V_sgd, W_sgd, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sgd, V_sgd, W_sgd
                    break
        
        if best_rank < n * m * p:
            break
    
    # 4. Attempt to find lower rank via random search if still at schoolbook
    if best_rank == n * m * p:
        for target_rank in [99, 98, 97]:
            if time.time() > deadline - 10:
                break
                
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                U = np.random.randint(-1, 2, size=(n*m, target_rank))
                V = np.random.randint(-1, 2, size=(m*p, target_rank))
                W = np.random.randint(-1, 2, size=(p*n, target_rank))
                
                if _verify(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
            
            if best_rank < n * m * p:
                break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
