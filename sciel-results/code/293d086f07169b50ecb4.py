"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    # Efficient tensor verification using einsum
    # We want sum_r W_arr[r, k*n+i] * U_arr[r, a*m+b] * V_arr[r, c*p+d]
    # == 1 if a==i and b==c and d==k else 0
    
    # Reshape to (rank, n, m) for U, (rank, m, p) for V, (rank, p, n) for W
    U_reshaped = U_arr.reshape(rank, n, m)
    V_reshaped = V_arr.reshape(rank, m, p)
    W_reshaped = W_arr.reshape(rank, p, n)
    
    # Compute the tensor T[a, b, c, d, i, k] = sum_r U[r, a, b] * V[r, b, c] * W[r, d, i]
    # Wait, the indices in the prompt are:
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # The bilinear form is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_{a,b} U[a*m+b, r] * A[a, b]) * (sum_{c,d} V[c*p+d, r] * B[c, d])
    # We require this to be sum_j A[i, j] * B[j, k].
    # The coefficient of A[a, b] * B[c, d] in C[i, k] is sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r].
    # This must equal delta_{a, i} * delta_{b, c} * delta_{d, k}.
    
    # Let's construct the full tensor T of shape (n, m, m, p, n, p)
    # T[a, b, c, d, i, k] = sum_r U[r, a, b] * V[r, c, d] * W[r, k, i]
    # Wait, V is (m, p) so V[r, c, d] where c is row (0..m-1) and d is col (0..p-1).
    # But in the term V[c*p+d, r], c corresponds to the row of B, which is the column of A.
    # So c is in 0..m-1? No, B is m x p. So rows of B are 0..m-1.
    # The index in V is c*p+d. So c is 0..m-1, d is 0..p-1.
    # The index in U is a*m+b. a is 0..n-1, b is 0..m-1.
    # The index in W is k*n+i. k is 0..p-1, i is 0..n-1.
    
    # So:
    # U_arr is (rank, n, m)
    # V_arr is (rank, m, p)
    # W_arr is (rank, p, n)
    
    # We want T[a, b, c, d, i, k] = sum_r U[r, a, b] * V[r, c, d] * W[r, k, i]
    # Note: The prompt says V[j*p+k] where j is col of A (0..m-1) and k is row of B (0..m-1).
    # So V index is row_of_B * p + col_of_B.
    # So V[r, c, d] where c is row_of_B, d is col_of_B.
    # And U index is row_of_A * m + col_of_A.
    # So U[r, a, b] where a is row_of_A, b is col_of_A.
    # And W index is row_of_C * n + col_of_C.
    # So W[r, k, i] where k is row_of_C, i is col_of_C.
    
    # The condition is:
    # T[a, b, c, d, i, k] = 1 if a==i and b==c and d==k else 0.
    
    # Let's compute this tensor.
    # T = np.einsum('rab,rcd,rki->abcdik', U_reshaped, V_reshaped, W_reshaped)
    # This might be large: 2*5*5*5*2*5 = 2500 elements.
    # rank is up to 50.
    # This is fast.
    
    T = np.einsum('rab,rcd,rki->abcdik', U_reshaped, V_reshaped, W_reshaped)
    
    # Construct target tensor
    # Target[a, b, c, d, i, k] = 1 if a==i and b==c and d==k else 0
    
    # We can check this efficiently.
    # For each (a, b, c, d, i, k), check if T[a, b, c, d, i, k] is close to expected.
    
    # Let's iterate over the non-zero entries of the target tensor.
    # The target tensor has 1s at (a, b, b, k, a, k) for a in 0..n-1, b in 0..m-1, k in 0..p-1.
    # There are n*m*p such entries.
    # All other entries should be 0.
    
    # Check non-zero entries
    for a in range(n):
        for b in range(m):
            for k in range(p):
                # c must be b, d must be k, i must be a
                if abs(T[a, b, b, k, a, k] - 1.0) > tol:
                    return False
    
    # Check zero entries
    # We can sample or check all.
    # Total entries: 2*5*5*5*2*5 = 2500.
    # This is very fast.
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            if abs(T[a, b, c, d, i, k] - expected) > tol:
                                return False
    
    return True

def _make_schoolbook(n, m, p):
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

def _try_rank_reduction(n, m, p, target_rank, time_budget=30.0):
    """Try to find a decomposition of the given target_rank.
    
    Uses a structured approach: start from schoolbook and try to merge terms
    by exploiting linear dependencies.
    """
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < time_budget
    
    if target_rank >= n * m * p:
        return None
    
    # Try random initialization and gradient descent
    np.random.seed(42)
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # Build the target tensor
    # T[a, b, c, d, i, k] = 1 if a==i and b==c and d==k else 0
    # We can represent this as a matrix M of shape (n*m*m*p, n*p) ?
    # Or just use the einsum approach for loss calculation.
    
    # For gradient descent, we need to compute the loss and gradients efficiently.
    # Loss = sum_{a,b,c,d,i,k} (T_current[a,b,c,d,i,k] - T_target[a,b,c,d,i,k])^2
    
    # Let's precompute the target tensor as a flat array or a 6D array.
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for k in range(p):
                T_target[a, b, b, k, a, k] = 1.0
    
    def compute_tensor(U, V, W):
        U_reshaped = U.T.reshape(target_rank, n, m)
        V_reshaped = V.T.reshape(target_rank, m, p)
        W_reshaped = W.T.reshape(target_rank, p, n)
        return np.einsum('rab,rcd,rki->abcdik', U_reshaped, V_reshaped, W_reshaped)
    
    # Gradient descent
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(5000):
        if not still_going():
            break
        
        # Compute loss
        T_current = compute_tensor(U, V, W)
        diff = T_current - T_target
        current_loss = np.sum(diff**2)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Compute gradients
        U_reshaped = U.T.reshape(target_rank, n, m)
        V_reshaped = V.T.reshape(target_rank, m, p)
        W_reshaped = W.T.reshape(target_rank, p, n)
        
        # grad_U[r, a, b] = 2 * sum_{c,d,i,k} diff[a,b,c,d,i,k] * V[r,c,d] * W[r,k,i]
        # grad_V[r, c, d] = 2 * sum_{a,b,i,k} diff[a,b,c,d,i,k] * U[r,a,b] * W[r,k,i]
        # grad_W[r, k, i] = 2 * sum_{a,b,c,d} diff[a,b,c,d,i,k] * U[r,a,b] * V[r,c,d]
        
        grad_U = np.einsum('abcdik,rcd,rki->rab', diff, V_reshaped, W_reshaped)
        grad_V = np.einsum('abcdik,rab,rki->rcd', diff, U_reshaped, W_reshaped)
        grad_W = np.einsum('abcdik,rab,rcd->rki', diff, U_reshaped, V_reshaped)
        
        grad_U *= 2
        grad_V *= 2
        grad_W *= 2
        
        # Reshape gradients back to (n*m, target_rank)
        grad_U = grad_U.reshape(target_rank, n*m).T
        grad_V = grad_V.reshape(target_rank, m*p).T
        grad_W = grad_W.reshape(target_rank, p*n).T
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        lr *= 0.9999
    
    if best_loss < 1e-10:
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to reduce rank
    # Note: For 2x5x5, rank 50 is schoolbook. 
    # Trying to go below 50 is very hard. 
    # We attempt a few ranks but fall back to schoolbook if not found quickly.
    for rank in range(best_rank - 1, max(best_rank - 3, 40), -1):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 15:
            break
        
        budget = min(remaining * 0.5, 10.0) # Tighter budget to ensure fallback
        
        result = _try_rank_reduction(n, m, p, rank, time_budget=budget)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
