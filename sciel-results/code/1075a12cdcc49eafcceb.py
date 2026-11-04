import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :])
                expected = 1.0 if (i == k) else 0.0
                # W[k*n+i, :] should be 1 for this term
                # Actually the tensor equation is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k)
                # Let me re-check...
                # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
                # So M[i,j,k] = 1 if the computation contributes, which is always 1
                # But we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all valid (i,j,k)
                # Wait, that's not right either. Let me think again.
                # 
                # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor M has M[i,j,k] = 1 for all i,j,k (since every term contributes)
                # But actually no - the tensor encodes which products are needed.
                # M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
                # That's always true, so M[i,j,k] = 1 for all i,j,k.
                # 
                # The decomposition requires:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k] = 1
                pass
    
    # Let me do it properly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast check using vectorized operations."""
    rank = U.shape[1]
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank by zeroing out columns
    and re-solving for the remaining entries.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, current_rank
    
    # We want to find a subset of columns of size target_rank that can
    # still represent the tensor. We'll try greedy removal.
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # Try to remove one column at a time
    removed = 0
    while removed < (current_rank - target_rank):
        if time.time() - start_time > deadline - 5:
            break
        
        found_removal = False
        # Try each column to see if we can remove it
        for col in range(current_rank - removed):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try removing column col
            # After removal, we have target_rank - (removed+1) columns left
            # We need to check if the remaining columns can be adjusted to still work
            
            # Simple approach: set this column to zero and see if the tensor is still satisfied
            # by the other columns. If not, try to adjust the other columns.
            
            # For now, just try zeroing and checking
            U_test = U_curr.copy()
            V_test = V_curr.copy()
            W_test = W_curr.copy()
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            
            # Check if still valid
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = np.dot(U_test[i*m + j, :], V_test[j*p + k, :] * W_test[k*n + i, :])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
            
            if valid:
                # Successfully removed this column
                # Remove it from the matrices
                mask = np.ones(current_rank - removed, dtype=bool)
                mask[col] = False
                U_curr = U_curr[:, mask]
                V_curr = V_curr[:, mask]
                W_curr = W_curr[:, mask]
                removed += 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return U_curr, V_curr, W_curr, U_curr.shape[1]

def _iterative_optimization(U, V, W, n, m, p, deadline):
    """
    Iteratively try to reduce rank by:
    1. Trying to zero out columns
    2. Re-solving for entries using least squares
    """
    start_time = time.time()
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    current_rank = U_curr.shape[1]
    
    # Strategy: iteratively try to reduce rank by solving for one factor
    # while fixing the others, then check validity
    
    for iteration in range(50):
        if time.time() - start_time > deadline - 10:
            break
        
        improved = False
        
        # Try to reduce rank by 1
        target_rank = current_rank - 1
        if target_rank < n * p:  # Minimum possible rank
            break
        
        # Try different approaches to reduce rank
        
        # Approach: randomly zero out some entries and re-solve
        for attempt in range(20):
            if time.time() - start_time > deadline - 10:
                break
            
            U_test = U_curr.copy()
            V_test = V_curr.copy()
            W_test = W_curr.copy()
            
            # Randomly set some entries to zero to encourage sparsity
            for _ in range(5):
                mat = np.random.choice([0, 1, 2])
                if mat == 0:
                    idx = np.random.randint(0, U_test.shape[1])
                    U_test[:, idx] *= 0.5
                elif mat == 1:
                    idx = np.random.randint(0, V_test.shape[1])
                    V_test[:, idx] *= 0.5
                else:
                    idx = np.random.randint(0, W_test.shape[1])
                    W_test[:, idx] *= 0.5
            
            # Check if still valid (it won't be, but we use this as starting point)
            # Then try to fix by adjusting one factor
            
            # Fix U and W, solve for V
            # For each (j,k), we need: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
            # This is a system of equations
            
            # Actually, let's try a different approach:
            # Use the fact that if we have a valid decomposition, we can try
            # to find linear dependencies among the columns
            
            # For now, just try the greedy removal approach
            U_new, V_new, W_new, new_rank = _try_reduce_rank(
                U_curr, V_curr, W_curr, n, m, p, target_rank, deadline
            )
            
            if new_rank < current_rank:
                U_curr = U_new
                V_curr = V_new
                W_curr = W_new
                current_rank = new_rank
                improved = True
                break
        
        if not improved:
            break
    
    return U_curr, V_curr, W_curr, current_rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition as baseline
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook(n, m, p)
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb
    
    # Check if we're within deadline
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve using iterative optimization
    U_opt, V_opt, W_opt, opt_rank = _iterative_optimization(
        U_sb, V_sb, W_sb, n, m, p, DEADLINE - 10
    )
    
    if opt_rank < best_rank:
        best_U = U_opt
        best_V = V_opt
        best_W = W_opt
        best_rank = opt_rank
    
    # Verify the result
    if time.time() - start_time < DEADLINE - 5:
        valid = _check_decomposition_fast(best_U, best_V, best_W, n, m, p)
        if not valid:
            # Fall back to schoolbook
            best_U = U_sb
            best_V = V_sb
            best_W = W_sb
            best_rank = rank_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
