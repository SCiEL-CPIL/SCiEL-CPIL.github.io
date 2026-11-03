"""Program for rank-minimal exact tensor decomposition of 3x3x5 matrix multiplication."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog

def _validate_decomposition_fast(U, V, W, n, m, p, rank):
    """Fast validation using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta_{j,j'} * delta_{k,k'} for the multiplication tensor
    # Actually, for C = A*B: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where the bilinear form gives C[i,k]
    
    # Reshape for efficient computation
    U_r = U.reshape(n, m, rank)  # U_r[i,j,r]
    V_r = V.reshape(m, p, rank)  # V_r[j,k,r]
    W_r = W.reshape(p, n, rank)  # W_r[k,i,r]
    
    # For each output (i,k), check: sum_j sum_r U_r[i,j,r]*V_r[j,k,r]*W_r[k,i,r] = sum_j delta_{j,j}*... 
    # Actually the tensor T should satisfy:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right. Let me think again.
    
    # For matrix multiplication C = A @ B where A is n×m, B is m×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # In the decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    #
    # This equals sum_r sum_j sum_j' W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] * A[i,j] * B[j',k]
    #
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's still not quite right. Let me use the standard formulation.
    
    # The correct condition is that the tensor T_{i,j,k}^{i',j',k'} defined by:
    # T = sum_r U_r[i,j] * V_r[j',k'] * W_r[k,i] (with appropriate index mapping)
    # should equal the matrix multiplication tensor.
    
    # Let me just verify by checking the bilinear identity directly for all basis elements.
    # For the decomposition to be valid, for all i,i',j,j',k,k':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # if we interpret the tensor correctly. But this seems to have too many deltas.
    
    # Actually, I think the correct interpretation is:
    # The matrix multiplication tensor T has components T[i,j,k] = 1 if this is a valid term
    # No, that's not right either.
    
    # Let me use the standard: for C = A*B, the bilinear form is:
    # <C, E_{i,k}> = sum_j <A, E_{i,j}> * <B, E_{j,k}>
    # 
    # In the decomposition, we write:
    # <C, E_{i,k}> = sum_r W[k*n+i, r] * (<A, u_r> * <B, v_r>)
    # where u_r = sum_j U[i*m+j, r] * E_{i,j} (but this doesn't make sense as u_r depends on i)
    
    # I think the correct formulation is:
    # u_r = sum_{i,j} U[i*m+j, r] * E_{i,j}
    # v_r = sum_{j,k} V[j*p+k, r] * E_{j,k}  
    # w_r = sum_{i,k} W[k*n+i, r] * E_{i,k}
    # And we need: sum_r <A, u_r> * <B, v_r> * w_r = A @ B
    # i.e., sum_r (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k]) * W[k*n+i, r] = C[i,k]
    
    # Let me just test with random matrices for efficiency
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            a_r = np.tensordot(A, U[:, r], axes=([0,1],[0,1]))
            b_r = np.tensordot(B, V[:, r], axes=([0,1],[0,1]))
            C_decomposed += W[:, r] * a_r * b_r
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    
    return True

def _validate_decomposition(U, V, W, n, m, p, rank):
    """Validate that the decomposition exactly computes matrix multiplication."""
    return _validate_decomposition_fast(U, V, W, n, m, p, rank)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _local_polish(U, V, W, n, m, p, max_iters=100, tol=1e-12):
    """Try to reduce rank by local optimization."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Try to zero out entries by adjusting others
    # This is a complex optimization, so let's use a simple approach:
    # Try to find linear dependencies and eliminate columns
    
    for _ in range(max_iters):
        # Check for linearly dependent columns
        # If column r is a linear combination of others, we can eliminate it
        # This requires solving a system, which is expensive
        
        # Simpler: try random perturbations and see if we can reduce effective rank
        improved = False
        for r in range(rank):
            # Try to set one entry to zero
            U_copy = U.copy()
            V_copy = V.copy()
            W_copy = W.copy()
            
            # Find a non-zero entry and try to zero it
            non_zero = np.argwhere(U_copy[r] != 0)
            if len(non_zero) == 0:
                continue
            
            row_idx = non_zero[0, 0]
            U_copy[row_idx, r] = 0
            
            # Check if still valid
            if _validate_decomposition_fast(U_copy, V_copy, W_copy, n, m, p, rank):
                U, V, W = U_copy, V_copy, W_copy
                improved = True
                break
        
        if not improved:
            break
    
    return U.tolist(), V.tolist(), W.tolist()

def _structured_search(n, m, p, target_rank, max_iterations=500, deadline=None):
    """Structured search using known patterns for small matrix multiplication."""
    best_rank = n * m * p
    best_decomp = None
    
    if deadline is not None and time.time() > deadline:
        return best_rank, best_decomp
    
    # For 3x3x5, try various structured patterns
    # Known: 2x2x2 has rank 7 (Strassen), 3x3x3 has rank 23 (Landsberg)
    # 3x3x5 might have rank around 30-40
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Generate random decomposition with structure
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for r in range(target_rank):
            # Each rank-1 component should have sparse structure
            # Choose random subsets of indices
            u_indices = random.sample(range(n*m), random.randint(1, min(3, n*m)))
            v_indices = random.sample(range(m*p), random.randint(1, min(3, m*p)))
            w_indices = random.sample(range(p*n), random.randint(1, min(3, p*n)))
            
            for idx in u_indices:
                U[idx, r] = random.choice([-1, 1])
            for idx in v_indices:
                V[idx, r] = random.choice([-1, 1])
            for idx in w_indices:
                W[idx, r] = random.choice([-1, 1])
        
        if _validate_decomposition_fast(U, V, W, n, m, p, target_rank):
            best_rank = target_rank
            best_decomp = (U.tolist(), V.tolist(), W.tolist(), target_rank)
            break
    
    return best_rank, best_decomp

def _try_random_rank_reduction(n, m, p, target_rank, max_iterations=1000, deadline=None):
    """Try to find a decomposition with target_rank using random search."""
    best_rank = n * m * p
    best_decomp = None
    
    if deadline is not None and time.time() > deadline:
        return best_rank, best_decomp
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Randomly initialize with small values
        U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
        W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
        
        # Check if this is a valid decomposition
        if _validate_decomposition_fast(U, V, W, n, m, p, target_rank):
            best_rank = target_rank
            best_decomp = (U.tolist(), V.tolist(), W.tolist(), target_rank)
            break
    
    return best_rank, best_decomp

def _greedy_rank_reduction(n, m, p, start_rank, max_iterations=500, deadline=None):
    """Try to reduce rank by removing columns one at a time."""
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook_decomposition(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_rank = current_rank
    best_decomp = (U.tolist(), V.tolist(), W.tolist(), current_rank)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        if current_rank <= best_rank:
            break
        
        # Try removing each column
        found_reduction = False
        for col_to_remove in range(current_rank):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Create new matrices without this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            # Check if still valid
            if _validate_decomposition_fast(U_new, V_new, W_new, n, m, p, current_rank - 1):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                best_rank = current_rank
                best_decomp = (U.tolist(), V.tolist(), W.tolist(), current_rank)
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    return best_rank, best_decomp

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 @ 3x5 matrix multiplication.
    Uses schoolbook as baseline and tries to reduce rank.
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook_decomposition(n, m, p)
    best_rank = current_rank
    best_decomp = (U, V, W, current_rank)
    
    # Try greedy rank reduction
    if time.time() < deadline - 10:
        rank, decomp = _greedy_rank_reduction(n, m, p, current_rank, max_iterations=200, deadline=deadline)
        if rank < best_rank:
            best_rank = rank
            best_decomp = decomp
    
    # Try structured search for lower ranks
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, max(best_rank - 15, 1), -1):
            if time.time() > deadline - 10:
                break
            rank, decomp = _structured_search(n, m, p, target_rank, max_iterations=200, deadline=deadline)
            if rank < best_rank:
                best_rank = rank
                best_decomp = decomp
            if best_rank <= 20:
                break
    
    # Try random search for lower ranks
    if time.time() < deadline - 5:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 1), -1):
            if time.time() > deadline - 5:
                break
            rank, decomp = _try_random_rank_reduction(n, m, p, target_rank, max_iterations=100, deadline=deadline)
            if rank < best_rank:
                best_rank = rank
                best_decomp = decomp
            if best_rank <= n * m * p // 2:
                break
    
    # Return the best decomposition found
    U, V, W, rank = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
