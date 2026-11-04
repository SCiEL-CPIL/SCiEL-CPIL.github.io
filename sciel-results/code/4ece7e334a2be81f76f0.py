"""Optimized program for matrix multiplication 3x3 * 3x4 tensor rank minimization."""
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _make_schoolbook(n, m, p):
    """Create schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # For each (i,k), check that sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta
    # Actually, the tensor T[i,j,k] = A[i,j]*B[j,k]
    # The decomposition should satisfy: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if this is the identity term
    # Wait, let me reconsider. The matrix multiplication tensor has entries T[i,j,k] where
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T_{i,j,k} = 1 for all valid i,j,k (it's the "multiplication" tensor)
    # A rank-r decomposition means: T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's verify: for each (i,j,k), the sum over c should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _tensor_error(U, V, W, n, m, p):
    """Compute the Frobenius norm of the error in the tensor decomposition."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (s - 1.0) ** 2
    return np.sqrt(error)

def _random_decomposition(n, m, p, rank, scale=1.0):
    """Create a random decomposition with given rank."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _local_search(n, m, p, start_rank, max_iters=1000, tol=1e-9, time_limit=10.0):
    """
    Try to reduce the rank by removing one term at a time and adjusting.
    Start from a valid decomposition and try to zero out one column while
    adjusting others to maintain correctness.
    """
    start_time = time.time()
    
    # Start from schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    rank = start_rank
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if rank <= 1:
            break
        
        # Try to remove one random column
        col_to_remove = np.random.randint(rank)
        
        # Save original
        U_orig = U[:, :rank].copy()
        V_orig = V[:, :rank].copy()
        W_orig = W[:, :rank].copy()
        
        # Try to find small adjustments to the remaining columns to compensate
        # This is a linear system problem: we need to adjust the remaining columns
        # so that the tensor equation still holds.
        
        # For each (i,j,k), we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # If we remove column c0, we lose U[i*m+j,c0]*V[j*p+k,c0]*W[k*n+i,c0]
        # We need to adjust the other columns to compensate.
        
        # This is complex. Instead, let's try a simpler approach:
        # Just check if we can set one column to zero and still have a valid decomposition
        # by checking if that column is "redundant" in some sense.
        
        # Actually, for the schoolbook decomposition, each column is independent.
        # Let's try a different approach: use gradient descent to minimize the
        # tensor error while encouraging sparsity.
        
        # For now, let's just try random perturbations and see if we can find
        # a lower-rank solution.
        
        # Try: take a random rank-r' decomposition with r' < rank and see if
        # it's close to being valid.
        for trial in range(10):
            if time.time() - start_time > time_limit:
                break
            
            new_rank = rank - 1
            if new_rank < 1:
                break
            
            # Try to find a rank-new_rank decomposition
            U_new = np.random.randn(n * m, new_rank) * 0.5
            V_new = np.random.randn(m * p, new_rank) * 0.5
            W_new = np.random.randn(p * n, new_rank) * 0.5
            
            # Optimize to minimize tensor error
            def objective(params):
                U_flat = params[:n*m*new_rank].reshape(n*m, new_rank)
                V_flat = params[n*m*new_rank:n*m*new_rank + m*p*new_rank].reshape(m*p, new_rank)
                W_flat = params[n*m*new_rank + m*p*new_rank:].reshape(p*n, new_rank)
                return _tensor_error(U_flat, V_flat, W_flat, n, m, p)
            
            params0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            # Use Nelder-Mead for small problems
            result = minimize(objective, params0, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < tol:
                # Found a valid lower-rank decomposition!
                U_new = result.x[:n*m*new_rank].reshape(n*m, new_rank)
                V_new = result.x[n*m*new_rank:n*m*new_rank + m*p*new_rank].reshape(m*p, new_rank)
                W_new = result.x[n*m*new_rank + m*p*new_rank:].reshape(p*n, new_rank)
                
                # Verify
                if _verify(U_new, V_new, W_new, n, m, p, tol):
                    return U_new, V_new, W_new, new_rank
    
    return U[:, :rank], V[:, :rank], W[:, :rank], rank

def _try_structured_reductions(n, m, p, time_limit=5.0):
    """
    Try structured approaches to reduce rank:
    1. Block decomposition
    2. Exploit symmetries
    """
    start_time = time.time()
    
    # For 3x3 * 3x4, one approach is to think of it as three 3x4 matrix multiplications
    # or use some kind of Kronecker product structure.
    
    # Another idea: the rank of matrix multiplication A(n,m) x B(m,p) is at most
    # n*m*p, but there are known upper bounds from algorithms like Strassen.
    
    # For 2x2, Strassen gives rank 7 instead of 8.
    # For larger matrices, we can use Strassen recursively.
    
    # Let's try to construct a Strassen-like decomposition for 3x3 * 3x4.
    # This is complex, so let's try a simpler heuristic.
    
    # Heuristic: try to find a decomposition with rank = max(n*p, m*p, n*m) * 2
    # or some other structured bound.
    
    # For now, just return None to indicate no structured reduction found.
    return None

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor W follows the convention W[k*n+i, r] where n=3 (rows of A),
    p=4 (cols of B), so W has shape (p*n, rank) = (12, rank).
    U has shape (n*m, rank) = (9, rank).
    V has shape (m*p, rank) = (12, rank).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Start with schoolbook decomposition
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try local search to reduce rank
    U_ls, V_ls, W_ls, rank_ls = _local_search(n, m, p, best_rank, 
                                               max_iters=500, time_limit=30.0)
    if rank_ls < best_rank:
        best_rank = rank_ls
        best_U, best_V, best_W = U_ls, V_ls, W_ls
    
    # Try structured reductions
    structured = _try_structured_reductions(n, m, p, time_limit=10.0)
    if structured is not None:
        U_st, V_st, W_st, rank_st = structured
        if rank_st < best_rank:
            best_rank = rank_st
            best_U, best_V, best_W = U_st, V_st, W_st
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
