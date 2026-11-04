"""Tensor decomposition for 3x4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_decomposition(n, m, p, rank, n_trials=500, seed=42):
    """
    Try to find a rank-`rank` decomposition via random initialization + iterative refinement.
    The constraint is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    
    We use a multiplicative-update style approach:
    - Initialize U, V, W randomly
    - Iteratively update each factor to reduce the error
    """
    rng = np.random.RandomState(seed)
    
    for trial in range(n_trials):
        # Random initialization with small values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Iterative refinement: alternate fixing two factors and solving for the third
        # For fixed V, W: we need U[i*m+j, c] such that sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
        # This is a linear system in U for each (i,j,k), but it's coupled across c.
        
        # Alternative: use gradient descent on the squared error
        best_err = float('inf')
        
        for iteration in range(200):
            # Compute current error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-12:
                return U, V, W
            
            if err < best_err:
                best_err = err
            
            # Gradient descent step
            lr = 0.01
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        grad = 2.0 * (val - 1.0)
                        dU[i*m+j, :] += grad * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += grad * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += grad * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodically renormalize to prevent drift
            if iteration % 20 == 0:
                for c in range(rank):
                    norm_U = np.linalg.norm(U[:, c])
                    norm_V = np.linalg.norm(V[:, c])
                    norm_W = np.linalg.norm(W[:, c])
                    total = norm_U * norm_V * norm_W
                    if total > 1e-15:
                        scale = (1.0 / total) ** (1.0/3.0)
                        U[:, c] *= scale
                        V[:, c] *= scale
                        W[:, c] *= scale
        
        # Check final result
        if best_err < 1e-10:
            return U, V, W
        
    return None

def _try_structured_decomposition(n, m, p, rank):
    """
    Try a structured decomposition based on the known structure of matrix multiplication.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: use the fact that we can group terms.
    For example, if we fix j, we can compute the contribution from that j.
    
    A more sophisticated approach uses the Strassen-like idea of combining terms.
    """
    # For 3x4x6, let's try a block-based approach
    # Split the middle dimension m=4 into parts
    
    # Actually, let's try a different structured approach:
    # Use the fact that the tensor T[i,j,k] = 1 for all i,j,k
    # We can try to find a low-rank factorization using algebraic methods
    
    # One known technique: use the "laser method" or "subrank" approach
    # For now, let's try a simple structured decomposition
    
    # Try: U[i*m+j, c] = a[i,j,c], V[j*p+k, c] = b[j,k,c], W[k*n+i, c] = w[k,i,c]
    # where a, b, w are chosen such that the sum works out
    
    # For rank r, we need r "equations" that together satisfy all n*m*p = 72 constraints
    
    # Let's try a Fourier-based approach or a combinatorial design
    # For small cases, we can try all possible rank values and use SVD-based methods
    
    return None

def _try_svd_based(n, m, p, target_rank):
    """
    Try to find a decomposition using SVD-based methods.
    
    The tensor T[i,j,k] = 1 for all i,j,k.
    We want to find U, V, W of rank r such that the trilinear form equals 1.
    
    One approach: 
    1. Create the full tensor T (n x m x p) with all ones
    2. Try to find a low-rank approximation
    3. Use the Tucker decomposition or CP decomposition
    """
    # Create the tensor
    T = np.ones((n, m, p))
    
    # Try CP decomposition using ALS (Alternating Least Squares)
    rng = np.random.RandomState(12345)
    
    for trial in range(100):
        # Initialize
        U_mat = rng.randn(n * m, target_rank) * 0.1
        V_mat = rng.randn(m * p, target_rank) * 0.1
        W_mat = rng.randn(p * n, target_rank) * 0.1
        
        for iteration in range(500):
            # Update U: for each (i,j), solve least squares
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:] for all k) = 1
            # i.e., U[i*m+j,:] @ M_{i,j} = 1_vector where M_{i,j}[k,c] = V[j*p+k,c]*W[k*n+i,c]
            
            for i in range(n):
                for j in range(m):
                    # Build the matrix M of shape (p, target_rank)
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j,:] @ M = ones(p)
                    # This is: (M.T @ U[i*m+j,:].T) = ones(p)
                    # i.e., M.T @ x = 1 where x = U[i*m+j,:].T
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U_mat[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n * m, target_rank))  # This doesn't work directly
                    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # i.e., V[j*p+k,:] @ N_{j,k} = 1 where N_{j,k}[i,c] = U[i*m+j,c]*W[k*n+i,c]
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V_mat[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    # i.e., W[k*n+i,:] @ P_{k,i} = 1 where P_{k,i}[j,c] = U[i*m+j,c]*V[j*p+k,c]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W_mat[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-10:
                return U_mat, V_mat, W_mat
        
    return None

def _try_direct_solve(n, m, p, rank):
    """
    Try to directly solve for a decomposition of the given rank.
    This is a system of n*m*p equations in (n*m + m*p + p*n) * rank unknowns.
    
    For 3x4x6 with rank r: 72 equations, (12 + 24 + 18) * r = 54r unknowns.
    For r >= 2, we have more unknowns than equations, so solutions may exist.
    
    We use a nonlinear solver approach.
    """
    from scipy.optimize import least_squares
    
    def _residuals(params):
        """Compute residuals for the tensor decomposition."""
        # Unpack parameters
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        
        return residuals
    
    # Initial guess: random
    rng = np.random.RandomState(999)
    x0 = rng.randn(n * m * rank + m * p * rank + p * n * rank) * 0.1
    
    try:
        result = least_squares(_residuals, x0, method='lm', max_nfev=10000)
        if result.cost < 1e-12:
            size_U = n * m * rank
            size_V = m * p * rank
            size_W = p * n * rank
            
            U_flat = result.x[:size_U]
            V_flat = result.x[size_U:size_U + size_V]
            W_flat = result.x[size_U + size_V:size_U + size_V + size_W]
            
            U = U_flat.reshape(n * m, rank)
            V = V_flat.reshape(m * p, rank)
            W = W_flat.reshape(p * n, rank)
            
            return U, V, W
    except:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Known lower bound for matrix multiplication rank
    # For n x m x p, the rank is at least max(n*m, m*p, p*n)
    # For 3x4x6: max(12, 24, 18) = 24
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower rank decompositions
    # Start from the lower bound and go up
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try SVD/ALS-based approach
        result = _try_svd_based(n, m, p, rank)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                best_rank = rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                break
        
        if time.time() > deadline:
            break
        
        # Try random gradient descent
        result = _try_random_decomposition(n, m, p, rank, n_trials=50, seed=rank*1000)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                best_rank = rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                break
    
    return best_result

# EVOLVE-BLOCK-END
