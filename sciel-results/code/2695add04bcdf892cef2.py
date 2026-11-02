"""Rewritten program: attempts to find a lower-rank exact tensor decomposition
for 2x5 * 5x6 matrix multiplication, falling back to the schoolbook (rank 60)."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check that the decomposition is exact for the matrix multiplication tensor.
    
    The tensor T[i,j,k] = delta for output position... 
    Actually, the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    So the tensor is: T_{i,j,k} = 1 if we're computing (i,k) using (j)
    More precisely, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    and the decomposition requires: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    Wait, let me re-read the convention.
    
    "output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k*n+i, r) where the row index k*n+i corresponds to output (k,i)
    which is the (i,k) entry of the product... wait, or is it (k,i)?
    
    Actually for A (n x m) * B (m x p) = C (n x p):
    C[i,k] = sum_j A[i,j] B[j,k]
    
    U has shape (n*m, rank), row indexed by (i*m+j) -> input A entry
    V has shape (m*p, rank), row indexed by (j*p+k) -> input B entry  
    W has shape (p*n, rank), row indexed by (k*n+i) -> output C entry
    
    So the decomposition is: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    (since each output element C[i,k] gets contribution 1 from each (i,j,k) term)
    
    Actually no. The tensor T is defined such that:
    T[i,j,k] = 1 for all (i,j,k) in range(n) x range(m) x range(p)
    
    And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This means each "slice" of the tensor at fixed (i,j,k) must equal 1.
    """
    n, m, p = n, m, p
    # Reshape U, V, W to 3D
    U3 = U.reshape(n, m, -1)  # (n, m, rank)
    V3 = V.reshape(m, p, -1)  # (m, p, rank)
    W3 = W.reshape(p, n, -1)  # (p, n, rank)
    
    # For each (i,j,k), compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U3[i, j, :], V3[j, k, :]), W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_random_search(n, m, p, max_rank, time_budget, rng):
    """Try to find a decomposition of rank <= max_rank using random initialization
    and iterative refinement."""
    deadline = time.time() + time_budget
    best = None
    best_rank = n * m * p  # schoolbook
    
    # Try various ranks starting from lower values
    # For 2x5x6, known lower bounds suggest rank might be around 40-55
    for target_rank in range(max(10, n * m * p - 20), max_rank + 1):
        if time.time() > deadline:
            break
        
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
            
            # Iteratively refine: for each (i,j,k), the constraint is
            # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # This is a system of n*m*p equations in n*m*R + m*p*R + p*n*R unknowns
            # We can use a simple iterative method:
            
            # Reshape for easier access
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            
            # Simple coordinate descent / alternating updates
            # For fixed V and W, solve for U: 
            #   U[i,j,:] * (V[j,k,:] * W[k,i,:]) = 1 for each k
            # This is a linear system in U[i,j,:]
            
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k, we need: U[i,j,:] dot (V[j,k,:] * W[k,i,:]) = 1
                        # This gives p equations in R unknowns for U[i,j,:]
                        # Solve least squares
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V3[j, k, :] * W3[k, i, :]
                        # Solve A_mat @ u = ones(p)
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U3[i, j, :] = u_sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U3[i, j, :] * W3[k, i, :]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V3[j, k, :] = v_sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U3[i, j, :] * V3[j, k, :]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W3[k, i, :] = w_sol
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(np.dot(U3[i, j, :], V3[j, k, :]), W3[k, i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged and target_rank < best_rank:
                best = (U.copy(), V.copy(), W.copy())
                best_rank = target_rank
                # We could break early if we find a very good rank
                if best_rank <= 35:
                    break
        
        if best_rank <= 35:
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Keep 20s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try random search for lower rank
    if time.time() < deadline - 10:
        rng = random.Random(42)
        np_rng = np.random.default_rng(42)
        
        # Try to find rank lower than schoolbook
        # Start with modest target
        result, rank = _try_random_search(n, m, p, max_rank=55, 
                                           time_budget=min(90, deadline - time.time()),
                                           rng=np_rng)
        if result is not None and rank < best_rank:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p, tol=1e-5):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank
    
    # Ensure we have a valid result
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-5):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
