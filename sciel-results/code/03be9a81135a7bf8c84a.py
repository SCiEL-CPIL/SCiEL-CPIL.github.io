"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the multiplication tensor."""
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check against the identity: T[i,j,k] should be 1 iff (i,j,k) is a valid triple (always is)
    # Actually the multiplication tensor is T[i,j,k] = 1 for all i,j,k (it's the identity map in a sense)
    # Wait - the matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor has T[i,j,k]=1 for all valid indices.
    # Let's verify: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    # For each (i,j,k): sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, 1.0, atol=tol)


def _try_random_rank(n, m, p, rank, time_budget=10.0, seed=None):
    """Try to find a decomposition of given rank using random initialization and gradient descent."""
    start = time.time()
    rng = np.random.RandomState(seed if seed is not None else random.randint(0, 2**31))
    
    # Initialize U, V, W randomly
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    # This is a non-convex problem. Use a simple iterative approach.
    # Actually, let's try a different approach: use the fact that we need
    # the tensor to be all-ones.
    
    # Let's use a simpler approach: try to find rank-r decomposition by
    # solving a least-squares problem iteratively.
    
    # Reshape for efficient computation
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Target tensor is all ones
    T_target = np.ones((n, m, p))
    
    # Iterative approach: fix two, solve for the third
    for iteration in range(500):
        if time.time() - start > time_budget:
            break
        
        # Update U: for each (i,j), U[i,j,:] should satisfy
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
        # This is: U[i,j,:] * (V[j,k,:] * W[k,i,:] summed over r) = 1
        # Actually it's: for each k, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # Let M[k,r] = V[j,k,r] * W[k,i,r], then sum_r U[i,j,r] * M[k,r] = 1 for all k
        # So U[i,j,:] * M^T = ones(p), i.e., M @ U[i,j,:]^T = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build M of shape (p, rank): M[k,r] = V[j,k,r] * W[k,i,r]
                M = V_3d[j, :, :] * W_3d[:, i, :]  # (p, rank)
                # Solve M @ U[i,j,:]^T = ones(p) in least squares
                try:
                    U_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = U_new
                except:
                    pass
                U_3d[i, j, :] = U[i * m + j, :]
        
        # Update V
        for j in range(m):
            for k in range(p):
                # For each i: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # M[i,r] = U[i,j,r] * W[k,i,r], solve M @ V[j,k,:]^T = ones(n)
                M = U_3d[:, j, :] * W_3d[k, :, :]  # (n, rank)
                try:
                    V_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = V_new
                except:
                    pass
                V_3d[j, k, :] = V[j * p + k, :]
        
        # Update W
        for k in range(p):
            for i in range(n):
                # For each j: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # M[j,r] = U[i,j,r] * V[j,k,r], solve M @ W[k,i,:]^T = ones(m)
                M = U_3d[i, :, :] * V_3d[:, k, :]  # (m, rank)
                try:
                    W_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = W_new
                except:
                    pass
                W_3d[k, i, :] = W[k * n + i, :]
        
        # Check convergence
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.max(np.abs(T - 1.0))
        if err < 1e-10:
            break
    
    # Final check
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if np.max(np.abs(T - 1.0)) < 1e-9:
        # Convert to list of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return U_list, V_list, W_list, rank
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 144
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Also rank >= n*p = 36 (from Strassen-type bounds)
    # Let's try ranks from a reasonable lower bound upward
    
    # Try a few random initializations for various ranks
    for rank in range(36, min(best_rank, 80)):
        if time.time() > deadline - 5.0:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            
            time_budget = min(5.0, deadline - time.time() - 2.0)
            if time_budget < 1.0:
                break
            
            result = _try_random_rank(n, m, p, rank, time_budget=time_budget, seed=rank * 100 + attempt)
            if result is not None:
                U, V, W, r = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break  # Found valid decomposition at this rank, try lower
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best_result
    
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
