"""Rewritten program: search for a low-rank exact decomposition of the 3x3x7 matrix multiplication tensor."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i, j, k] = 1 if A[i,j]*B[j,k] contributes to C[i,k].
    Actually, the multiplication tensor has T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1.
    """
    # T is a 3D array of shape (n, m, p) with all ones
    return np.ones((n, m, p), dtype=np.float64)

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    rank = U.shape[1]
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1 for all i, j, k
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster check using matrix operations."""
    rank = U.shape[1]
    # For each r, we have outer products. The sum is:
    # sum_r U_r ⊗ V_r ⊗ W_r where U_r is (n*m), V_r is (m*p), W_r is (p*n)
    # We need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Reshape approach: for each j, consider the (n x p) matrix M_j[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This is tricky. Let's just do the direct check but vectorized over r.
    
    # Build a 3D array of shape (n, m, p)
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r]  # (n*m,)
        v = V[:, r]  # (m*p,)
        w = W[:, r]  # (p*n,)
        # u[i*m+j], v[j*p+k], w[k*n+i]
        # For fixed j, the contribution to M_j[i,k] is u[i*m+j] * v[j*p+k] * w[k*n+i]
        for j in range(m):
            uj = u[j::m]  # this doesn't work directly since u is indexed by i*m+j
            # u[i*m+j] for i in range(n)
            u_slice = np.array([u[i * m + j] for i in range(n)])
            # v[j*p+k] for k in range(p)
            v_slice = np.array([v[j * p + k] for k in range(p)])
            # w[k*n+i] for i in range(n), k in range(p)
            w_mat = np.array([[w[k * n + i] for i in range(n)] for k in range(p)])
            # M_j[i,k] += u_slice[i] * v_slice[k] * w_mat[k,i]
            contrib = np.outer(u_slice, v_slice) * w_mat.T  # w_mat.T[i,k] = w_mat[k,i]
            result[i, j, k] += contrib[i, k]  # wait, this indexing is wrong
    
    # Let me redo this more carefully
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r]  # index: i*m + j
        v = V[:, r]  # index: j*p + k
        w = W[:, r]  # index: k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    if not np.allclose(result, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _check_decomposition_vectorized(U, V, W, n, m, p, tol=1e-9):
    """Vectorized check."""
    rank = U.shape[1]
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)  # u[i,j] = U[i*m+j, r]
        v = V[:, r].reshape(m, p)  # v[j,k] = V[j*p+k, r]
        w = W[:, r].reshape(p, n)  # w[k,i] = W[k*n+i, r]
        # For each j: result[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for j in range(m):
            # u[:,j] is (n,), v[j,:] is (p,)
            # w[:,j]? No, w[k,i] so for fixed j... wait w doesn't have j index
            # result[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # = sum over i,k: u[i,j] * v[j,k] * w[k,i]
            # This is like: for fixed j, outer product of u[:,j] and v[j,:] scaled by w
            u_col = u[:, j]  # (n,)
            v_row = v[j, :]  # (p,)
            # contrib[i,k] = u_col[i] * v_row[k] * w[k,i]
            contrib = np.outer(u_col, v_row) * w.T  # w.T[i,k] = w[k,i]
            result[:, j, :] += contrib
    if not np.allclose(result, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = np.array(mat, dtype=np.float64)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            nearest = np.round(val * 2.0) / 2.0
            if abs(val - nearest) < tol:
                result[i, j] = nearest
    return result

def _try_rank_search(n, m, p, start_rank=20, max_rank=63, time_limit=135.0):
    """Search for the minimal rank decomposition."""
    deadline = time.time() + time_limit
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Try to find a decomposition for each rank from low to high
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Use a simple gradient-free approach: iterative refinement
            # The constraint is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # We can use a fixed-point / alternating minimization approach
            # Or use the fact that this is a tensor decomposition problem
            
            # Let's try a simple approach: initialize and then try to satisfy constraints
            # Using Newton-like method or just check if random works (unlikely for low rank)
            
            # Alternative: use the structure. For matrix multiplication, there are known
            # decompositions. For 3x3x3, the rank is at most 23 (Strassen-like).
            # For 3x3x7, we can think of it as 7 independent 3x3 multiplications sharing structure.
            
            # Let's try a more structured approach:
            # The tensor T[i,j,k] = 1 for all i,j,k.
            # A decomposition of rank r means: T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r ∈ R^(n*m), v_r ∈ R^(m*p), w_r ∈ R^(p*n)
            
            # For the "all ones" tensor, one trivial decomposition is rank 1:
            # u_r = ones(n*m), v_r = ones(m*p), w_r = ones(p*n)
            # Then sum_r u[i*m+j]*v[j*p+k]*w[k*n+i] = 1*1*1 = 1. YES!
            
            # Wait, that's rank 1! Let me verify:
            # U[i*m+j, 0] = 1 for all i,j
            # V[j*p+k, 0] = 1 for all j,k  
            # W[k*n+i, 0] = 1 for all k,i
            # Then sum_0 U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
            
            # So the rank is 1!
            break  # Found rank 1
        
        if rank == 1:
            best_rank = 1
            break
    
    # The rank-1 decomposition is trivially valid
    if best_rank == 1:
        U = np.ones((n * m, 1))
        V = np.ones((m * p, 1))
        W = np.ones((p * n, 1))
        best_decomp = (U, V, W)
    
    return best_rank, best_decomp

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for multiplying A (n x m) by B (m x p) to get C (n x p)
    # has the property that T[i, j, k] = 1 for all valid i, j, k.
    # This is because A[i,j] * B[j,k] contributes to C[i,k] with coefficient 1.
    
    # A rank-1 decomposition exists:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # This gives sum_0 U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k.
    
    # Verify:
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Double-check correctness
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(1):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to schoolbook
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
    else:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
