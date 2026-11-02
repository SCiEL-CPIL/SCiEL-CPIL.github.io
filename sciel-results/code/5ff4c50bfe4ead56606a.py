"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] where:
    T[i,j,k] = 1 if this corresponds to a valid product term.
    For multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor has entries T[i, j, k] = 1 for all valid (i,j,k).
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                # sum over c of u_val[c] * v_val[c] * w_val[c]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _try_rank_reduction(n, m, p, max_rank, time_budget=100.0):
    """Try to find a decomposition of rank <= max_rank using iterative search.
    Strategy: Start with schoolbook and try to merge/eliminate terms.
    """
    T = _build_tensor(n, m, p)
    
    # Try various approaches to find lower rank
    # Approach: Use the fact that for small matrices, we can sometimes
    # exploit symmetries and linear combinations.
    
    # For 2x3x6, let's try a structured search
    # The key constraint: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's try rank values from max_rank down to 1
    best_U, best_V, best_W = None, None, None
    
    # Try a greedy/iterative approach
    # For each candidate rank r, try to solve the system
    # This is a bilinear/trilinear system, so we use iterative methods
    
    for r in range(max_rank, 1, -1):
        if time.time() - START_TIME > time_budget:
            break
        # Try to find a rank-r decomposition
        # Use random initialization and gradient-free search
        success = _search_rank_r(n, m, p, r, T)
        if success:
            U, V, W = success
            best_U, best_V, best_W = U, V, W
            break  # Found one, try to go lower
    
    return best_U, best_V, best_W

def _search_rank_r(n, m, p, r, T, max_iters=5000):
    """Try to find a rank-r decomposition using iterative optimization."""
    np.random.seed(42)
    
    # Initialize with random values
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > 105.0:
            break
        
        # Compute current tensor estimate
        # T_est[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    loss += (s - T[i,j,k])**2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            return best_U, best_V, best_W
        
        # Simple coordinate descent: update one variable at a time
        # Update U
        for idx in range(n * m):
            i = idx // m
            j = idx % m
            for c in range(r):
                # d/dU[idx,c] of loss = 2 * sum_{i',j',k'} (T_est[i',j',k'] - T[i',j',k']) * V[j'*p+k',c] * W[k'*n+i',c] * delta(i',i)*delta(j',j)
                # For fixed i,j: d/dU[i*m+j,c] = 2 * sum_k (T_est[i,j,k] - T[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
                grad = 0.0
                for k in range(p):
                    s = 0.0
                    for cc in range(r):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    grad += (s - T[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
                U[idx, c] -= 0.01 * grad
        
        # Update V
        for idx in range(m * p):
            j = idx // p
            k = idx % p
            for c in range(r):
                grad = 0.0
                for i in range(n):
                    s = 0.0
                    for cc in range(r):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    grad += (s - T[i,j,k]) * U[i*m+j, c] * W[k*n+i, c]
                V[idx, c] -= 0.01 * grad
        
        # Update W
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            for c in range(r):
                grad = 0.0
                for j in range(m):
                    s = 0.0
                    for cc in range(r):
                        s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    grad += (s - T[i,j,k]) * U[i*m+j, c] * V[j*p+k, c]
                W[idx, c] -= 0.01 * grad
        
        if it % 100 == 0 and best_loss < 1e-6:
            # Check if we have an exact solution
            ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
            if ok:
                return best_U, best_V, best_W
    
    # Final check
    if best_loss < 1e-12:
        ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
        if ok:
            return best_U, best_V, best_W
    
    return None

def _try_structured_decomposition(n, m, p):
    """Try known structured low-rank decompositions.
    For 2x3 x 3x6, we can try to exploit the structure.
    """
    # The matmul C = A*B where A is 2x3, B is 3x6
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # ... etc.
    
    # One approach: group terms by the middle index j
    # For each j, we have A[:,j] * B[j,:] which is an outer product
    # This gives rank m = 3 if we could do outer products, but that's not how it works.
    
    # Actually, the standard lower bound is n*m*p, but for specific small cases
    # there might be better. Let's try rank 18, 24, 30, etc.
    
    # Try a systematic approach: for rank r, set up the equations
    # and try to solve using linear algebra tricks.
    
    # For now, let's try to use the fact that we can combine terms.
    # If we have two terms that differ only in one index, we might combine them.
    
    # Let's try rank = n*p*m / 2 = 18 (optimistic)
    # Or use the Strassen-like idea for small blocks.
    
    # Actually, let me try a different approach: use the tensor structure directly.
    # The tensor T has n*m*p = 36 ones. We need to write each as a sum of products.
    
    # Let's try to find a rank-30 decomposition by eliminating 6 terms.
    # This requires finding linear dependencies.
    
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # Search from rank 35 down to some minimum
    # Use time budget carefully
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try structured approaches first (fast)
    if time.time() - START_TIME < 5.0:
        result = _try_structured_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    # Try iterative search for lower ranks
    # Start from best_rank - 1 and go down
    for r in range(best_rank - 1, max(10, best_rank - 10), -1):
        if time.time() - START_TIME > 100.0:
            break
        result = _search_rank_r(n, m, p, r, T, max_iters=2000)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break  # Found lower rank, try even lower
    
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
