"""Matrix multiplication tensor decomposition for 3x5x6.

Strategy: Use a structured low-rank ansatz (sum of Kronecker-product blocks
plus a dense residual) and solve for the coefficients via least squares,
then polish with a few Newton steps.  We search over small block-rank
decompositions that can beat the schoolbook rank of 90.

The tensor T[i,j,k] = e_{i,k} (i.e. T[i,j,k]=1 iff i==k) for 3x5x6 matrix
mult.  We seek rank r such that there exist U (15 x r), V (30 x r), W (18 x r)
with T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _tensor_entry(i, j, k, n=3, m=5, p=6):
    """T[i,j,k] = 1 if i==k else 0 (matrix mult tensor)."""
    return 1.0 if i == k else 0.0


def _schoolbook():
    """Fallback: schoolbook rank n*m*p = 90."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _validate(U, V, W, tol=1e-9):
    """Check that U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] sums to T[i,j,k]."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape != (n * m, r) or V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    # For each (i,j,k), check the tensor entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                target = 1.0 if i == k else 0.0
                if abs(s - target) > tol:
                    return False
    return True


def _validate_fast(U, V, W, tol=1e-9):
    """Vectorized validation using einsum."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Build the full tensor and check
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Use einsum: U_idx = (n,m,r), V_idx = (m,p,r), W_idx = (p,n,r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is tricky with einsum; do it in steps
    # First: A[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c]  -> (n,m,p)
    A = np.einsum('ijc,jkc->ijk', U3, V3)
    # Then: T[i,j,k] = sum_c A[i,j,c] ... no, W depends on c too.
    # Actually T[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
    # = sum_c (U3[i,j,c]*V3[j,k,c])*W3[k,i,c]
    # Let B[i,j,k,c] = U3[i,j,c]*V3[j,k,c]  -> (n,m,p,r)
    B = U3[:, :, :, None] * V3[None, :, :, :]  # (n, m, p, r)
    T = np.einsum('ijkc,kic->ijk', B, W3)
    # Check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0 if i == k else 0.0
                if abs(T[i, j, k] - target) > tol:
                    return False
    return True


def _block_ansatz(rank):
    """
    Structured ansatz: partition the rank into blocks corresponding to
    Kronecker products of smaller matrix multiplications.
    
    For 3x5x6, try a decomposition based on:
    - Block 1: rank-3 block using 3x3x3 structure (for i==k coupling)
    - Block 2: rank-25 for the m=5 middle dimension
    - Block 3: rank-18 for the p=6 dimension
    
    Actually, let's use a more direct approach:
    T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    Key insight: T[i,j,k] = delta(i,k). So the tensor is the identity in (i,k)
    with a "free" j index. This means:
    T[i,j,k] = delta(i,k)
    
    So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta(i,k)
    
    This is a system where for fixed (i,k), the sum over j and c must give delta(i,k).
    
    Let's try: U[i*m+j,c] depends on i,j and c; V[j*p+k,c] on j,k,c; W[k*n+i,c] on k,i,c.
    
    A natural low-rank structure:
    For each pair (i,k) with i==k, we need the sum to be 1, otherwise 0.
    
    Let's try rank = n * m + n * p + m * p = 15 + 18 + 30 = 63? Or smaller.
    
    Actually, the "Landsberg" bound gives rank >= n*m*p / (n+m+p-2) roughly.
    For 3x5x6: 90/(3+5+6-2) = 90/12 = 7.5, so lower bound ~8.
    But that's not tight. The actual bound for matrix mult is at least max(n*m, m*p, n*p).
    
    Let me try a direct optimization approach.
    """
    pass


def _try_rank(r, max_iter=20000, tol=1e-10, time_budget=None):
    """
    Try to find a rank-r decomposition using nonlinear least squares.
    
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is a bilinear (trilinear) problem. Use Gauss-Newton / Levenberg-Marquardt
    style iterations: fix two factors, solve for the third, cycle.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Initialize with random factors
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    # Target tensor entries
    # For each (i,j,k), target = 1 if i==k else 0
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        
        # Update U: for each (i,j) row, solve least squares
        # T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        # For fixed (i,j), define A[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then T[i,j,:] = U[i*m+j,:] @ A[i,j,:].T ... 
        # Actually: T[i,j,k] = sum_c U[i*m+j,c] * A[i,j,k,c]
        # So for fixed (i,j): T[i,j,:] = U[i*m+j,:] @ A[i,j,:,:].T  (A is (p, r))
        for i in range(n):
            for j in range(m):
                # A[k,c] = V[j*p+k,c] * W[k*n+i,c]  shape (p, r)
                A = V[j * p:(j + 1) * p, :] * W[np.arange(p) * n + i, :]  # (p, r)
                # Target: T[i,j,k] for k=0..p-1
                target = np.zeros(p)
                for k in range(p):
                    target[k] = 1.0 if i == k else 0.0
                # Solve: U[i*m+j,c] @ A[k,c] = target[k]  for all k
                # i.e., find u (r,) such that u @ A.T ≈ target  (A is p x r, so A.T is r x p)
                # u @ A.T is (p,), target is (p,)
                # So: A @ u = target  (p x r) @ (r,) = (p,)
                try:
                    u_new, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                except np.linalg.LinAlgError:
                    u_new = U[i * m + j, :]
                U[i * m + j, :] = u_new
        
        # Update V: for each (j,k) row
        # T[i,j,k] = sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c]
        # For fixed (j,k): B[i,c] = U[i*m+j,c] * W[k*n+i,c]  shape (n, r)
        # Target: T[i,j,k] for i=0..n-1
        for j in range(m):
            for k in range(p):
                B = U[np.arange(n) * m + j, :] * W[k * n + np.arange(n), :]  # (n, r)
                target = np.zeros(n)
                for i in range(n):
                    target[i] = 1.0 if i == k else 0.0
                try:
                    v_new, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                except np.linalg.LinAlgError:
                    v_new = V[j * p + k, :]
                V[j * p + k, :] = v_new
        
        # Update W: for each (k,i) row
        # T[i,j,k] = sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]
        # For fixed (k,i): C[j,c] = U[i*m+j,c] * V[j*p+k,c]  shape (m, r)
        # Target: T[i,j,k] for j=0..m-1
        for k in range(p):
            for i in range(n):
                C = U[np.arange(m) + i * m, :] * V[np.arange(m) * p + k, :]  # (m, r)
                target = np.zeros(m)
                for j in range(m):
                    target[j] = 1.0 if i == k else 0.0
                try:
                    w_new, _, _, _ = np.linalg.lstsq(C, target, rcond=None)
                except np.linalg.LinAlgError:
                    w_new = W[k * n + i, :]
                W[k * n + i, :] = w_new
        
        # Check residual periodically
        if it % 5 == 0 or it == max_iter - 1:
            if _validate_fast(U, V, W, tol=1e-6):
                # Verify with tighter tolerance
                if _validate_fast(U, V, W, tol=1e-10):
                    return {"U": U, "V": V, "W": W}
            
            # Compute residual norm
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            B = U3[:, :, :, None] * V3[None, :, :, :]
            T = np.einsum('ijkc,kic->ijk', B, W3)
            target_T = np.zeros((n, m, p))
            for i in range(n):
                for k in range(p):
                    if i == k:
                        target_T[i, :, k] = 1.0
            resid = np.linalg.norm(T - target_T)
            if resid < best_residual:
                best_residual = resid
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return None


def _try_structured_decomposition():
    """
    Try a structured decomposition exploiting T[i,j,k] = delta(i,k).
    
    Key observation: T[i,j,k] = delta(i,k) means the tensor is separable as:
    T[i,j,k] = sum_s delta(i,s) * delta(s,k) = sum_s M_i_s * N_s_k * P_j
    
    But P_j must be constant in j for this to work, which gives rank >= n (for the i-k part)
    but the j dimension adds complexity.
    
    Actually: T[i,j,k] = delta(i,k) doesn't depend on j at all!
    So T[i,j,k] = delta(i,k) * 1_j where 1_j = 1 for all j.
    
    This means: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta(i,k)
    
    For this to hold for all j, the j-dependence in U and V must cancel out.
    
    One approach: Let U[i*m+j,c] = a[i,c] (independent of j) and V[j*p+k,c] = b[j,c]*d[k,c]... 
    This is getting complicated. Let's just try the iterative solver.
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    
    # First try the schoolbook as a safe baseline
    result = _schoolbook()
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: at least max(n, m, p) = 6, but more realistically
    # for matrix mult, rank >= n*m, m*p, n*p are not required...
    # Actually the tensor rank of matrix multiplication for n x m by m x p is
    # at least max(n*m, m*p, n*p) / something... Let's just try various ranks.
    
    # Try ranks from a reasonable minimum up to schoolbook
    # Lower bound: the tensor has "border rank" at least n*m*p/(n+m+p-2) ~ 7.5
    # But exact rank is likely higher. Let's try from 20 up.
    
    ranks_to_try = list(range(20, 91, 2))  # Even ranks from 20 to 90
    # Also try some specific values
    for r in [15, 16, 18, 20, 24, 25, 30, 36, 40, 45, 50, 54, 60, 63, 70, 75, 80, 84, 88, 90]:
        if r in ranks_to_try:
            continue
        ranks_to_try.append(r)
    ranks_to_try = sorted(set(ranks_to_try))
    
    best_result = result  # schoolbook
    best_rank = n * m * p
    
    for r in ranks_to_try:
        if time.time() - global_start > TIME_LIMIT - 5:
            break
        
        time_budget = TIME_LIMIT - 5 - (time.time() - global_start)
        if time_budget < 2:
            break
        
        # Try multiple random seeds for this rank
        found = False
        for seed in range(3):
            if time.time() - global_start > TIME_LIMIT - 5:
                break
            
            old_seed = np.random.get_state()
            np.random.seed(1000 + seed * 100 + r)
            
            res = _try_rank(r, max_iter=5000, tol=1e-10, time_budget=time_budget)
            
            np.random.set_state(old_seed)
            
            if res is not None:
                # Verify
                U, V, W = res["U"], res["V"], res["W"]
                if _validate_fast(U, V, W, tol=1e-10):
                    found = True
                    if r < best_rank:
                        best_rank = r
                        best_result = res
                    break
        
        if found:
            # Found a valid decomposition at this rank, continue to find lower
            continue
    
    # Polish: if we found a solution, try to reduce rank further with more iterations
    # But for now, return the best found
    return best_result
