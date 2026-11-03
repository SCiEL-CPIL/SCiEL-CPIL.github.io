import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 150.0  # wall-clock limit
MARGIN = 5.0      # safety margin in seconds

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err <= tol

def _schoolbook(n, m, p):
    """Fallback: trivial rank n*m*p decomposition."""
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

def _snap_to_halfint(W, tol=1e-8):
    """Try to snap W entries to nearest half-integer if close enough."""
    W = np.array(W, dtype=float)
    snapped = np.round(W * 2) / 2.0
    if np.max(np.abs(W - snapped)) < tol:
        return snapped
    return None

def _als_optimize(U, V, W, n, m, p, max_iters=200, time_limit=100.0):
    """
    Alternating Least Squares for tensor decomposition.
    Minimizes ||T - U⊗V⊗W||_F where T is the matmul tensor.
    """
    start = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Update W: for each (k,i), minimize sum_j (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # For fixed (k,i): it's a linear system in W[k*n+i,:]
        # Let a_{j,r} = U[i*m+j,r]*V[j*p+k,r], b_j = T[i,j,k]
        # Minimize sum_j (b_j - sum_r a_{j,r}*w_r)^2 => w = (A^T A)^{-1} A^T b
        for k in range(p):
            for i in range(n):
                # Build A: (m, R) where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])  # (m, R)
                b = np.array([1.0] * m)  # T[i,j,k] = 1 for all i,j,k in matmul
                # Solve (A^T A) w = A^T b
                AT = A.T  # (R, m)
                ATA = AT @ A  # (R, R)
                ATb = AT @ b  # (R,)
                try:
                    W[k * n + i, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    W[k * n + i, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Update U: for each (i,j), minimize sum_{k,r} (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # For fixed (i,j): linear in U[i*m+j,:]
        for i in range(n):
            for j in range(m):
                # A[r, r2] = sum_k V[j*p+k, r] * W[k*n+i, r2]
                # b[r2] = sum_k V[j*p+k, r2] * W[k*n+i, r2] * T[i,j,k] = sum_k V[j*p+k, r2]*W[k*n+i, r2]
                # Actually: minimize sum_k (1 - sum_r U_{i,j,r} * (V[j*p+k,r]*W[k*n+i,r]))^2
                # Let c_r = sum_k V[j*p+k,r]*W[k*n+i,r]  ... no, this is wrong.
                # Let d_r = sum_k V[j*p+k,r] * W[k*n+i,r] ... no.
                # Expand: sum_k (1 - sum_r U_r * V_{j,k,r} * W_{k,i,r})^2
                # = sum_k (1 - sum_r U_r * a_{k,r})^2 where a_{k,r} = V[j*p+k,r]*W[k*n+i,r]
                # Gradient w.r.t U_r: -2 sum_k a_{k,r} (1 - sum_{r'} U_{r'} a_{k,r'})
                # = -2 (sum_k a_{k,r}) + 2 sum_k a_{k,r} sum_{r'} U_{r'} a_{k,r'}
                # Set to 0: sum_k a_{k,r} a_{k,r'} U_{r'} = sum_k a_{k,r}
                # So (A^T A) U = A^T 1 where A[k, r] = V[j*p+k,r]*W[k*n+i,r]
                A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])  # (p, R)
                b = np.ones(p)
                AT = A.T  # (R, p)
                ATA = AT @ A  # (R, R)
                ATb = AT @ b  # (R,)
                try:
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    U[i * m + j, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Update V: for each (j,k), minimize sum_{i,r} (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # For fixed (j,k): linear in V[j*p+k,:]
        for j in range(m):
            for k in range(p):
                A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])  # (n, R)
                b = np.ones(n)
                AT = A.T  # (R, n)
                ATA = AT @ A  # (R, R)
                ATb = AT @ b  # (R,)
                try:
                    V[j * p + k, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    V[j * p + k, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
    
    return U, V, W

def _try_rank(R, n, m, p, time_budget=100.0):
    """Try to find a rank-R decomposition using random restarts + ALS."""
    start = time.time()
    np.random.seed(42 + R)
    
    best_err = float('inf')
    best_solution = None
    
    num_restarts = 200  # will be limited by time
    
    for restart in range(num_restarts):
        if time.time() - start > time_budget:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-1, 2, size=(n * m, R)).astype(float)
        V = np.random.randint(-1, 2, size=(m * p, R)).astype(float)
        W = np.random.randint(-1, 2, size=(p * n, R)).astype(float)
        
        # Add small noise
        U += np.random.randn(n * m, R) * 0.5
        V += np.random.randn(m * p, R) * 0.5
        W += np.random.randn(p * n, R) * 0.5
        
        # Run ALS
        U, V, W = _als_optimize(U, V, W, n, m, p, max_iters=50, time_limit=2.0)
        
        # Check error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < best_err:
            best_err = max_err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        # If we found an exact solution, try to snap
        if max_err < 1e-6:
            # Try snapping W to half-integers
            W_snapped = _snap_to_halfint(W)
            if W_snapped is not None:
                if _validate(U, V, W_snapped, n, m, p, tol=1e-9):
                    return U, V, W_snapped
            
            # Try snapping all
            U_snapped = _snap_to_halfint(U)
            V_snapped = _snap_to_halfint(V)
            if U_snapped is not None and V_snapped is not None:
                if _validate(U_snapped, V_snapped, W, n, m, p, tol=1e-9):
                    return U_snapped, V_snapped, W
                W_snapped2 = _snap_to_halfint(W)
                if W_snapped2 is not None and _validate(U_snapped, V_snapped, W_snapped2, n, m, p, tol=1e-9):
                    return U_snapped, V_snapped, W_snapped2
        
        # If very close, do a polish
        if max_err < 1e-4:
            U2, V2, W2 = _als_optimize(U, V, W, n, m, p, max_iters=100, time_limit=3.0)
            max_err2 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U2[i * m + j, :], V2[j * p + k, :] * W2[k * n + i, :])
                        err = abs(val - 1.0)
                        if err > max_err2:
                            max_err2 = err
            if max_err2 < best_err:
                best_err = max_err2
                best_solution = (U2.copy(), V2.copy(), W2.copy())
            
            if max_err2 < 1e-6:
                W2_snapped = _snap_to_halfint(W2)
                if W2_snapped is not None and _validate(U2, V2, W2_snapped, n, m, p, tol=1e-9):
                    return U2, V2, W2_snapped
                U2_snapped = _snap_to_halfint(U2)
                V2_snapped = _snap_to_halfint(V2)
                if U2_snapped is not None and V2_snapped is not None:
                    if _validate(U2_snapped, V2_snapped, W2, n, m, p, tol=1e-9):
                        return U2_snapped, V2_snapped, W2
                    W2_snapped2 = _snap_to_halfint(W2)
                    if W2_snapped2 is not None and _validate(U2_snapped, V2_snapped, W2_snapped2, n, m, p, tol=1e-9):
                        return U2_snapped, V2_snapped, W2_snapped2
    
    if best_solution is not None and best_err < 1e-6:
        U, V, W = best_solution
        # Final snap attempt
        W_snapped = _snap_to_halfint(W)
        if W_snapped is not None and _validate(U, V, W_snapped, n, m, p, tol=1e-9):
            return U, V, W_snapped
        U_snapped = _snap_to_halfint(U)
        V_snapped = _snap_to_halfint(V)
        if U_snapped is not None and V_snapped is not None:
            if _validate(U_snapped, V_snapped, W, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W
            W_snapped2 = _snap_to_halfint(W)
            if W_snapped2 is not None and _validate(U_snapped, V_snapped, W_snapped2, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped2
    
    return None

def construct():
    n, m, p = DIMENSIONS
    overall_start = time.time()
    deadline = overall_start + DEADLINE - MARGIN
    
    # Lower bound for rank
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    for R in range(min_rank, n * m * p + 1):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
        
        # Allocate time proportional to remaining
        time_for_rank = min(remaining_time - 5.0, 30.0)  # cap at 30s per rank
        
        result = _try_rank(R, n, m, p, time_budget=time_for_rank)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
