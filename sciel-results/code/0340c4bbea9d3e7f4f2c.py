"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

_DEADLINE = time.time() + 145.0

def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, rank):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _try_alternating_minimization(rank, max_iter=500, tol=1e-10):
    """
    Alternating minimization for tensor decomposition.
    Fix two matrices, solve for the third via least squares.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute index mappings
    # For each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(nm, rank) * 0.5
    V = np.random.randn(mp, rank) * 0.5
    W = np.random.randn(pn, rank) * 0.5
    
    prev_cost = float('inf')
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        
        # Update W: for each (i,j,k), we need sum_r W[k*n+i, r] * (U[i*m+j,r]*V[j*p+k,r]) = 1
        # This is a linear system for W
        # For each k*n+i, we have equations from all (j) where the coefficient is U[i*m+j,r]*V[j*p+k,r]
        
        # Build system for W: 
        # For fixed (k,i), the equation is: sum_r W[k*n+i, r] * A_{(k,i),r} = b_{(k,i)}
        # where A_{(k,i),r} = sum over j of U[i*m+j, r] * V[j*p+k, r]
        # Wait, that's not right. Let me reconsider.
        
        # The constraint is: for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # 
        # If we fix U and V, then for each (i,j,k):
        #   sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
        #
        # For a fixed (k,i), we have multiple equations (one for each j):
        #   sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1  for j=0..m-1
        #
        # This is an overdetermined linear system for W[k*n+i, :] (a row of W)
        # The system is: M_{(k,i)} @ W[k*n+i, :] = ones(m)
        # where M_{(k,i)} is an m x rank matrix with M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Build M: m x rank
                M = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Solve M @ w_row = ones(m)
                # Least squares
                w_row, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                W[k*n+i, :] = w_row
        
        # Update V: fix U and W
        # For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # For fixed (j,k): sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
        # M_{(j,k)}: n x rank, M[i,r] = U[i*m+j,r] * W[k*n+i,r]
        # Solve M @ v_row = ones(n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                v_row, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                V[j*p+k, :] = v_row
        
        # Update U: fix V and W
        # For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # For fixed (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        # M_{(i,j)}: p x rank, M[k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Solve M @ u_row = ones(p)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                u_row, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[i*m+j, :] = u_row
        
        # Compute cost
        cost = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    cost += (val - 1.0)**2
        
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost
    
    if prev_cost > 1e-6:
        return None
    
    # Try to round to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify rounded version
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_i[i*m+j, r] * V_i[j*p+k, r] * W_i[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # If rounding doesn't work, return the non-rounded version if it's close enough
    if prev_cost < 1e-8:
        return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_scipy(rank, timeout=10.0):
    """
    Attempt to find a rank-`rank` decomposition using scipy least_squares.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def _residuals(params, rank):
        u_size = nm * rank
        v_size = mp * rank
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:]

        U = u_flat.reshape(nm, rank)
        V = v_flat.reshape(mp, rank)
        W = w_flat.reshape(pn, rank)

        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    residuals[i, j, k] = val - 1.0
        return residuals.flatten()

    from scipy.optimize import least_squares

    best_result = None
    best_cost = float('inf')

    n_attempts = 6
    for attempt in range(n_attempts):
        if _remaining() < timeout:
            break
        np.random.seed(2000 + rank * 100 + attempt)
        scale = 0.5
        u0 = np.random.randn(nm, rank) * scale
        v0 = np.random.randn(mp, rank) * scale
        w0 = np.random.randn(pn, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])

        try:
            result = least_squares(_residuals, params0, args=(rank,), method='lm', max_nfev=1500)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                if result.cost < 1e-12:
                    break
        except Exception:
            pass

    if best_result is None:
        return None

    if best_cost > 1e-8:
        return None

    u_size = nm * rank
    v_size = mp * rank
    U = best_result.x[:u_size].reshape(nm, rank)
    V = best_result.x[u_size:u_size+v_size].reshape(mp, rank)
    W = best_result.x[u_size+v_size:].reshape(pn, rank)

    # Try to round to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2

    params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
    res_r = _residuals(params_r, rank)
    if np.max(np.abs(res_r)) < 1e-6:
        U, V, W = U_r, V_r, W_r

    if _verify(U, V, W, rank):
        return U.tolist(), V.tolist(), W.tolist()

    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    params_i = np.concatenate([U_i.flatten(), V_i.flatten(), W_i.flatten()])
    res_i = _residuals(params_i, rank)
    if np.max(np.abs(res_i)) < 1e-6:
        U, V, W = U_i, V_i, W_i
        if _verify(U, V, W, rank):
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _schoolbook():
    """Return the schoolbook rank-18 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def _try_structured_construction(rank):
    """
    Try to construct a decomposition using algebraic structure.
    For 2x3x3, we can try to use Strassen-like ideas.
    """
    n, m, p = DIMENSIONS
    
    # One approach: use the fact that 2x3x3 can be related to smaller multiplications
    # Let's try a direct construction if rank is small enough
    
    # For now, this is a placeholder for structured constructions
    return None


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Searches for lower-rank decompositions using multiple methods.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18

    # Try ranks from low to high
    # Start from a reasonable lower bound
    # The tensor rank is at least max(n*m, m*p, p*n) / max(n,m,p) but that's not tight
    # For (2,3,3), let's try from rank 10 upward
    
    for rank in range(10, schoolbook_rank + 1):
        if _remaining() < 5:
            break
        
        # Try alternating minimization first (faster for this problem)
        result = _try_alternating_minimization(rank, max_iter=300)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try scipy if time permits
        if _remaining() > 15:
            result = _try_scipy(rank, timeout=8.0)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    return _schoolbook()
