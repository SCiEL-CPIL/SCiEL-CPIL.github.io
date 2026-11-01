import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _build_rank_28():
    """Return the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        # M2
        U[(i0+1) * m + j0, col] = 1
        U[(i0+1) * m + (j0+1), col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        W[(k0+1) * n + (i0+1), col] = -1
        col += 1
        # M3
        U[i0 * m + j0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = -1
        W[(k0+1) * n + i0, col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        # M4
        U[(i0+1) * m + (j0+1), col] = 1
        V[(j0+1) * p + k0, col] = 1
        V[j0 * p + k0, col] = -1
        W[k0 * n + i0, col] = 1
        W[k0 * n + (i0+1), col] = 1
        col += 1
        # M5
        U[i0 * m + j0, col] = 1
        U[i0 * m + (j0+1), col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = -1
        W[(k0+1) * n + i0, col] = 1
        col += 1
        # M6
        U[(i0+1) * m + j0, col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + (k0+1), col] = 1
        W[(k0+1) * n + (i0+1), col] = 1
        col += 1
        # M7
        U[i0 * m + (j0+1), col] = 1
        U[(i0+1) * m + (j0+1), col] = -1
        V[(j0+1) * p + k0, col] = 1
        V[(j0+1) * p + (k0+1), col] = 1
        W[k0 * n + i0, col] = 1
        col += 1

    # Block A: j=0 (2x1 by 1x5) -> rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    # Block B: j=1,2 (2x2 by 2x5)
    add_strassen(0, 1, 0)
    for i in range(n):
        for j in (1, 2):
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    add_strassen(0, 1, 3)

    assert col == rank, f"Column count mismatch: {col} != {rank}"
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = delta(i,k) which is the matmul tensor
    # The decomposition says: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (i==k ? 1 : 0)
    # Wait, the matmul tensor is T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor entries are T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index sense)
    # Actually the matmul tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify using the flattenings
    # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r should give the tensor T where T[i,j,k] = 1
    
    # Efficient check: for each (i,k) pair, sum over j: (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) should equal m (since sum over j of 1 = m)
    # Actually no. Let me think again.
    # The matmul tensor T satisfies T[i,j,k] = 1 for all i,j,k.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Check a few random entries
    rng = np.random.RandomState(42)
    for _ in range(50):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        val = 0.0
        for r in range(rank):
            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if abs(val - 1.0) > tol:
            return False
    # Also check full tensor via matrix operations for speed
    # For each r, form M_r = W[:,r] * (U[:,r] outer V[:,r] reshaped)
    # Actually let's just do the full check
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        tensor += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    if not np.allclose(tensor, 1.0, atol=tol):
        return False
    return True


def _search_rank_27(deadline):
    """Try to find a rank-27 decomposition via nonlinear optimization."""
    n, m, p = 2, 3, 5
    target_rank = 27
    
    # Parameterize: 27 columns in U (6x27), V (15x27), W (10x27)
    # Total parameters: 6*27 + 15*27 + 10*27 = 27*31 = 837
    # Objective: minimize sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    # This is a non-convex polynomial optimization.
    
    # Use the rank-28 solution as a starting point, zero out one column, and optimize
    U28, V28, W28 = _build_rank_28()
    
    # Strategy: try removing each of the 28 columns and see if the remaining 27 can be adjusted
    # to still give a valid decomposition. We'll use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    best_residual = np.inf
    best_params = None
    
    # For each column to remove, try to optimize the remaining 27 columns
    for remove_col in range(28):
        if time.time() > deadline:
            break
        
        # Create initial guess: rank-28 with column remove_col zeroed out
        U_init = U28[:, :remove_col].copy()
        if remove_col < 27:
            U_init = np.hstack([U28[:, :remove_col], U28[:, remove_col+1:28]])
        V_init = V28[:, :remove_col].copy()
        if remove_col < 27:
            V_init = np.hstack([V28[:, :remove_col], V28[:, remove_col+1:28]])
        W_init = W28[:, :remove_col].copy()
        if remove_col < 27:
            W_init = np.hstack([W28[:, :remove_col], W28[:, remove_col+1:28]])
        
        # Flatten parameters
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        def objective(x):
            U = x[:6*27].reshape(6, 27)
            V = x[6*27:6*27+15*27].reshape(15, 27)
            W = x[6*27+15*27:].reshape(10, 27)
            # Compute residual
            tensor = np.zeros((2, 3, 5))
            for r in range(27):
                tensor += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            residual = tensor - 1.0
            return np.sum(residual**2)
        
        # Try to optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_residual:
                best_residual = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Check if we found a valid rank-27 solution
    if best_params is not None and best_residual < 1e-8:
        U = best_params[:6*27].reshape(6, 27)
        V = best_params[6*27:6*27+15*27].reshape(15, 27)
        W = best_params[6*27+15*27:].reshape(10, 27)
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try rank-27 search
    result_27 = _search_rank_27(deadline)
    if result_27 is not None:
        U, V, W = result_27
        # Convert to list format
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to rank-28
    U28, V28, W28 = _build_rank_28()
    U_list = [[float(x) for x in row] for row in U28]
    V_list = [[float(x) for x in row] for row in V28]
    W_list = [[float(x) for x in row] for row in W28]
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
