"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_rank55():
    """Build the known rank-55 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0, col] += 1
        U[1 * 3 + 1, col] += 1
        V[0 * 10 + k0, col] += 1
        V[1 * 10 + k1, col] += 1
        W[k0 * 2 + 0, col] += 1
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        U[1 * 3 + 0, col] += 1
        U[1 * 3 + 1, col] += 1
        V[0 * 10 + k0, col] += 1
        W[k0 * 2 + 1, col] += 1
        W[k1 * 2 + 1, col] -= 1
        col += 1
        
        U[0 * 3 + 0, col] += 1
        V[0 * 10 + k1, col] += 1
        V[1 * 10 + k1, col] -= 1
        W[k1 * 2 + 0, col] += 1
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        U[1 * 3 + 1, col] += 1
        V[1 * 10 + k0, col] += 1
        V[0 * 10 + k0, col] -= 1
        W[k0 * 2 + 0, col] += 1
        W[k0 * 2 + 1, col] += 1
        col += 1
        
        U[0 * 3 + 0, col] += 1
        U[0 * 3 + 1, col] += 1
        V[1 * 10 + k1, col] += 1
        W[k0 * 2 + 0, col] -= 1
        W[k1 * 2 + 0, col] += 1
        col += 1
        
        U[1 * 3 + 0, col] += 1
        U[0 * 3 + 0, col] -= 1
        V[0 * 10 + k0, col] += 1
        V[0 * 10 + k1, col] += 1
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        U[0 * 3 + 1, col] += 1
        U[1 * 3 + 1, col] -= 1
        V[1 * 10 + k0, col] += 1
        V[1 * 10 + k1, col] += 1
        W[k0 * 2 + 0, col] += 1
        col += 1
        
        U[0 * 3 + 2, col] += 1
        V[2 * 10 + k0, col] += 1
        W[k0 * 2 + 0, col] += 1
        col += 1
        
        U[0 * 3 + 2, col] += 1
        V[2 * 10 + k1, col] += 1
        W[k1 * 2 + 0, col] += 1
        col += 1
        
        U[1 * 3 + 2, col] += 1
        V[2 * 10 + k0, col] += 1
        W[k0 * 2 + 1, col] += 1
        col += 1
        
        U[1 * 3 + 2, col] += 1
        V[2 * 10 + k1, col] += 1
        W[k1 * 2 + 1, col] += 1
        col += 1
    
    return U, V, W

def _try_eliminate_column(U, V, W, col_to_remove, n, m, p):
    """
    Try to eliminate column `col_to_remove` by solving a linear system.
    We fix all other columns and try to find new values for one "free" column
    that compensates for the removed column's contribution.
    """
    R = U.shape[1]
    if R <= 1:
        return None
    
    # Remove the target column
    mask = np.ones(R, dtype=bool)
    mask[col_to_remove] = False
    U_reduced = U[:, mask]
    V_reduced = V[:, mask]
    W_reduced = W[:, mask]
    
    # The residual we need to compensate:
    # For each (i,j,k), we need: sum_{r != removed} U[i,j,r]*V[j,k,r]*W[k,i,r] + new_contribution = 1
    # So new_contribution = 1 - sum_{r != removed} ...
    # We'll try to express this as a single rank-1 term: a[i,j] * b[j,k] * c[k,i]
    # which is underdetermined. Instead, let's try to find a column vector
    # that, when added as a new column, fixes things.
    
    # Alternative: try to find if the removed column is "redundant" by checking
    # if its contribution can be absorbed. We'll use a least-squares approach.
    
    # For each (i,j,k), compute the current contribution without the removed col
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    if r == col_to_remove:
                        continue
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                residual[i, j, k] = 1.0 - s
    
    # Check if residual is all zeros (column was redundant)
    if np.allclose(residual, 0, atol=1e-9):
        # Just drop the column
        U_new = U_reduced
        V_new = V_reduced
        W_new = W_reduced
        return U_new, V_new, W_new
    
    # Try to find a rank-1 correction: u_vec * v_vec * w_vec
    # where u_vec has shape (n*m,), v_vec has shape (m*p,), w_vec has shape (p*n,)
    # and u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i] = residual[i,j,k]
    
    # This is a tensor decomposition problem for the residual.
    # We'll use a simple approach: fix some entries and solve for others.
    
    # Strategy: pick a "pivot" (i0, j0, k0) where we can determine one factor,
    # then propagate.
    
    # Let's try: for each i, j, k, we need u[i,j] * v[j,k] * w[k,i] = residual[i,j,k]
    # Take logs (when nonzero) - but we need to handle signs and zeros.
    
    # Simpler: try to find u, v, w such that the product matches.
    # Use iterative approach: initialize u, v, w randomly, then iterate.
    
    best_err = float('inf')
    best_u = None
    best_v = None
    best_w = None
    
    np.random.seed(123 + col_to_remove)
    
    for attempt in range(200):
        # Initialize
        u_vec = np.random.choice([-1, 0, 1], size=n*m)
        v_vec = np.random.choice([-1, 0, 1], size=m*p)
        w_vec = np.random.choice([-1, 0, 1], size=p*n)
        
        # Ensure at least some non-zeros
        if np.all(u_vec == 0):
            u_vec[0] = 1
        if np.all(v_vec == 0):
            v_vec[0] = 1
        if np.all(w_vec == 0):
            w_vec[0] = 1
        
        # Iterate: update each factor
        for iteration in range(50):
            # Update u_vec
            for ij in range(n*m):
                i = ij // m
                j = ij % m
                # residual[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                # For each k, we need u[i,j] * (v[j,k] * w[k,i]) = residual[i,j,k]
                # So u[i,j] should be proportional to residual[i,j,k] / (v[j,k]*w[k,i])
                # Use a consensus: pick k that maximizes |v[j,k]*w[k,i]|
                best_k = -1
                best_val = 0
                for k in range(p):
                    prod = v_vec[j*p+k] * w_vec[k*n+i]
                    if abs(prod) > abs(best_val) and prod != 0:
                        best_val = prod
                        best_k = k
                if best_k >= 0:
                    u_vec[ij] = residual[i, j, best_k] / best_val
            
            # Update v_vec
            for jk in range(m*p):
                j = jk // p
                k = jk % p
                best_i = -1
                best_val = 0
                for i in range(n):
                    prod = u_vec[i*m+j] * w_vec[k*n+i]
                    if abs(prod) > abs(best_val) and prod != 0:
                        best_val = prod
                        best_i = i
                if best_i >= 0:
                    v_vec[jk] = residual[best_i, j, k] / best_val
            
            # Update w_vec
            for ki in range(p*n):
                k = ki // n
                i = ki % n
                best_j = -1
                best_val = 0
                for j in range(m):
                    prod = u_vec[i*m+j] * v_vec[j*p+k]
                    if abs(prod) > abs(best_val) and prod != 0:
                        best_val = prod
                        best_j = j
                if best_j >= 0:
                    w_vec[ki] = residual[i, best_j, k] / best_val
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                    err += (val - residual[i,j,k])**2
        
        if err < best_err:
            best_err = err
            best_u = u_vec.copy()
            best_v = v_vec.copy()
            best_w = w_vec.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Add the new column
        U_new = np.column_stack([U_reduced, best_u])
        V_new = np.column_stack([V_reduced, best_v])
        W_new = np.column_stack([W_reduced, best_w])
        return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    U, V, W = _build_rank55()
    current_rank = 55
    
    # Try to eliminate columns one by one
    for target_rank in range(current_rank - 1, max(40, current_rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Try each column as candidate for elimination
        found = False
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            result = _try_eliminate_column(U, V, W, col, n, m, p)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    current_rank = target_rank
                    found = True
                    break
        
        if not found:
            # Can't reduce further with this approach
            break
    
    # Convert to list format with integer/half-integer entries
    U_list = [[0] * current_rank for _ in range(n * m)]
    V_list = [[0] * current_rank for _ in range(m * p)]
    W_list = [[0] * current_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(current_rank):
            val = U[i, r]
            if abs(val - round(val)) < 1e-6:
                U_list[i][r] = int(round(val))
            elif abs(val * 2 - round(val * 2)) < 1e-6:
                U_list[i][r] = round(val * 2) / 2
            else:
                U_list[i][r] = val
    
    for j in range(m * p):
        for r in range(current_rank):
            val = V[j, r]
            if abs(val - round(val)) < 1e-6:
                V_list[j][r] = int(round(val))
            elif abs(val * 2 - round(val * 2)) < 1e-6:
                V_list[j][r] = round(val * 2) / 2
            else:
                V_list[j][r] = val
    
    for k in range(p * n):
        for r in range(current_rank):
            val = W[k, r]
            if abs(val - round(val)) < 1e-6:
                W_list[k][r] = int(round(val))
            elif abs(val * 2 - round(val * 2)) < 1e-6:
                W_list[k][r] = round(val * 2) / 2
            else:
                W_list[k][r] = val
    
    # Final verification
    if not _verify(U_list, V_list, W_list, n, m, p):
        # Fall back to the original rank-55
        U, V, W = _build_rank55()
        current_rank = 55
        U_list = [[0] * current_rank for _ in range(n * m)]
        V_list = [[0] * current_rank for _ in range(m * p)]
        W_list = [[0] * current_rank for _ in range(p * n)]
        
        for i in range(n * m):
            for r in range(current_rank):
                U_list[i][r] = int(U[i, r])
        for j in range(m * p):
            for r in range(current_rank):
                V_list[j][r] = int(V[j, r])
        for k in range(p * n):
            for r in range(current_rank):
                W_list[k][r] = int(W[k, r])
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
