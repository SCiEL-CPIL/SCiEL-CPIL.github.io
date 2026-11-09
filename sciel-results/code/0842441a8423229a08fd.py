import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Strategy: Use Alternating Least Squares (ALS) with random restarts to minimize 
    the tensor rank. We search for the smallest rank r such that the reconstruction 
    error is negligible.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8
    
    # Construct the target tensor T as a 3D array or list of non-zero coordinates
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # We will represent T as a sparse structure for efficient ALS updates.
    # Coordinates (alpha, beta, gamma) where T is 1.
    coords = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                coords.append((alpha, beta, gamma))
                
    num_nonzeros = len(coords)
    
    # We want to find min rank r.
    # Schoolbook is 32.
    # We try ranks from 20 to 32.
    # If a rank works (error < 1e-9), we return it.
    
    def try_rank(r, num_restarts=10, max_iters=500, tol=1e-9):
        """
        Try to find a rank-r decomposition using ALS.
        Returns (U, V, W) if successful, else None.
        """
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(num_restarts):
            # Random initialization
            # Scale matters for convergence. Standard deviation 1/sqrt(dim)
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # Pre-compute structures for faster ALS
            # For each r, we need to solve for U, V, W.
            # T = sum_r U_r V_r^T W_r^T
            # This is a sum of rank-1 tensors.
            # ALS update for U:
            # For each row alpha of U, we have equations from coords where first index is alpha.
            # Let's group coords by alpha, beta, gamma.
            
            # To speed up, we can pre-group indices.
            # But with small dims, we can just iterate.
            
            for it in range(max_iters):
                # Update U
                # For each alpha, we want to minimize sum_{(alpha, beta, gamma) in coords} (T - sum_r' U_{alpha r'} V_{beta r'} W_{gamma r'})^2
                # This is a linear least squares problem for U_{alpha, :}.
                # Let M_alpha be the matrix where M_alpha[coord_idx, r] = V_{beta, r} * W_{gamma, r}
                # Then U_{alpha, :} = argmin || M_alpha * U_{alpha, :} - t_alpha ||^2
                # where t_alpha is the vector of T values (all 1s) for coords starting with alpha.
                
                # We will build M_alpha on the fly for each alpha.
                # Since dim_u=8, this is cheap.
                
                new_U = np.zeros((dim_u, r))
                for alpha in range(dim_u):
                    # Find all coords with first index alpha
                    # And build the design matrix
                    rows = []
                    target_vals = []
                    for (a, b, g) in coords:
                        if a == alpha:
                            # The contribution of U_{alpha, r} is multiplied by V_{b, r} * W_{g, r}
                            # So the row in design matrix is [V_{b,0}*W_{g,0}, V_{b,1}*W_{g,1}, ...]
                            row = V[b, :] * W[g, :]
                            rows.append(row)
                            target_vals.append(1.0)
                    
                    if not rows:
                        continue
                        
                    M = np.array(rows)
                    t = np.array(target_vals)
                    # Solve M * x = t
                    # x is U[alpha, :]
                    try:
                        # Use lstsq for stability
                        sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                        new_U[alpha, :] = sol
                    except:
                        pass
                        
                U = new_U
                
                # Update V
                new_V = np.zeros((dim_v, r))
                for beta in range(dim_v):
                    rows = []
                    target_vals = []
                    for (a, b, g) in coords:
                        if b == beta:
                            row = U[a, :] * W[g, :]
                            rows.append(row)
                            target_vals.append(1.0)
                    
                    if not rows:
                        continue
                        
                    M = np.array(rows)
                    t = np.array(target_vals)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                        new_V[beta, :] = sol
                    except:
                        pass
                        
                V = new_V
                
                # Update W
                new_W = np.zeros((dim_w, r))
                for gamma in range(dim_w):
                    rows = []
                    target_vals = []
                    for (a, b, g) in coords:
                        if g == gamma:
                            row = U[a, :] * V[b, :]
                            rows.append(row)
                            target_vals.append(1.0)
                    
                    if not rows:
                        continue
                        
                    M = np.array(rows)
                    t = np.array(target_vals)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
                        new_W[gamma, :] = sol
                    except:
                        pass
                        
                W = new_W
                
                # Calculate error
                # Error = sum_{coords} (1 - sum_r U[a,r] V[b,r] W[g,r])^2
                err = 0.0
                for (a, b, g) in coords:
                    val = np.dot(U[a, :], V[b, :] * W[g, :])
                    err += (1.0 - val) ** 2
                
                # Also check entries that should be 0?
                # The tensor is sparse. The ALS minimizes error only on non-zero entries.
                # We must ensure that the "zero" entries are also close to 0.
                # However, typically if the non-zero part fits perfectly and rank is low,
                # the zero part might not be 0. We need to check the full tensor error.
                
                # Full tensor error check is expensive (8*16*8 = 1024 entries).
                # But 1024 is small. Let's do it every few iterations or at the end.
                if it % 10 == 0 or it == max_iters - 1:
                    full_err = 0.0
                    # Iterate over all possible (a, b, g)
                    for a in range(dim_u):
                        for b in range(dim_v):
                            for g in range(dim_w):
                                val = np.dot(U[a, :], V[b, :] * W[g, :])
                                # Determine if this entry should be 1 or 0
                                # Check if (a,b,g) is in coords
                                # We can precompute a set of tuples for O(1) lookup
                                # But for now, we just check if it's in the list? No, too slow.
                                # Let's use a set.
                                pass
                    # Optimization: Precompute target tensor as a dict or set
                    # For now, let's just assume if non-zero error is low, we check zeros separately.
                    
                    # Let's do a proper full error check
                    full_err = 0.0
                    for a in range(dim_u):
                        for b in range(dim_v):
                            for g in range(dim_w):
                                val = np.dot(U[a, :], V[b, :] * W[g, :])
                                # Check if (a,b,g) is a valid coord
                                # a = i*m+j, b=j*p+k, g=k*n+i
                                # We can check consistency:
                                # j = a % m
                                # i = a // m
                                # k = b // p
                                # j2 = b % p  (wait, b = j*p + k, so j = b//p, k = b%p)
                                # Let's re-verify indices.
                                # A index: i in [0,n), j in [0,m). a = i*m + j.
                                # B index: j in [0,m), k in [0,p). b = j*p + k.
                                # C index: k in [0,p), i in [0,n). g = k*n + i.
                                
                                # Consistency:
                                # From a: i_a = a // m, j_a = a % m
                                # From b: j_b = b // p, k_b = b % p
                                # From g: k_g = g // n, i_g = g % n
                                
                                # Valid if j_a == j_b and k_b == k_g and i_g == i_a
                                i_a = a // m
                                j_a = a % m
                                j_b = b // p
                                k_b = b % p
                                k_g = g // n
                                i_g = g % n
                                
                                if j_a == j_b and k_b == k_g and i_g == i_a:
                                    target = 1.0
                                else:
                                    target = 0.0
                                    
                                full_err += (target - val) ** 2
                    
                    if full_err < best_error:
                        best_error = full_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                    if best_error < tol:
                        return best_U, best_V, best_W

        if best_error < tol:
            return best_U, best_V, best_W
        return None

    # Search for minimal rank
    # Start from low ranks to find the minimum
    # Range: 20 to 32
    for r in range(20, 33):
        start_time = time.time()
        result = try_rank(r, num_restarts=5, max_iters=200, tol=1e-9)
        elapsed = time.time() - start_time
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer snapping if necessary
            # The problem allows integer/half-integer.
            # Check if entries are close to integers or half-integers
            def snap_matrix(M):
                snapped = []
                for row in M:
                    new_row = []
                    for val in row:
                        # Check if close to integer or half-integer
                        # Round to nearest 0.5
                        half_int = np.round(val * 2) / 2
                        if abs(val - half_int) < 1e-6:
                            new_row.append(half_int)
                        else:
                            # If not close, keep as is? But requirement is integer/half-integer.
                            # If ALS didn't converge to such a solution, this rank might be invalid for the constraint.
                            # However, if the error is 0, the solution is exact.
                            # Matrix multiplication tensors often have integer decompositions.
                            new_row.append(half_int)
                    snapped.append(new_row)
                return snapped

            U_list = snap_matrix(U)
            V_list = snap_matrix(V)
            W_list = snap_matrix(W)
            
            # Verify the snapped solution
            def verify(U_l, V_l, W_l):
                for a in range(dim_u):
                    for b in range(dim_v):
                        for g in range(dim_w):
                            val = 0.0
                            for rr in range(r):
                                val += U_l[a][rr] * V_l[b][rr] * W_l[g][rr]
                            
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_g = g // n
                            i_g = g % n
                            
                            if j_a == j_b and k_b == k_g and i_g == i_a:
                                target = 1.0
                            else:
                                target = 0.0
                                
                            if abs(val - target) > 1e-6:
                                return False
                return True

            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # If snapping failed verification, try without snapping or return raw floats?
                # The contract says "integer/half-integer entries".
                # If we can't snap, we might need a higher rank or different init.
                # But for now, if error was < 1e-9, it should be snap-able.
                pass

    # Fallback to schoolbook rank 32
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
    return {"U": U, "V": V, "W": W}
