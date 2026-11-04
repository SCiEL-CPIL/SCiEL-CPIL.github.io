import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Pre-compute target tensor T
    # T[i, j, k] = 1 if this is a valid multiplication term?
    # The tensor for matrix multiplication C = A * B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T_{i, j, k} has 1 at (i, j, k) for the term A[i, j]B[j, k].
    # So T is 1 everywhere? No.
    # The decomposition is:
    # sum_t U[i, j, t] V[j, k, t] W[k, i, t] = delta_{i, k}^{C} ?
    # No, the identity is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This is represented by the tensor T where T[i, j, k] = 1 for all i, j, k?
    # No, the "matrix multiplication tensor" is the linear map from A, B to C.
    # In the tensor decomposition context, we want to represent the bilinear map.
    # The standard tensor is $T_{i, j, k}$ such that $C_{i, k} = \sum_{j} A_{i, j} B_{j, k}$.
    # This is not a standard "tensor" in the sense of a 3-way array of coefficients for a single scalar.
    # It is a collection of $n \times p$ matrices?
    # Actually, the tensor $\mathcal{T}$ is defined by $\mathcal{T}_{i, j, k} = 1$ if we are computing $C_{i, k}$ using $A_{i, j}$ and $B_{j, k}$.
    # So $\mathcal{T}$ has 1s at all $(i, j, k)$?
    # If so, the rank is $n m p$ for schoolbook.
    # Yes, the "matrix multiplication tensor" $M_{n, m, p}$ has entries 1 at $(i, j, k)$ for all $i, j, k$.
    # Wait, if T is all 1s, then $U V W$ sum must be 1 for all i,j,k.
    # This is exactly what the previous code did: `err[eq_idx] = s - 1.0`.
    
    T_target = np.ones((n, m, p))

    def compute_error_numpy(U, V, W, r):
        # U: (nm, r) -> reshape to (n, m, r)
        # V: (mp, r) -> reshape to (m, p, r)
        # W: (pn, r) -> reshape to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # einsum: 'ijr,jkr,kir->ijk'
        try:
            result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            # Fallback to loop if einsum fails or is too slow (unlikely)
            result = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U[i,j,:] dot (V[j,k,:] * W[k,i,:])
                        result[i, j, k] = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
        
        err = result - T_target
        return err

    def check_integer_half(U, V, W, r, tol=1e-9):
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def search_rank(r, time_limit):
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            curr_err = np.sum(compute_error_numpy(U, V, W, r) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                if check_integer_half(U, V, W, r):
                    return U, V, W

            # Coordinate Descent
            no_improve = 0
            max_iters = 5000
            step_size = 0.5
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                # Pick random variable
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                    new_val = U[a, t]
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                    new_val = V[b, t]
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                    new_val = W[c, t]
                
                new_err = np.sum(compute_error_numpy(U, V, W, r) ** 2)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if curr_err < 1e-10:
                        if check_integer_half(U, V, W, r):
                            return U, V, W
                else:
                    # Revert
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
                    no_improve += 1
                    if no_improve > 200:
                        break
                        
            # Snap to grid and check
            if curr_err < 5.0: # If reasonably close
                U_snap = np.round(2 * U) / 2.0
                V_snap = np.round(2 * V) / 2.0
                W_snap = np.round(2 * W) / 2.0
                snap_err = np.sum(compute_error_numpy(U_snap, V_snap, W_snap, r) ** 2)
                if snap_err < 1e-9:
                    return U_snap, V_snap, W_snap

        return None

    # Try ranks lower than schoolbook
    # We prioritize finding *any* improvement.
    # Rank 56 is a known bound for 3x3x8?
    # Let's try a range.
    
    ranks_to_try = list(range(56, 65)) # 56 to 64
    # If 56 fails, try higher.
    # If we find one, return it.
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        sol = search_rank(r, deadline)
        if sol:
            U, V, W = sol
            # Convert to lists of lists for output
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
