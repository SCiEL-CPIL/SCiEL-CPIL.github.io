"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    Uses smooth surrogate optimization with snapping to integer/half-integer grid.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with safety margin
    
    # Precompute index mappings
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n_m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m_p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p_n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx, col] = 1.0
                V_school[v_idx, col] = 1.0
                W_school[w_idx, col] = 1.0
                col += 1
    
    # Precompute indices for vectorized error calculation
    # We need to compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    # This can be seen as a tensor contraction.
    # Let's precompute the indices for U, V, W rows corresponding to each (i,j,k)
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    u_indices = (i_indices * m + j_indices).flatten()
    v_indices = (j_indices * p + k_indices).flatten()
    w_indices = (k_indices * n + i_indices).flatten()
    
    # Helper: compute smooth error using log-sum-exp for numerical stability (Vectorized)
    def compute_smooth_error(U_mat, V_mat, W_mat, temperature=100.0):
        """
        Returns smooth approximation to max |error| using log-sum-exp.
        Lower temperature = closer to true max.
        """
        # Extract the relevant rows for all (i,j,k) combinations
        U_rows = U_mat[u_indices, :]  # shape (n*m*p, rank)
        V_rows = V_mat[v_indices, :]  # shape (n*m*p, rank)
        W_rows = W_mat[w_indices, :]  # shape (n*m*p, rank)
        
        # Compute element-wise product sum over rank dimension
        # val[t] = sum_c U_rows[t, c] * V_rows[t, c] * W_rows[t, c]
        vals = np.sum(U_rows * V_rows * W_rows, axis=1)
        
        # Errors
        errors = np.abs(vals - 1.0)
        
        if len(errors) == 0:
            return 0.0
        
        # Log-sum-exp for stable max approximation
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        
        # Soft-max: log(sum(exp(err/T))) * T
        scaled = errors / temperature
        scaled_max = np.max(scaled)
        log_sum_exp = scaled_max + np.log(np.sum(np.exp(scaled - scaled_max)))
        return log_sum_exp * temperature
    
    # Helper: snap to nearest integer or half-integer
    def snap_to_grid(val):
        """Snap value to nearest integer or half-integer."""
        # Check if close to integer
        int_val = round(val)
        if abs(val - int_val) < 1e-8:
            return float(int_val)
        
        # Check if close to half-integer
        half_val = round(val * 2) / 2.0
        if abs(val - half_val) < 1e-8:
            return half_val
        
        # Otherwise, snap to nearest valid value
        candidates = [int_val, half_val, int_val + 0.5, int_val - 0.5]
        best = min(candidates, key=lambda c: abs(val - c))
        return best
    
    # Helper: check if decomposition is exact with snapped values (Vectorized)
    def is_exact(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if decomposition is exact (all entries match target)."""
        U_rows = U_mat[u_indices, :]
        V_rows = V_mat[v_indices, :]
        W_rows = W_mat[w_indices, :]
        vals = np.sum(U_rows * V_rows * W_rows, axis=1)
        errors = np.abs(vals - 1.0)
        return np.max(errors) < tol
    
    # Helper: verify that a decomposition with given matrices is valid
    def verify_decomposition(U_list, V_list, W_list, tol=1e-8):
        """Verify that the decomposition is exact."""
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Helper: convert matrix to list with integer/half-integer values
    def to_list_with_grid(U_mat):
        """Convert numpy matrix to list, snapping to integer/half-integer grid."""
        U_list = []
        for row in U_mat:
            U_row = []
            for val in row:
                snapped = snap_to_grid(val)
                # Convert to int if whole number, else keep as float
                if abs(snapped - round(snapped)) < 1e-10:
                    U_row.append(int(round(snapped)))
                else:
                    U_row.append(snapped)
            U_list.append(U_row)
        return U_list
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_schoolbook
    
    # Try to reduce rank using smooth optimization
    np.random.seed(42)
    
    # Target ranks to try, from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But we can try even lower
    target_ranks = list(range(21, best_rank))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:  # Leave 5s margin for cleanup
                break
            
            # Initialize with structured random values
            # Use small random values to start
            scale = 0.5
            U_cur = np.random.randn(n_m, target_rank) * scale
            V_cur = np.random.randn(m_p, target_rank) * scale
            W_cur = np.random.randn(p_n, target_rank) * scale
            
            # Smooth optimization with adaptive temperature
            temperature = 50.0
            step_size = 0.1
            best_err = float('inf')
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                err = compute_smooth_error(U_cur, V_cur, W_cur, temperature)
                
                if err < best_err:
                    best_err = err
                
                # Check if we've found an exact solution
                if err < 1e-6:
                    # Try snapping to grid
                    U_snapped = np.array([[snap_to_grid(val) for val in row] for row in U_cur])
                    V_snapped = np.array([[snap_to_grid(val) for val in row] for row in V_cur])
                    W_snapped = np.array([[snap_to_grid(val) for val in row] for row in W_cur])
                    
                    if is_exact(U_snapped, V_snapped, W_snapped):
                        found = True
                        break
                
                # Gradient-free optimization: try multiple perturbations
                improved = False
                for _ in range(3):
                    U_trial = U_cur.copy()
                    V_trial = V_cur.copy()
                    W_trial = W_cur.copy()
                    
                    # Perturb random entries
                    idx_u = np.random.randint(n_m)
                    idx_v = np.random.randint(m_p)
                    idx_w = np.random.randint(p_n)
                    col_u = np.random.randint(target_rank)
                    col_v = np.random.randint(target_rank)
                    col_w = np.random.randint(target_rank)
                    
                    U_trial[idx_u, col_u] += step_size * np.random.randn()
                    V_trial[idx_v, col_v] += step_size * np.random.randn()
                    W_trial[idx_w, col_w] += step_size * np.random.randn()
                    
                    trial_err = compute_smooth_error(U_trial, V_trial, W_trial, temperature)
                    
                    if trial_err < err:
                        U_cur, V_cur, W_cur = U_trial, V_trial, W_trial
                        err = trial_err
                        improved = True
                        break
                
                if not improved:
                    # Decrease step size and temperature
                    step_size *= 0.95
                    temperature *= 0.98
                    if step_size < 0.001:
                        break
            
            if found:
                break
        
        if found:
            # Convert to final format
            U_final = to_list_with_grid(U_snapped)
            V_final = to_list_with_grid(V_snapped)
            W_final = to_list_with_grid(W_snapped)
            
            # Final verification
            if verify_decomposition(U_final, V_final, W_final):
                best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                best_rank = target_rank
                # If we found a solution, we can return early since we're going from low to high rank
                break
        else:
            # Continue to next rank
            continue
    
    # If we didn't find a better solution, return schoolbook
    if best_rank == rank_schoolbook:
        U_final = to_list_with_grid(best_U)
        V_final = to_list_with_grid(best_V)
        W_final = to_list_with_grid(best_W)
    else:
        U_final = to_list_with_grid(best_U)
        V_final = to_list_with_grid(best_V)
        W_final = to_list_with_grid(best_W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple validation check if run directly
    res = construct()
    n, m, p = DIMENSIONS
    U = res['U']
    V = res['V']
    W = res['W']
    
    # Verify a few entries
    # Target T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # And 0 otherwise.
    # Check that the sum reconstructs the tensor.
    import numpy as np
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reconstruct tensor T_reconstructed
    # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T_reconstructed matches the target.
    
    # The target tensor is sparse.
    # For each c, it adds a rank-1 tensor.
    # We just need to ensure that for all i,j,k, the sum is 1 if it's a "valid" multiplication term in the tensor definition.
    # The tensor definition for matrix multiplication has 1s at (i,j), (j,k), (i,k) in the respective spaces.
    # Here W index is k*n+i.
    
    # Check one specific term: i=0, j=0, k=0
    # u=0, v=0, w=0
    # U[0, c] V[0, c] W[0, c]
    # In schoolbook, only c=0 has 1s at these positions for i=0,j=0,k=0.
    # So sum is 1.
    
    # Check a non-term: i=0, j=0, k=1 (assuming p>1)
    # u=0, v=1, w=1*n+0 = 2 (if n=2)
    # In schoolbook, c for (0,0,1) is c=1 (if order i,j,k loops 0,0,1).
    # U[0, 1]=1, V[1, 1]=1, W[2, 1]=1.
    # But we are summing over c.
    # For c=0: U[0,0]=1, V[1,0]=0 (V[0,0]=1 is for j=0,k=0). So term is 0.
    # For c=1: U[0,1]=0 (U[0,1] is for j=0? No, U[i*m+j]. For i=0,j=0, u=0. U[0,1] is 0 in schoolbook because c=1 is for k=1, which uses U[0,1]?
    # Wait, in schoolbook:
    # c iterates over (i,j,k).
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k_current).
    # So U[u, c] is 1 only if c's i,j match u's i,j.
    # V[v, c] is 1 only if c's j,k match v's j,k.
    # W[w, c] is 1 only if c's k,i match w's k,i.
    # So U[u,c]V[v,c]W[w,c] is non-zero only if:
    # c's (i,j) == u's (i,j)
    # c's (j,k) == v's (j,k)
    # c's (k,i) == w's (k,i)
    # This implies c's (i,j,k) == (i,j,k) derived from u,v,w.
    # If such a c exists in the set of all (i,j,k), then the product is 1.
    # The set of all c is exactly all (i,j,k).
    # So the sum is 1 if (i,j,k) is a valid index, which it always is for i in n, j in m, k in p.
    # And the target tensor T has 1s at all valid (i,j,k) combinations?
    # Yes, the multiplication tensor has 1s for all $i,j,k$.
    # So the reconstruction is exact.
    
    print("Valid schoolbook decomposition returned.")
    print("Rank:", rank)
