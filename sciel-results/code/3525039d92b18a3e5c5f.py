import time
import numpy as np
import sys

DEADLINE_S = 110.0
DIMENSIONS = [4, 5, 5]

def _build_tensor_3way(n, m, p):
    """Build the multiplication tensor T[i, j, k] = 1 for all (i,j,k)."""
    return np.ones((n, m, p), dtype=np.float64)

def _validate_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Validate rank-R decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U_reshaped = U.reshape((n, m, -1))
    V_reshaped = V.reshape((m, p, -1))
    W_reshaped = W.reshape((p, n, -1))
    
    # Reconstruct T_reconstructed[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # This is a batched dot product effectively.
    # T_reconstructed[i, j, k] = sum_r (U_reshaped[i, j, r] * W_reshaped[k, i, r]) * V_reshaped[j, k, r]
    
    # Let's do it efficiently.
    # First contract U and W over r and i? No, i is in U and W.
    # U_reshaped: (n, m, R)
    # W_reshaped: (p, n, R) -> transpose to (n, p, R) to match i index? No, W is indexed by k, n, i. 
    # W[k, n, i]. In the tensor contraction for matrix mult:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] corresponds to A[i, j] B[j, k].
    # Decomposition: T[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[k, i].
    # Note W is indexed [k*n+i, r] in flattened form, so W_reshaped[k, n, i] where dim 1 is n (0..n-1) and index is i.
    
    # T_reconstructed[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r] # Wait, W index order.
    # Problem states: W[k*n+i, r]. So W_reshaped should be W.reshape(p, n, R).
    # Index i is the last spatial dimension of W.
    # So W_reshaped[k, i, r] is correct.
    
    # Let's check dimensions:
    # U_reshaped: (n, m, R)
    # V_reshaped: (m, p, R)
    # W_reshaped: (p, n, R)
    
    # We want Result[i, j, k]
    # Sum over r: U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # This can be computed as:
    # Temp[i, j, k] = sum_r U[i, j, r] * W[k, i, r]  -> shape (n, m, p)
    # Then Result[i, j, k] = sum_r Temp[i, j, k] * V[j, k, r]? No, r is summed in Temp.
    
    # Better approach:
    # 1. Compute M[j, k, i, r] = V_reshaped[j, k, r] * W_reshaped[k, i, r]
    #    This mixes indices from V and W.
    # 2. Contract with U.
    
    # Actually, just doing the loop is safer for correctness given small N, but vectorization is faster.
    # Let's try:
    # Product_1[i, j, k] = sum_r U_reshaped[i, j, r] * W_reshaped[k, i, r]
    # This requires W to be indexed [k, i, r]. W_reshaped is (p, n, R).
    # We need to contract U (n, m, R) and W (p, n, R) over R and n(i).
    # Result_P1[i, j, k] = sum_r U[i, j, r] * W[k, i, r]
    # This is effectively a matrix multiplication where i is a "batch" dimension shared.
    # Let's use einsum: 'ijr,kr->ik' is wrong because of i.
    # U[i, j, r] * W[k, i, r] -> sum over r.
    # Result is shape (n, m, p).
    # np.einsum('ijr,kir->ijk', U_reshaped, W_reshaped)
    # Check indices:
    # U: i, j, r
    # W: k, i, r (since W_reshaped is p, n, R -> k, i, r)
    # Output: i, j, k
    # This sums over r. It keeps i, j, k.
    # P1[i, j, k] = sum_r U[i, j, r] * W[k, i, r]
    
    # Then we need to multiply this by V[j, k, r]? No, the decomposition is a sum of outer products.
    # T[i, j, k] = sum_r (U_r[i, j] * V_r[j, k] * W_r[k, i])
    # This is not just P1 * V. V is part of the sum.
    # T[i, j, k] = sum_r (U_reshaped[i, j, r] * V_reshaped[j, k, r]) * W_reshaped[k, i, r]
    
    # Let's construct intermediate tensor X[j, k, r] ? No.
    # Let's compute Y[i, j, k, r] is too big.
    
    # We can compute:
    # Z[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r]
    # This sums over r.
    # Z = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
    # Shape of Z: (n, m, p).
    # Then T[i, j, k] should be sum_r ... wait.
    # The formula is T[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[k, i].
    # If we sum over r in Z, we lose the r index needed for W.
    # So we cannot sum over r yet.
    
    # We need to sum over r the product of three tensors.
    # T = sum_r U_r[i, j] * V_r[j, k] * W_r[k, i]
    
    # This is:
    # 1. Multiply U and V: Y[i, j, k, r] = U[i, j, r] * V[j, k, r]
    #    This is an outer product of U and V along the r dimension? No, same r.
    #    Y = np.einsum('ijr,jkr->ijk' ??? No, r is kept.
    #    Y[i, j, k, r] = U[i, j, r] * V[j, k, r]
    #    np.einsum('ijr,jkr->ijk' ... no, we need r in output.
    #    np.einsum('ijr,jkr->ijkr' ... no, i, j, k, r.
    #    Actually: U is (n, m, R), V is (m, p, R).
    #    We want Y[i, k, r]?? No, j is shared.
    #    Y[i, j, k, r] = U[i, j, r] * V[j, k, r]
    #    This results in a tensor of size n*m*p*R = 4*5*5*R = 100R. For R=80, 8000 floats. Very small.
    #    Then T[i, j, k] = sum_r Y[i, j, k, r] * W[k, i, r]
    #    T = np.einsum('ijkr, kir -> ijk', Y, W_reshaped)
    
    # This is perfectly feasible and fast.
    
    Y = np.einsum('ijr,jkr->ijkr', U_reshaped, V_reshaped)
    T_reconstructed = np.einsum('ijkr,kir->ijk', Y, W_reshaped)
    
    # Check max error
    max_err = np.max(np.abs(T_reconstructed - 1.0))
    return max_err < tol

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _find_decomposition_greedy(n, m, p, target_rank, t_start, max_time_per_attempt=2.0, num_restarts=100):
    """
    Attempts to find a rank-target_rank decomposition using greedy peeling.
    """
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    for restart in range(num_restarts):
        current_time = time.time()
        if current_time - t_start > DEADLINE_S - 10:
            return None
            
        # Reset residual
        residual = T_target.copy()
        U_list = []
        V_list = []
        W_list = []
        
        success = True
        for r in range(target_rank):
            if time.time() - t_start > DEADLINE_S - 10:
                return None
                
            # Find best rank-1 component to peel
            best_U = None
            best_V = None
            best_W = None
            min_residual_norm = np.sum(residual**2)
            
            # Try multiple random initializations for peeling
            for _ in range(20):
                # Pick a random mode to project
                mode = np.random.choice(['i', 'j', 'k'])
                
                if mode == 'i':
                    u_rand = np.random.randn(n)
                    # M[j, k] = sum_i residual[i, j, k] * u_rand[i]
                    M = np.tensordot(u_rand, residual, axes=[0, 0]) # (m, p)
                    # SVD of M
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                        if len(s_vals) == 0 or s_vals[0] < 1e-10:
                            continue
                        v_comp = U_svd[:, 0]
                        w_comp = Vt_svd[0, :]
                        # We need to scale. M ≈ v_comp * w_comp^T
                        # The component is outer(u_rand, v_comp, w_comp)
                        # To maximize reduction, we should scale v_comp and w_comp?
                        # The subtraction is residual - u_rand[:, None, None] * v_comp[None, :, None] * w_comp[None, None, :]
                        # We already have the direction. The magnitude is fixed by the SVD of the projection.
                        # Actually, the best rank-1 approx of M is s0 * u_svd[:,0] * v_svd[0,:]
                        # So v_comp = s_vals[0] ** 0.5 * U_svd[:, 0]
                        # w_comp = s_vals[0] ** 0.5 * Vt_svd[0, :]
                        # This makes v_comp * w_comp^T = M_approx.
                        v_comp = U_svd[:, 0] * np.sqrt(s_vals[0])
                        w_comp = Vt_svd[0, :] * np.sqrt(s_vals[0])
                        
                    except np.linalg.LinAlgError:
                        continue
                    
                    u_comp = u_rand
                    # Normalize? No, the projection determines the scale relative to u_rand.
                    
                elif mode == 'j':
                    v_rand = np.random.randn(m)
                    M = np.tensordot(v_rand, residual, axes=[0, 1]) # (n, p)
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                        if len(s_vals) == 0 or s_vals[0] < 1e-10:
                            continue
                        u_comp = U_svd[:, 0] * np.sqrt(s_vals[0])
                        w_comp = Vt_svd[0, :] * np.sqrt(s_vals[0])
                        v_comp = v_rand
                    except np.linalg.LinAlgError:
                        continue
                        
                else: # mode == 'k'
                    w_rand = np.random.randn(p)
                    M = np.tensordot(w_rand, residual, axes=[0, 2]) # (n, m)
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                        if len(s_vals) == 0 or s_vals[0] < 1e-10:
                            continue
                        u_comp = U_svd[:, 0] * np.sqrt(s_vals[0])
                        v_comp = Vt_svd[0, :] * np.sqrt(s_vals[0])
                        w_comp = w_rand
                    except np.linalg.LinAlgError:
                        continue
                    
                # Construct the rank-1 tensor
                comp_tensor = np.einsum('i,j,k->ijk', u_comp, v_comp, w_comp)
                
                new_residual = residual - comp_tensor
                new_norm = np.sum(new_residual**2)
                
                if new_norm < min_residual_norm:
                    min_residual_norm = new_norm
                    best_U = u_comp
                    best_V = v_comp
                    best_W = w_comp
                    best_res = new_residual
            
            if best_U is None or min_residual_norm >= np.sum(residual**2) * 0.99:
                success = False
                break
                
            residual = best_res
            U_list.append(best_U)
            V_list.append(best_V)
            W_list.append(best_W)
            
        if success:
            if np.max(np.abs(residual)) < 1e-8:
                # Build the final matrices
                U_full = np.zeros((n * m, target_rank))
                V_full = np.zeros((m * p, target_rank))
                W_full = np.zeros((p * n, target_rank))
                
                for r in range(target_rank):
                    u_r = U_list[r]
                    v_r = V_list[r]
                    w_r = W_list[r]
                    
                    for i in range(n):
                        for j in range(m):
                            U_full[i * m + j, r] = u_r[i]
                    for j in range(m):
                        for k in range(p):
                            V_full[j * p + k, r] = v_r[j]
                    for k in range(p):
                        for i in range(n):
                            W_full[k * n + i, r] = w_r[k]
                
                # Validate
                if _validate_decomposition(U_full, V_full, W_full, n, m, p):
                    return (U_full, V_full, W_full)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # We try to find a better rank.
    # Known bounds for 4x5x5 are not widely publicized as having a very small rank.
    # However, random search for rank ~ 80-90 is unlikely to succeed quickly.
    # Let's try a few higher ranks first to see if we can improve slightly.
    # If we don't find anything better quickly, we return schoolbook.
    
    # We attempt ranks from 90 down to 85? Or just a few tries at 95?
    # Given the time constraint, extensive random search for a specific rank is risky.
    # A more reliable approach is to just return the schoolbook if we don't have a specific
    # algebraic construction in mind.
    
    # However, the prompt asks to rewrite for a stronger algorithm.
    # If we can't find a lower rank, returning schoolbook is the safe bet.
    
    # Let's try a very limited search for a slightly lower rank, e.g., 95.
    # If it fails, we proceed.
    
    search_ranks = [90, 85, 80]
    
    for target_r in search_ranks:
        if time.time() - t0 > DEADLINE_S - 20:
            break
        
        result = _find_decomposition_greedy(n, m, p, target_r, t0, max_time_per_attempt=1.0, num_restarts=50)
        if result is not None:
            U_new, V_new, W_new = result
            best_U = U_new
            best_V = V_new
            best_W = W_new
            best_rank = target_r
            break  # Stop if we find a better one
            
    # If no better rank found, use schoolbook
    U_best = best_U
    V_best = best_V
    W_best = best_W
    
    # Convert to lists with integer cleanup
    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xv = float(x)
                if abs(xv - round(xv)) < 1e-6:
                    xv = int(round(xv))
                lrow.append(xv)
            out.append(lrow)
        return out

    U_final = _to_list(np.array(U_best))
    V_final = _to_list(np.array(V_best))
    W_final = _to_list(np.array(W_best))
    
    return {
        "U": U_final,
        "V": V_final,
        "W": W_final
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
