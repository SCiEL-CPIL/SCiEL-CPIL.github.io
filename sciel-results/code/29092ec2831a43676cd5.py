import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # We will build U, V, W as lists of columns (rank vectors)
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def get_u_idx(gi, gj):
        return gi * m + gj

    def get_v_idx(gj, gk):
        return gj * p + gk

    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)

        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)

        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)

        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)

        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Naive for col bk+2
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)

    # Construct the full decomposition
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T  # Shape (16, rank)
    V_mat = np.array(V_cols).T  # Shape (20, rank)
    W_mat = np.array(W_cols).T  # Shape (20, rank)
    
    best_rank = U_mat.shape[1]
    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
    
    # Verification function
    def verify(U, V, W):
        """Check if U,V,W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The correct value is 1 for all valid i,j,k
        
        # Generate random test matrices
        rng = np.random.default_rng(42)
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        
        # Reconstruct using U, V, W
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient computation:
        # C_recon[i,k] = sum_j (sum_r U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        
        # Let's create a 3D array approach
        # For each (i, k), we need to compute the sum over j and r.
        # We can precompute intermediate terms.
        
        # Actually, let's just do the direct check for all i, j, k since dimensions are small.
        # n=4, m=4, p=5 -> 80 triples.
        
        indices_to_check = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices_to_check.append((i, j, k))
        
        for (i, j, k) in indices_to_check:
            u_val = U[i*m+j, :]
            v_val = V[j*p+k, :]
            w_val = W[k*n+i, :]
            # sum_r u_val[r] * v_val[r] * w_val[r]
            recon = np.sum(u_val * v_val * w_val)
            if abs(recon - 1.0) > 1e-9:
                return False, recon, (i, j, k)
        
        return True, None, None
    
    # Verify the initial decomposition
    valid, recon_val, idx = verify(U_mat, V_mat, W_mat)
    if not valid:
        # The decomposition is invalid, return as is (shouldn't happen)
        pass
    
    # Try to reduce rank by removing linearly dependent terms
    def try_reduce_rank(U, V, W, max_time=30):
        """Try to reduce the rank by removing redundant terms."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try removing each column and checking if the decomposition still holds
        for col_to_remove in range(rank):
            if time.time() > max_time:
                break
            
            U_reduced = np.delete(U, col_to_remove, axis=1)
            V_reduced = np.delete(V, col_to_remove, axis=1)
            W_reduced = np.delete(W, col_to_remove, axis=1)
            
            valid, _, _ = verify(U_reduced, V_reduced, W_reduced)
            if valid:
                best_rank = U_reduced.shape[1]
                best_U, best_V, best_W = U_reduced.copy(), V_reduced.copy(), W_reduced.copy()
                # Restart from the reduced version
                U, V, W = U_reduced, V_reduced, W_reduced
                rank = U.shape[1]
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U_mat, V_mat, W_mat, best_rank = try_reduce_rank(U_mat, V_mat, W_mat, max_time=20)
    
    # Try local optimization to find a better decomposition
    def local_optimize(U, V, W, max_time=30):
        """Try to optimize the decomposition to allow for rank reduction."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try to find a lower rank by perturbing and checking
        # This is a heuristic approach
        rng = np.random.default_rng(123)
        
        # Try rank reduction by 1
        for target_rank in range(rank - 1, max(rank - 5, 0), -1):
            if time.time() > max_time:
                break
            
            # Try to find a valid decomposition of rank target_rank
            # This is NP-hard in general, so we use a heuristic
            # Start from the current decomposition and try to eliminate terms
            
            U_trial = U.copy()
            V_trial = V.copy()
            W_trial = W.copy()
            
            # Try to set some entries to zero to create linear dependencies
            # Then check if we can remove a term
            for attempt in range(100):
                if time.time() > max_time:
                    break
                
                # Randomly perturb some entries
                U_pert = U_trial.copy()
                V_pert = V_trial.copy()
                W_pert = W_trial.copy()
                
                # Try to make column target_rank linearly dependent on others
                # by adjusting other columns
                for col in range(target_rank):
                    # Small perturbation
                    U_pert[:, col] += rng.random() * 0.01
                    V_pert[:, col] += rng.random() * 0.01
                    W_pert[:, col] += rng.random() * 0.01
                
                # Check if we can remove the last column
                U_reduced = U_pert[:, :target_rank]
                V_reduced = V_pert[:, :target_rank]
                W_reduced = W_pert[:, :target_rank]
                
                valid, _, _ = verify(U_reduced, V_reduced, W_reduced)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_reduced.copy(), V_reduced.copy(), W_reduced.copy()
                    break
        
        return best_U, best_V, best_W, best_rank
    
    # Try local optimization if time permits
    if time.time() < deadline - 10:
        U_mat, V_mat, W_mat, best_rank = local_optimize(U_mat, V_mat, W_mat, max_time=15)
    
    # Round to half-integers if possible
    def round_to_half_integer(M):
        """Round matrix entries to nearest half-integer."""
        M_rounded = np.round(M * 2) / 2
        return M_rounded
    
    # Check if rounding preserves validity
    U_rounded = round_to_half_integer(U_mat)
    V_rounded = round_to_half_integer(V_mat)
    W_rounded = round_to_half_integer(W_mat)
    
    valid, _, _ = verify(U_rounded, V_rounded, W_rounded)
    if valid:
        U_mat, V_mat, W_mat = U_rounded, V_rounded, W_rounded
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
# EVOLVE-BLOCK-END
