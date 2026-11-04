"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build target tensor indices (sparse representation)
    # T[i*m+j, j*p+k, k*n+i] = 1
    a_idx = []
    b_idx = []
    c_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx.append(i * m + j)
                b_idx.append(j * p + k)
                c_idx.append(k * n + i)
    a_idx = np.array(a_idx)
    b_idx = np.array(b_idx)
    c_idx = np.array(c_idx)
    n_entries = len(a_idx)  # 112

    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_validity(U, V, W, tol=1e-9):
        """Check if decomposition is valid by verifying all tensor entries."""
        # Check all nonzero entries
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check zero entries
        nonzero_set = set()
        for idx in range(n_entries):
            nonzero_set.add((a_idx[idx], b_idx[idx], c_idx[idx]))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nonzero_set:
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > tol * 10:
                            return False
        return True

    def compute_error(U, V, W):
        """Compute max error on nonzero entries."""
        max_err = 0.0
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err

    def peeling_decompose(max_rank=100, time_budget=120):
        """
        Peeling algorithm: iteratively extract rank-1 components.
        At each step, find a rank-1 component (u, v, w) such that
        the residual tensor T - u⊗v⊗w has lower rank.
        """
        start_time = time.time()
        
        # Build full tensor as sparse
        T = np.zeros((nm, mp, pn))
        for idx in range(n_entries):
            T[a_idx[idx], b_idx[idx], c_idx[idx]] = 1.0
        
        # Factor vectors collected
        U_factors = []
        V_factors = []
        W_factors = []
        
        rank = 0
        while rank < max_rank:
            if time.time() - start_time > time_budget:
                break
            
            # Check if residual is zero
            if np.allclose(T, 0, atol=1e-10):
                break
            
            # Try to find a rank-1 component
            # Strategy: pick a random (a,b) pair with T[a,b,:] nonzero
            # Then solve for w such that T[a,b,:] = w * (u_a * v_b)
            # and u_a, v_b are scalars from the factor vectors
            
            # Better strategy: use the structure of matrix multiplication
            # For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1
            # A rank-1 term contributes: U[i*m+j] * V[j*p+k] * W[k*n+i]
            
            # Pick a specific (i,j,k) entry to anchor
            # We want to find u, v, w such that u[a]*v[b]*w[c] matches T[a,b,c]
            
            # Use iterative approach: fix one factor, solve for others
            # Try multiple random initializations
            
            found = False
            for attempt in range(50):
                if time.time() - start_time > time_budget:
                    break
                
                # Random initialization of one factor
                # Try to extract by looking at the support pattern
                
                # Pick a random nonzero entry (a0, b0, c0)
                idx0 = np.random.randint(0, n_entries)
                a0, b0, c0 = a_idx[idx0], b_idx[idx0], c_idx[idx0]
                
                # For matrix multiplication tensor:
                # a0 = i0*m + j0, b0 = j0*p + k0, c0 = k0*n + i0
                i0 = a0 // m
                j0 = a0 % m
                k0 = b0 % p
                # Verify: c0 should be k0*n + i0
                assert c0 == k0 * n + i0, f"Index mismatch: {c0} != {k0*n+i0}"
                
                # The rank-1 component must satisfy:
                # u[i*m+j] * v[j*p+k] * w[k*n+i] contributes to T[i*m+j, j*p+k, k*n+i]
                
                # For a rank-1 term to be consistent with the multiplication tensor,
                # we need: for all (i,j,k), u[i*m+j]*v[j*p+k]*w[k*n+i] = T[i*m+j, j*p+k, k*n+i]
                # But T is not rank 1, so we're looking for a rank-1 component that
                # "peels off" part of the tensor.
                
                # Alternative: use the fact that for matrix multiplication,
                # we can try to find u, v, w such that the product matches
                # a subset of entries.
                
                # Let's try a different approach: 
                # For the multiplication tensor, a rank-1 component corresponds to
                # choosing specific "lines" in the three modes.
                
                # Actually, let's use a more systematic peeling:
                # 1. Pick a random (i, j) pair
                # 2. The slice T[i*m+j, :, :] has nonzeros at positions j*p+k for k in range(p)
                # 3. For a rank-1 component u⊗v⊗w, the slice at a=i*m+j is u[a]*v⊗w
                # 4. So v⊗w must be proportional to the slice pattern
                
                # Pick random i, j
                i_rand = np.random.randint(0, n)
                j_rand = np.random.randint(0, m)
                a_rand = i_rand * m + j_rand
                
                # The slice T[a_rand, :, :] has nonzeros at b = j_rand*p + k for k in range(p)
                # With values 1
                slice_T = T[a_rand, :, :].copy()
                if np.allclose(slice_T, 0, atol=1e-10):
                    continue
                
                # For rank-1: slice_T[b, c] = u[a_rand] * v[b] * w[c]
                # Nonzero entries of slice_T are at (b = j_rand*p+k, c = k*n+i_rand) for k in range(p)
                # So v[j_rand*p+k] * w[k*n+i_rand] = 1/u[a_rand] for all k
                
                # This gives us constraints. Let's set u[a_rand] = 1 (wlog)
                # Then v[j_rand*p+k] * w[k*n+i_rand] = 1 for all k in range(p)
                
                # We have p equations with 2p unknowns (v values and w values)
                # Plus the constraint that these must be consistent with other slices.
                
                # Let's parameterize: let v[j_rand*p+k] = alpha_k, w[k*n+i_rand] = 1/alpha_k
                # Choose alpha_k randomly (or all 1s)
                
                alpha = np.ones(p)
                # Perturb for diversity
                if attempt > 0:
                    alpha = np.random.randn(p)
                    alpha[alpha < 0.1] = 0.1  # keep positive for simplicity
                
                # Initialize u, v, w
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                
                u[a_rand] = 1.0
                for k in range(p):
                    v[j_rand * p + k] = alpha[k]
                    w[k * n + i_rand] = 1.0 / alpha[k]
                
                # Now extend to other entries using the tensor structure
                # For entry (i*m+j, j*p+k, k*n+i) = 1:
                # u[i*m+j] * v[j*p+k] * w[k*n+i] should contribute 1
                # But we're peeling, so we want this rank-1 to match a subset.
                
                # Actually, for proper peeling, the rank-1 component should match
                # exactly a subset of the tensor entries (where T=1) and be zero elsewhere.
                
                # Let's check: for the current u, v, w, what entries are nonzero?
                # u[a] nonzero only at a = i_rand*m + j_rand
                # v[b] nonzero only at b = j_rand*p + k for k in range(p)
                # w[c] nonzero only at c = k*n + i_rand for k in range(p)
                
                # So the rank-1 tensor u⊗v⊗w is nonzero at:
                # (a, b, c) where a = i_rand*m+j_rand, b = j_rand*p+k, c = k*n+i_rand
                # These are exactly the p entries: (i_rand*m+j_rand, j_rand*p+k, k*n+i_rand) for k in range(p)
                
                # Check if all these entries have T=1
                valid = True
                for k in range(p):
                    a_check = i_rand * m + j_rand
                    b_check = j_rand * p + k
                    c_check = k * n + i_rand
                    if T[a_check, b_check, c_check] != 1.0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check that u⊗v⊗w is zero where T is zero
                # The nonzero entries of u⊗v⊗w are exactly the p entries above
                # We need T=1 at all of them (checked) and T=0 elsewhere for u⊗v⊗w
                # Since u, v, w have specific supports, u⊗v⊗w is zero elsewhere
                # So this is automatically satisfied!
                
                # Great! We found a valid rank-1 component
                # Subtract it from T
                for k in range(p):
                    a_sub = i_rand * m + j_rand
                    b_sub = j_rand * p + k
                    c_sub = k * n + i_rand
                    T[a_sub, b_sub, c_sub] -= 1.0
                
                U_factors.append(u)
                V_factors.append(v)
                W_factors.append(w)
                rank += 1
                found = True
                break
            
            if not found:
                # Try a different strategy: look for a (i, j) where the slice
                # has a simpler pattern
                found_any = False
                for i_try in range(n):
                    for j_try in range(m):
                        if time.time() - start_time > time_budget:
                            break
                        a_try = i_try * m + j_try
                        slice_T_try = T[a_try, :, :]
                        if np.allclose(slice_T_try, 0, atol=1e-10):
                            continue
                        
                        # Check if the nonzero pattern is consistent with
                        # v[j_try*p+k] * w[k*n+i_try] for some v, w
                        # Nonzero entries should be at (j_try*p+k, k*n+i_try)
                        
                        nonzero_positions = []
                        for k in range(p):
                            b_pos = j_try * p + k
                            c_pos = k * n + i_try
                            if abs(slice_T_try[b_pos, c_pos]) > 1e-10:
                                nonzero_positions.append((k, b_pos, c_pos))
                        
                        if len(nonzero_positions) == 0:
                            continue
                        
                        # Check no other nonzeros
                        has_other_nonzero = False
                        for b in range(mp):
                            for c in range(pn):
                                is_expected = False
                                for k, b_exp, c_exp in nonzero_positions:
                                    if b == b_exp and c == c_exp:
                                        is_expected = True
                                        break
                                if not is_expected and abs(slice_T_try[b, c]) > 1e-10:
                                    has_other_nonzero = True
                                    break
                            if has_other_nonzero:
                                break
                        
                        if has_other_nonzero:
                            continue
                        
                        # All nonzero values should be equal (for rank-1 with u[a]=1)
                        vals = [slice_T_try[b_pos, c_pos] for k, b_pos, c_pos in nonzero_positions]
                        if not np.allclose(vals, vals[0], atol=1e-10):
                            continue
                        
                        # Scale: u[a_try] = 1/vals[0]
                        u = np.zeros(nm)
                        v = np.zeros(mp)
                        w = np.zeros(pn)
                        
                        u[a_try] = 1.0 / vals[0]
                        for k, b_pos, c_pos in nonzero_positions:
                            v[b_pos] = 1.0
                            w[c_pos] = 1.0
                        
                        # Subtract
                        for k, b_pos, c_pos in nonzero_positions:
                            T[a_try, b_pos, c_pos] -= vals[0]
                        
                        U_factors.append(u)
                        V_factors.append(v)
                        W_factors.append(w)
                        rank += 1
                        found_any = True
                        break
                    if found_any:
                        break
                    if time.time() - start_time > time_budget:
                        break
                
                if not found_any:
                    break
        
        if rank == 0:
            return None
        
        # Build factor matrices
        U_mat = np.column_stack(U_factors)
        V_mat = np.column_stack(V_factors)
        W_mat = np.column_stack(W_factors)
        
        return U_mat, V_mat, W_mat, rank

    # Try peeling first
    peel_result = peeling_decompose(max_rank=100, time_budget=100)
    if peel_result is not None:
        U_peel, V_peel, W_peel, rank_peel = peel_result
        if check_validity(U_peel, V_peel, W_peel):
            best_U, best_V, best_W = U_peel, V_peel, W_peel
            best_rank = rank_peel
            # Convert to lists and return
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to ALS if peeling fails
    def build_design_matrices(U, V, W, rank):
        """Build sparse design matrices for each mode update."""
        M_U_rows = []
        M_U_cols = []
        M_U_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                b, c = b_idx[idx], c_idx[idx]
                val = V[b, r] * W[c, r]
                if abs(val) > 1e-15:
                    M_U_rows.append(r)
                    M_U_cols.append(idx)
                    M_U_vals.append(val)
        
        M_V_rows = []
        M_V_cols = []
        M_V_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                a, c = a_idx[idx], c_idx[idx]
                val = U[a, r] * W[c, r]
                if abs(val) > 1e-15:
                    M_V_rows.append(r)
                    M_V_cols.append(idx)
                    M_V_vals.append(val)
        
        M_W_rows = []
        M_W_cols = []
        M_W_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                a, b = a_idx[idx], b_idx[idx]
                val = U[a, r] * V[b, r]
                if abs(val) > 1e-15:
                    M_W_rows.append(r)
                    M_W_cols.append(idx)
                    M_W_vals.append(val)
        
        M_U = csr_matrix((M_U_vals, (M_U_rows, M_U_cols)), shape=(rank, n_entries))
        M_V = csr_matrix((M_V_vals, (M_V_rows, M_V_cols)), shape=(rank, n_entries))
        M_W = csr_matrix((M_W_vals, (M_W_rows, M_W_cols)), shape=(rank, n_entries))
        
        return M_U, M_V, M_W

    def als_iteration(U, V, W, rank, M_U, M_V, M_W, max_iter=200):
        """Run ALS iterations."""
        T_flat = np.ones(n_entries)
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            U_new = np.zeros((nm, rank))
            for a in range(nm):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                T_a = T_flat[mask]
                M_a = M_U[:, mask]
                try:
                    if M_a.shape[1] >= rank:
                        U_new[a], _, _, _ = np.linalg.lstsq(M_a.T, T_a, rcond=None)
                    else:
                        U_new[a] = np.linalg.pinv(M_a.T) @ T_a
                except:
                    U_new[a] = U[a]
            U = U_new
            
            V_new = np.zeros((mp, rank))
            for b in range(mp):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                T_b = T_flat[mask]
                M_b = M_V[:, mask]
                try:
                    if M_b.shape[1] >= rank:
                        V_new[b], _, _, _ = np.linalg.lstsq(M_b.T, T_b, rcond=None)
                    else:
                        V_new[b] = np.linalg.pinv(M_b.T) @ T_b
                except:
                    V_new[b] = V[b]
            V = V_new
            
            W_new = np.zeros((pn, rank))
            for c in range(pn):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                T_c = T_flat[mask]
                M_c = M_W[:, mask]
                try:
                    if M_c.shape[1] >= rank:
                        W_new[c], _, _, _ = np.linalg.lstsq(M_c.T, T_c, rcond=None)
                    else:
                        W_new[c] = np.linalg.pinv(M_c.T) @ T_c
                except:
                    W_new[c] = W[c]
            W = W_new
            
            if it % 10 == 0:
                err = compute_error(U, V, W)
                if err < 1e-10:
                    break
        
        return U, V, W

    for target_rank in range(80, rank_school):
        if time.time() > deadline - 20:
            break
        
        n_restarts = 3 if target_rank < 90 else 2
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart == 0:
                if best_rank == target_rank + 1:
                    U = best_U[:, :target_rank] + np.random.randn(nm, target_rank) * 0.1
                    V = best_V[:, :target_rank] + np.random.randn(mp, target_rank) * 0.1
                    W = best_W[:, :target_rank] + np.random.randn(pn, target_rank) * 0.1
                else:
                    U = np.random.randn(nm, target_rank) * 0.5
                    V = np.random.randn(mp, target_rank) * 0.5
                    W = np.random.randn(pn, target_rank) * 0.5
            else:
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            M_U, M_V, M_W = build_design_matrices(U, V, W, target_rank)
            U, V, W = als_iteration(U, V, W, target_rank, M_U, M_V, M_W, max_iter=150)
            
            if check_validity(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
        else:
            break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
