"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    deadline = time.time() + 145  # Safety margin before 150s kill

    # Build target tensor indices
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
    
    # Precompute masks for each index in U, V, W
    # mask_U[a] contains indices in [0, n_entries) where a_idx == a
    mask_U = [np.where(a_idx == a)[0] for a in range(nm)]
    mask_V = [np.where(b_idx == b)[0] for b in range(mp)]
    mask_W = [np.where(c_idx == c)[0] for c in range(pn)]

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
        """Check if decomposition is valid."""
        # Check non-zero entries
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries (sample)
        # To be safe, check all zero entries if rank is small, otherwise sample
        # There are 16*28*28 - 112 = 12432 zero entries
        # Checking all is 12432 dot products of length R.
        # R~100. 12432 * 100 = 1.2M ops. Fast enough.
        # But let's sample 200 random zero entries to save time during search.
        # If it passes sample, do full check at the end.
        
        checked = set()
        for _ in range(200):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a non-zero entry
            # Efficient check:
            # a = i*m+j, b = j*p+k, c = k*n+i
            # i = a // m, j = a % m
            # j' = b // p, k = b % p
            # k' = c // n, i' = c % n
            # Non-zero if j == j' and k == k' and i == i'
            i, j = divmod(a, m)
            j_p, k = divmod(b, p)
            k_n, i_p = divmod(c, n)
            
            if j == j_p and k == k_n and i == i_p:
                continue # This is a non-zero entry, skip for zero check
            
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > tol * 10:
                return False
        return True

    def check_validity_full(U, V, W, tol=1e-9):
        """Full check for final verification."""
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        
        # Iterate all possible a,b,c and check if zero
        # We can do this more efficiently by iterating over all a,b,c
        # but skipping the known non-zeros.
        # Since nm, mp, pn are small (16, 28, 28), we can just iterate.
        # 16 * 28 * 28 = 12544 iterations.
        # Each is a dot product of length R.
        # This might be slow if R is large and we do it many times.
        # But we only do it when we think we found a solution.
        
        # Create a set of non-zero tuples for fast lookup
        nz_set = set(zip(a_idx, b_idx, c_idx))
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in nz_set:
                        continue
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
        return True

    def als_iteration(U, V, W, rank, max_iter=500):
        """Run ALS iterations."""
        T_flat = np.ones(n_entries)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U
            U_new = np.zeros((nm, rank))
            for a in range(nm):
                mask = mask_U[a]
                if len(mask) == 0:
                    continue
                # M_U[r, idx] = V[b_idx[idx], r] * W[c_idx[idx], r]
                # We need M_a = M_U[:, mask]
                # M_a[r, k] = V[b_idx[mask[k]], r] * W[c_idx[mask[k]], r]
                # Let B_mask = b_idx[mask], C_mask = c_idx[mask]
                # M_a = V[B_mask].T * W[C_mask].T ? No.
                # V[B_mask] is (n_a, rank). W[C_mask] is (n_a, rank).
                # M_a is (rank, n_a).
                # M_a[r, k] = V[B_mask[k], r] * W[C_mask[k], r]
                # This is element-wise product of V[B_mask].T and W[C_mask].T
                B_m = b_idx[mask]
                C_m = c_idx[mask]
                M_a = (V[B_m].T * W[C_m].T) # (rank, n_a)
                
                # Solve U[a, :] @ M_a = T_flat[mask]
                # U[a, :] = T_flat[mask] @ pinv(M_a)
                try:
                    if M_a.shape[1] >= rank:
                        U_new[a], _, _, _ = np.linalg.lstsq(M_a.T, T_flat[mask], rcond=None)
                    else:
                        U_new[a] = np.linalg.pinv(M_a.T) @ T_flat[mask]
                except:
                    U_new[a] = U[a]
            U = U_new
            
            # Update V
            V_new = np.zeros((mp, rank))
            for b in range(mp):
                mask = mask_V[b]
                if len(mask) == 0:
                    continue
                A_m = a_idx[mask]
                C_m = c_idx[mask]
                M_b = (U[A_m].T * W[C_m].T) # (rank, n_b)
                try:
                    if M_b.shape[1] >= rank:
                        V_new[b], _, _, _ = np.linalg.lstsq(M_b.T, T_flat[mask], rcond=None)
                    else:
                        V_new[b] = np.linalg.pinv(M_b.T) @ T_flat[mask]
                except:
                    V_new[b] = V[b]
            V = V_new
            
            # Update W
            W_new = np.zeros((pn, rank))
            for c in range(pn):
                mask = mask_W[c]
                if len(mask) == 0:
                    continue
                A_m = a_idx[mask]
                B_m = b_idx[mask]
                M_c = (U[A_m].T * V[B_m].T) # (rank, n_c)
                try:
                    if M_c.shape[1] >= rank:
                        W_new[c], _, _, _ = np.linalg.lstsq(M_c.T, T_flat[mask], rcond=None)
                    else:
                        W_new[c] = np.linalg.pinv(M_c.T) @ T_flat[mask]
                except:
                    W_new[c] = W[c]
            W = W_new
            
            # Early stopping check
            if it % 20 == 0:
                # Compute error on non-zero entries
                err = 0.0
                for idx in range(n_entries):
                    a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
                    val = np.dot(U[a], V[b] * W[c])
                    err += (val - 1.0)**2
                if err < 1e-10:
                    break
        
        return U, V, W

    # Try decreasing ranks
    # Start from 112 down to 90
    # If we find a valid solution, we can try to go lower
    found_better = False
    for target_rank in range(111, 89, -1):
        if time.time() > deadline - 20:
            break
        
        n_restarts = 10
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize
            if restart == 0:
                # Warm start from schoolbook
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
                # Add small noise to break symmetry
                U += np.random.randn(nm, target_rank) * 0.01
                V += np.random.randn(mp, target_rank) * 0.01
                W += np.random.randn(pn, target_rank) * 0.01
            else:
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, target_rank, max_iter=300)
            
            # Check validity
            if check_validity(U, V, W):
                # Full check
                if check_validity_full(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            found_better = True
            # We can continue to search for even lower ranks, but we have a valid one
            # The loop continues to target_rank - 1
        else:
            # If we failed to find a solution at this rank, it might be impossible
            # or we just didn't find it. 
            # Given the structure, if 111 is impossible, 110 might be too.
            # But let's try a few more.
            # If we fail 3 ranks in a row, we might stop.
            # For now, just continue.
            pass

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
