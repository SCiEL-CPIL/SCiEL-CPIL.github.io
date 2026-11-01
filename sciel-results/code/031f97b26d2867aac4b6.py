"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the multiplication tensor
    # T is a 3-way tensor of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exist i,j,k such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # and the indices are consistent (same j in a and b, same k in b and c, same i in a and c)
    
    U_dim = n * m  # 6
    V_dim = m * p  # 9
    W_dim = p * n  # 6
    
    # Build tensor as list of (a, b, c) indices with value 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Try to find rank-r decomposition for r < 18
    # Use ALS with random initialization
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Reconstruct tensor: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check all entries
        for (a, b, c) in tensor_entries:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > tol:
                return False
        # Check that non-tensor entries are ~0
        for a in range(U_dim):
            for b in range(V_dim):
                # Check if this (a,b) pair is valid (consistent j)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                if j_a != j_b:
                    continue  # This entry should be 0, but it's structurally 0
                # For consistent (a,b), check all c
                for c in range(W_dim):
                    # Check if (a,b,c) is a tensor entry
                    is_entry = False
                    for (a2, b2, c2) in tensor_entries:
                        if a2 == a and b2 == b and c2 == c:
                            is_entry = True
                            break
                    if not is_entry:
                        val = 0.0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val) > tol:
                            return False
        return True
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx, (a, b, c) in enumerate(tensor_entries):
        U_school[a, idx] = 1.0
        V_school[b, idx] = 1.0
        W_school[c, idx] = 1.0
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ALS for lower ranks
    np.random.seed(42)
    
    for target_rank in range(17, 1, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        n_trials = 50
        found = False
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(U_dim, target_rank) * 0.5
            V = np.random.randn(V_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
                # This is a linear system in U[a,:]
                # Let X_a be the matrix where X_a[b_idx, r] = V[b,r] * W[c,r]... 
                # Actually, for fixed a, we need:
                # For all (b,c) where T[a,b,c] is defined:
                #   sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                
                # Build the system for U[a]
                # Collect all (b,c) pairs for this a
                bc_pairs = []
                for (a2, b2, c2) in tensor_entries:
                    if a2 == a:
                        bc_pairs.append((b2, c2))
                
                if not bc_pairs:
                    continue
                
                # X matrix: rows are (b,c) pairs, cols are r
                # X[row, r] = V[b,r] * W[c,r]
                # y[row] = 1.0 (since all tensor entries are 1)
                X = np.zeros((len(bc_pairs), target_rank))
                y = np.ones(len(bc_pairs))
                for row_idx, (b2, c2) in enumerate(bc_pairs):
                    X[row_idx, :] = V[b2, :] * W[c2, :]
                
                # Solve X * U[a,:] = y (least squares)
                try:
                    U[a, :], residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                except:
                    break
                
                # Update V similarly
                for b in range(V_dim):
                    ac_pairs = []
                    for (a2, b2, c2) in tensor_entries:
                        if b2 == b:
                            ac_pairs.append((a2, c2))
                    if not ac_pairs:
                        continue
                    X = np.zeros((len(ac_pairs), target_rank))
                    y = np.ones(len(ac_pairs))
                    for row_idx, (a2, c2) in enumerate(ac_pairs):
                        X[row_idx, :] = U[a2, :] * W[c2, :]
                    try:
                        V[b, :], residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                    except:
                        break
                
                # Update W
                for c in range(W_dim):
                    ab_pairs = []
                    for (a2, b2, c2) in tensor_entries:
                        if c2 == c:
                            ab_pairs.append((a2, b2))
                    if not ab_pairs:
                        continue
                    X = np.zeros((len(ab_pairs), target_rank))
                    y = np.ones(len(ab_pairs))
                    for row_idx, (a2, b2) in enumerate(ab_pairs):
                        X[row_idx, :] = U[a2, :] * V[b2, :]
                    try:
                        W[c, :], residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
                    except:
                        break
                
                # Check convergence
                if verify_decomposition(U, V, W, tol=1e-8):
                    found = True
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break
            
            if found:
                break
        
        if found:
            # Found a decomposition of this rank, try lower
            continue
        else:
            # Could not find decomposition of this rank, try next lower
            continue
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5
    def snap_to_half(x):
        return np.round(x * 2) / 2
    
    U_out = snap_to_half(best_U)
    V_out = snap_to_half(best_V)
    W_out = snap_to_half(best_W)
    
    # Verify after snapping
    if not verify_decomposition(U_out, V_out, W_out, tol=1e-6):
        # Use un-snapped version
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to list of lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
