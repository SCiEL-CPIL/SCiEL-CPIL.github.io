"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    U_dim = n * m  # 6
    V_dim = m * p  # 9
    W_dim = p * n  # 6
    
    # Build the target tensor T as a dictionary of non-zero entries
    # T[a, b, c] = 1 if indices are consistent
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0

    def verify(U, V, W, tol=1e-6):
        rank = U.shape[1]
        # Check all target entries
        for (a, b, c), val in target.items():
            s = 0.0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s - val) > tol:
                return False
        
        # Check that non-target entries are ~0
        # We only need to check entries that are not in target but could be non-zero
        # due to the structure of the sum.
        # Actually, if we check all (a,b,c) in the full box, it's 6*9*6 = 324 checks.
        # This is fast enough.
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    if (a, b, c) in target:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
        return True

    # Schoolbook decomposition (Rank 18)
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx, ((a, b, c), val) in enumerate(target.items()):
        U_school[a, idx] = val
        V_school[b, idx] = val
        W_school[c, idx] = val

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find Rank 17
    # Strategy: Random search with local refinement
    # We represent the decomposition by U, V, W matrices.
    # We will use a "coordinate descent" like approach on the flattened parameters,
    # but with snapping to half-integers.
    
    target_rank = 17
    np.random.seed(42)
    
    # Pre-compute indices for fast access
    a_indices = np.array([a for (a,b,c) in target.keys()])
    b_indices = np.array([b for (a,b,c) in target.keys()])
    c_indices = np.array([c for (a,b,c) in target.keys()])
    
    def compute_error(U, V, W):
        # Compute sum for all target entries
        # U[a, r], V[b, r], W[c, r]
        # We can vectorize this:
        # For each r, we have vectors u_r, v_r, w_r.
        # The contribution to T is outer(u_r, v_r, w_r).
        # Sum over r.
        # Efficient way:
        # Reshape U, V, W to (dim, rank)
        # For each target entry (a, b, c), error = sum_r U[a,r]V[b,r]W[c,r] - 1
        # This is a dot product of (U[a,:], V[b,:]*W[c,:])
        
        # Let's just do a loop for clarity, it's small (18 entries * 17 rank)
        err = 0.0
        for idx in range(len(target)):
            a = a_indices[idx]
            b = b_indices[idx]
            c = c_indices[idx]
            # Vectorized dot product
            dot = np.dot(U[a, :], V[b, :] * W[c, :])
            err += (dot - 1.0)**2
        
        # Also check non-target entries to prevent "leakage"
        # We sample some non-target entries to keep the gradient pointing towards 0
        # There are 324 - 18 = 306 non-target entries.
        # Checking all is 306 * 17 ops. Very fast.
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    if (a, b, c) in target:
                        continue
                    dot = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += dot**2
        return err

    def refine(U, V, W):
        # Local search: try to snap to half-integers and fix errors
        # We will iterate:
        # 1. Snap current U, V, W to nearest half-integers
        # 2. Calculate error
        # 3. If error is small, verify
        # 4. If not, try to adjust one vector to reduce error
        
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Check if valid
        if verify(U, V, W):
            return U, V, W, True
        
        # If not, try local moves
        # This is a heuristic
        return U, V, W, False

    # Main Search Loop for Rank 17
    found = False
    max_trials = 500
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook, remove one random term, and add noise
        # Or start random
        if trial % 2 == 0:
            # Start from schoolbook minus one term
            U = U_school[:, :target_rank].copy()
            V = V_school[:, :target_rank].copy()
            W = W_school[:, :target_rank].copy()
            # Add small noise to escape local minima
            U += np.random.randn(U_dim, target_rank) * 0.1
            V += np.random.randn(V_dim, target_rank) * 0.1
            W += np.random.randn(W_dim, target_rank) * 0.1
        else:
            # Random initialization
            U = np.random.randn(U_dim, target_rank) * 0.5
            V = np.random.randn(V_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5

        # Perform a few iterations of gradient descent or coordinate updates
        # We will use a simple coordinate descent on the flattened parameters
        # but snap to half-integers frequently
        
        current_err = compute_error(U, V, W)
        
        for step in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random tensor factor and coordinate
            # Update U[a, r]
            # Or update whole vector U[:, r]
            
            # Let's update one column of U, V, or W
            choice = np.random.randint(3)
            r = np.random.randint(target_rank)
            
            if choice == 0:
                # Update U[:, r]
                # We want to minimize error by changing U[:, r]
                # This is a least squares problem for U[:, r] given V[:, r] and W[:, r]
                # But it's coupled with other r's.
                # Simpler: try small perturbations
                pass
            
            # Instead of complex optimization, let's just do random snapping and verification
            # This is a "random walk" in the space of half-integer matrices
            # But we need to be guided.
            
            # Let's try to solve for one factor at a time using the others as fixed
            # This is ALS, but with snapping.
            
            # Update U
            # For each a, U[a, :] is determined by V and W
            # But we are fixing the rank to 17.
            
            # Let's just do a random perturbation + snap
            U[np.random.randint(U_dim), np.random.randint(target_rank)] += np.random.choice([-0.5, 0.5])
            U = np.clip(U, -5, 5) # Bound the values
            U = np.round(U * 2) / 2
            
            new_err = compute_error(U, V, W)
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                U[np.random.randint(U_dim), np.random.randint(target_rank)] -= np.random.choice([-0.5, 0.5])
                U = np.round(U * 2) / 2
        
        # Check if we found a solution
        U, V, W, success = refine(U, V, W)
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            found = True
            break
            
    # If rank 17 not found, return rank 18
    # Convert to lists
    def to_list(M):
        return M.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
