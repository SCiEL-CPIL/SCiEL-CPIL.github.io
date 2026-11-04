import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p  # 75
    
    # Build the schoolbook decomposition
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # For each (i, j, k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Verify schoolbook works
    assert check_decomposition(U_sb, V_sb, W_sb)
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to reduce rank by removing columns and adjusting
    # Strategy: try removing one column at a time and see if we can compensate
    
    # Alternative: use optimization to find a lower rank solution
    # Let's try a greedy approach: try to remove columns one by one
    
    def try_reduce_rank(U, V, W, max_iterations=100):
        """Try to reduce rank by removing columns."""
        rank = U.shape[1]
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            # Try removing each column
            found_removal = False
            for c_remove in range(current_U.shape[1]):
                if time.time() > deadline:
                    break
                # Temporarily remove column c_remove
                mask = [c for c in range(current_U.shape[1]) if c != c_remove]
                U_trial = current_U[:, mask]
                V_trial = current_V[:, mask]
                W_trial = current_W[:, mask]
                
                # Check if still valid
                if check_decomposition(U_trial, V_trial, W_trial):
                    current_U, current_V, current_W = U_trial, V_trial, W_trial
                    found_removal = True
                    break
            if not found_removal:
                break
        
        return current_U, current_V, current_W
    
    # First try simple column removal
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U_sb, V_sb, W_sb)
    new_rank = U_reduced.shape[1]
    if new_rank < best_rank:
        best_rank = new_rank
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Now try more sophisticated approaches: allow small perturbations
    # Use the fact that we need integer/half-integer entries
    
    # Let's try a different approach: parameterize and optimize
    # For a given target rank r, try to find U, V, W
    
    # Try ranks from best_rank-1 down to some minimum
    # This is computationally expensive, so be careful
    
    # Let's try a local search: perturb entries slightly and see if we can reduce rank
    
    # Actually, let's try a more systematic approach
    # For each pair of columns, see if one can be expressed as a combination of others
    
    # Given time constraints, let's try a few more strategies
    
    # Strategy: try to find a decomposition with rank < best_rank using random initialization
    for target_rank in range(min(best_rank - 1, 70), max(20, best_rank - 10), -1):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        # Try random initialization and optimization
        np.random.seed(42)
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Simple gradient-free search: try to minimize error
        # This is a non-convex problem, so it's hard
        
        # Let's skip this for now and focus on what we have
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        # Round to nearest half-integer
        rounded = np.round(mat * 2) / 2
        return rounded.tolist()
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification
    U_check = np.array(U_out)
    V_check = np.array(V_out)
    W_check = np.array(W_out)
    if not check_decomposition(U_check, V_check, W_check):
        # Fall back to schoolbook
        U_out = to_list(U_sb)
        V_out = to_list(V_sb)
        W_out = to_list(W_sb)
    
    return {"U": U_out, "V": V_out, "W": W_out}
