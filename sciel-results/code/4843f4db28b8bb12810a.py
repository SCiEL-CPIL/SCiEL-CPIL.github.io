import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def make_schoolbook(rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def check_half_integer(mat):
        """Check if all entries are integers or half-integers."""
        arr = np.array(mat, dtype=np.float64)
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.any(np.abs(doubled - rounded) > 1e-6):
            return False
        return True
    
    def is_valid_solution(U, V, W, rank):
        """Full validation: exact decomposition + half-integer entries."""
        if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
            return False
        if not check_half_integer(U.tolist()):
            return False
        if not check_half_integer(V.tolist()):
            return False
        if not check_half_integer(W.tolist()):
            return False
        return True
        
    best_result = make_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    # Strategy: Remove one term from schoolbook, solve for W, check if half-integer.
    # If successful, we have rank 44.
    # If not, try removing two terms, etc.
    
    # Pre-compute schoolbook U, V as numpy arrays for speed
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 44 down to 40
    for target_rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_removed = rank_schoolbook - target_rank
        
        # We need to choose 'num_removed' terms to remove.
        # For num_removed = 1, we iterate over all 45 terms.
        # For num_removed > 1, the combinations explode.
        # We will use a random sampling of removed terms for num_removed > 1.
        
        if num_removed == 1:
            candidates = [list(range(rank_schoolbook))]
        else:
            # Randomly sample sets of removed terms
            import itertools
            # We can't iterate all combinations for large num_removed.
            # We'll just try random subsets.
            candidates = []
            for _ in range(1000):
                if time.time() > deadline - 10:
                    break
                removed = np.random.choice(rank_schoolbook, size=num_removed, replace=False)
                candidates.append(list(removed))
        
        for removed in candidates:
            if time.time() > deadline - 10:
                break
            
            # Create U, V with removed columns zeroed out
            U_new = U_school.copy()
            V_new = V_school.copy()
            
            # Zero out the removed columns
            for idx in removed:
                U_new[:, idx] = 0
                V_new[:, idx] = 0
            
            # We now have target_rank non-zero columns (effectively).
            # We solve for W_new.
            # The system is: sum_{r not in removed} U_new[:, r] * V_new[:, r] * W_new[:, r] = Tensor
            # Note: U_new and V_new have zeros in the removed columns, so they don't contribute.
            # We can just solve the full system with the constraint that W_new[:, removed] = 0.
            # Or simpler: extract the active columns.
            
            active_cols = [c for c in range(rank_schoolbook) if c not in removed]
            U_active = U_school[:, active_cols]
            V_active = V_school[:, active_cols]
            
            num_eqs = n * m * p
            num_unk = p * n * len(active_cols)
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx, col_idx in enumerate(active_cols):
                            # W[k*n+i, r_idx] is the unknown
                            # The coefficient is U[i*m+j, col_idx] * V[j*p+k, col_idx]
                            # The unknown is at position (k*n+i) * len(active_cols) + r_idx
                            col_pos = (k * n + i) * len(active_cols) + r_idx
                            A[eq_idx, col_pos] = U_active[i * m + j, r_idx] * V_active[j * p + k, r_idx]
                        eq_idx += 1
            
            # Solve A * w_vec = b
            try:
                w_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is exact
            if residuals.size > 0:
                if residuals[0] > 1e-10:
                    continue
            
            # Check if w_vec has half-integer entries
            if not check_half_integer(w_vec.tolist()):
                continue
            
            # Construct W_new
            W_new = np.zeros((p * n, rank_schoolbook))
            for r_idx, col_idx in enumerate(active_cols):
                W_new[:, col_idx] = w_vec[(p * n) * r_idx : (p * n) * (r_idx + 1)]
            
            # Construct U_new, V_new (already done above, but let's make them clean lists)
            # We need to return U, V, W of shape (rows, target_rank)?
            # The interface says: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
            # So we should compress the matrices to only include active columns.
            
            U_final = U_active
            V_final = V_active
            W_final = W_new[:, active_cols] # This is already just w_vec reshaped
            
            # Verify the final solution
            U_list = U_final.tolist()
            V_list = V_final.tolist()
            W_list = W_final.tolist()
            
            if is_valid_solution(U_list, V_list, W_list, len(active_cols)):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = len(active_cols)
                found = True
                break
        
        if found:
            break
    
    return best_result
