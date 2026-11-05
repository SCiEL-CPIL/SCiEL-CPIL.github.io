import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        rank = n * m * p
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
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    def _verify_tensor(U, V, W, rank):
        """Verify by checking the tensor equation directly."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def _try_random_search(rank, num_attempts=50):
        """Try to find a valid 0/1 decomposition at given rank."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Initialize with random 0/1
            for i in range(n * m):
                for r in range(rank):
                    U[i][r] = 1 if np.random.random() < 0.3 else 0
            for j in range(m * p):
                for r in range(rank):
                    V[j][r] = 1 if np.random.random() < 0.3 else 0
            for k in range(p * n):
                for r in range(rank):
                    W[k][r] = 1 if np.random.random() < 0.3 else 0
            
            # Quick check: for each (i,j,k), there must be at least one r where all three are 1
            # and the sum must be exactly 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            prod = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            total += prod
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try random search at various ranks below schoolbook
    # Start from rank 95 down to 32
    for rank in range(schoolbook_rank - 1, 31, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts
        result = _try_random_search(rank, num_attempts=20)
        if result is not None:
            # Verify with tensor check
            if _verify_tensor(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank
                break  # Found a valid decomposition at this rank
    
    # Structured approach: exploit block structure since p = 2*m
    # Split B into B1 = B[:, 0:4] and B2 = B[:, 4:8]
    # C = [A@B1, A@B2]
    # If we have rank-r decomposition for 3x4 by 4x4, we can use 2r for the combined problem
    # But we can potentially do better by sharing structure
    
    def _try_structured_block(rank):
        """Try to find a decomposition exploiting the block structure."""
        if rank < 2 * n * m * (p // 2):
            return None
        
        # For each half of p, we need a 3x4 by 4x4 decomposition
        # Use the same U for both halves, but different V and W
        
        half_p = p // 2  # 4
        base_rank = rank // 2
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First half: k = 0..3
        for r in range(base_rank):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if (i * m + j) == r else 0
            for j in range(m):
                for k in range(half_p):
                    V[j * p + k][r] = 1 if (j * half_p + k) == r else 0
            for k in range(half_p):
                for i in range(n):
                    W[k * n + i][r] = 1 if (k * n + i) == r else 0
        
        # Second half: k = 4..7
        for r in range(base_rank):
            r2 = base_rank + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r2] = U[i * m + j][r]
            for j in range(m):
                for k in range(half_p):
                    V[j * p + half_p + k][r2] = V[j * p + k][r]
            for k in range(half_p):
                for i in range(n):
                    W[(half_p + k) * n + i][r2] = W[k * n + i][r]
        
        # Verify
        if _verify_tensor(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try structured approach with various ranks
    for rank in range(64, 31, -1):
        if time.time() > deadline:
            break
        
        result = _try_structured_block(rank)
        if result is not None:
            if _verify_tensor(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank
                break
    
    # Greedy elimination approach from schoolbook
    def _greedy_elimination():
        """Try to eliminate terms from schoolbook decomposition."""
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        active_terms = list(range(rank))
        
        def check_feasibility(active_terms, U_init, V_init):
            """Check if there exist U, V, W for the active terms."""
            r = len(active_terms)
            if r == 0:
                return False, None, None, None
            
            # Build matrix A where A[eq_idx, w_idx] = U[i*m+j,c] * V[j*p+k,c]
            A = np.zeros((n*m*p, p*n * r))
            b = np.ones(n*m*p)
            
            for c_local, c_global in enumerate(active_terms):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            w_idx = c_local * (p*n) + (k * n + i)
                            A[eq_idx, w_idx] = U_init[i*m+j, c_global] * V_init[j*p+k, c_global]
            
            # Solve least squares
            try:
                W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol = W_sol.reshape(r, p*n).T  # shape (p*n, r)
                
                # Check residual
                residual = np.linalg.norm(A @ W_sol.flatten() - b)
                if residual < 1e-8:
                    return True, U_init[:, active_terms], V_init[:, active_terms], W_sol
                else:
                    return False, None, None, None
            except:
                return False, None, None, None
        
        current_U, current_V, current_W = U, V, W
        
        while time.time() < deadline - 20:
            removed_this_round = False
            for c_remove in list(active_terms):
                if time.time() > deadline - 20:
                    break
                
                candidate_terms = [c for c in active_terms if c != c_remove]
                if len(candidate_terms) == 0:
                    break
                
                # Try to find U, V, W for candidate_terms
                feasible, U_new, V_new, W_new = check_feasibility(candidate_terms, current_U, current_V)
                
                if feasible:
                    active_terms = candidate_terms
                    current_U, current_V, current_W = U_new, V_new, W_new
                    removed_this_round = True
                    break  # Restart the loop with the new active set
            
            if not removed_this_round:
                break
        
        if len(active_terms) < rank:
            # Final verification
            U_final = current_U[:, active_terms]
            V_final = current_V[:, active_terms]
            W_final = current_W[:, active_terms]
            
            T = np.zeros((n, m, p))
            for c in range(U_final.shape[1]):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U_final[i*m+j, c] * V_final[j*p+k, c] * W_final[k*n+i, c]
            
            if np.allclose(T, 1.0, atol=1e-10):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}, len(active_terms)
        
        return None, rank
    
    result, new_rank = _greedy_elimination()
    if result is not None and new_rank < best_rank:
        best_result = result
        best_rank = new_rank
    
    return best_result
# EVOLVE-BLOCK-END
