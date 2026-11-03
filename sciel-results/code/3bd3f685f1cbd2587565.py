"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def construct():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matmul.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try Winograd-style blocking for 2x2 blocks
    # For 3x4 * 4x4, we can decompose into overlapping 2x2 blocks
    
    # Method 1: Try rank 43 (known lower bound is around 43-45 for 3x4x4)
    # Use structured search with small integer entries
    
    def try_decomposition(rank, max_entries=3, time_limit=None):
        """Try to find a decomposition of given rank with small integer entries."""
        if time_limit is not None and time.time() > time_limit:
            return None
        
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # For each output position (i,k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
        # Actually, the constraint is:
        # For each (i,j,k): sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # No wait, the tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But T[i,j,k] = 1 for all i,j,k in our case (matrix mult tensor)
        
        # Actually for matrix multiplication A(3x4) * B(4x4) = C(3x4):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i,j,k) where T[i,j,k] = 1
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        # But that's not right either. The correct formulation:
        # The multiplication tensor M has M_{i,j,k} = 1 for all valid (i,j,k)
        # A rank-r decomposition means M = sum_{r=1}^{r} u_r v_r w_r (outer product)
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # So: M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And M_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # So we need: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is a system of nonlinear equations. Let's try a simpler approach:
        # Greedy construction with verification
        
        entries = list(range(-max_entries, max_entries + 1))
        
        # Simple greedy: for each rank term, try to satisfy as many unsatisfied constraints as possible
        # Constraint (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        
        satisfied = np.zeros((n, m, p), dtype=bool)
        residual = np.ones((n, m, p), dtype=int)
        
        for r in range(rank):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Try all combinations of single entries for this rank term
            best_combo = None
            best_score = 0
            
            # For efficiency, try combinations where most entries are 0 or 1
            # U has n*m = 12 entries, V has m*p = 16, W has p*n = 12
            # Too many combinations. Use a heuristic.
            
            # Heuristic: pick random sparse vectors
            for attempt in range(200):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                u_vec = np.zeros(n * m, dtype=int)
                v_vec = np.zeros(m * p, dtype=int)
                w_vec = np.zeros(p * n, dtype=int)
                
                # Make sparse vectors
                for _ in range(np.random.randint(2, 6)):
                    u_vec[np.random.randint(0, n * m)] = np.random.choice(entries)
                for _ in range(np.random.randint(2, 6)):
                    v_vec[np.random.randint(0, m * p)] = np.random.choice(entries)
                for _ in range(np.random.randint(2, 6)):
                    w_vec[np.random.randint(0, p * n)] = np.random.choice(entries)
                
                # Compute contribution
                contrib = np.zeros((n, m, p), dtype=int)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib[i, j, k] = u_vec[i * m + j] * v_vec[j * p + k] * w_vec[k * n + i]
                
                # Score: how well this helps satisfy constraints
                # We want residual to become 0
                new_residual = residual - contrib
                # Penalize if we overshoot
                score = np.sum(np.abs(new_residual)) 
                # We want to minimize |residual|, so lower score is better
                # But we also want to make progress, so reward reducing |residual|
                progress = np.sum(np.abs(residual)) - np.sum(np.abs(new_residual))
                
                if progress > best_score:
                    best_score = progress
                    best_combo = (u_vec.copy(), v_vec.copy(), w_vec.copy())
                    # Check if this satisfies all remaining
                    if np.all(new_residual == 0):
                        U[:, r] = u_vec
                        V[:, r] = v_vec
                        W[:, r] = w_vec
                        residual = new_residual
                        satisfied = np.all(residual == 0)
                        if satisfied:
                            return U, V, W, True
                        break
            
            if best_combo is not None:
                U[:, r] = best_combo[0]
                V[:, r] = best_combo[1]
                W[:, r] = best_combo[2]
                residual -= best_combo[0][:, None, None] * best_combo[1][None, :, None] * best_combo[2][None, None, :]
                # Need to reshape properly
                # Actually let me recompute
                residual = np.ones((n, m, p), dtype=int)
                for rr in range(r + 1):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i, j, k] -= U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                
                if np.all(residual == 0):
                    return U, V, W, True
        
        # Final check
        residual_final = np.ones((n, m, p), dtype=int)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual_final[i, j, k] -= U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        if np.all(residual_final == 0):
            return U, V, W, True
        return None
    
    # Try various ranks
    for rank in range(40, min(48, best_rank)):
        if time.time() > deadline - 5:
            break
        result = try_decomposition(rank, max_entries=2, time_limit=deadline - 5)
        if result is not None:
            U_r, V_r, W_r, valid = result
            if valid:
                best_rank = rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break  # Found a valid decomposition, stop searching lower ranks
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
