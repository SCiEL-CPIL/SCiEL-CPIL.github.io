"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]
import time
import sys
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication of dimensions (2, 3, 9).
    Attempts to find a rank-minimal decomposition using randomized local search.
    Falls back to the standard schoolbook algorithm (rank 54) if no better solution is found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Function to compute the error of a decomposition
    # Error = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
    def compute_error(U, V, W):
        error = 0.0
        # Pre-compute U*V for each (i,j,k) is not straightforward because of W
        # We need to sum over c for each (i,j,k)
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = 0
                    for c in range(len(u_row)):
                        s += u_row[c] * v_val[c] * w_val[c]
                    diff = s - 1
                    error += diff * diff
        return error

    # Local search function for a given rank
    def local_search(rank, max_iters, time_limit):
        # Initialize with random sparse integers
        U = np.zeros((n_rows, rank), dtype=np.int64)
        V = np.zeros((m_rows, rank), dtype=np.int64)
        W = np.zeros((p_rows, rank), dtype=np.int64)
        
        # Fill with random -1, 0, 1
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 0, 1], size=n_rows)
            V[:, c] = np.random.choice([-1, 0, 1], size=m_rows)
            W[:, c] = np.random.choice([-1, 0, 1], size=p_rows)
            
        current_error = compute_error(U.tolist(), V.tolist(), W.tolist())
        
        if current_error == 0:
            return U.tolist(), V.tolist(), W.tolist()

        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Hill climbing
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random variable to modify
            mat_idx = random.choice([0, 1, 2]) # 0:U, 1:V, 2:W
            if mat_idx == 0:
                mat = U
                rows = n_rows
            elif mat_idx == 1:
                mat = V
                rows = m_rows
            else:
                mat = W
                rows = p_rows
                
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, rank - 1)
            
            old_val = mat[r_idx, c_idx]
            # Try changing to old_val +/- 1
            new_val = old_val + random.choice([-1, 1])
            
            mat[r_idx, c_idx] = new_val
            
            new_error = compute_error(U.tolist(), V.tolist(), W.tolist())
            
            if new_error < current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U.tolist(), best_V.tolist(), best_W.tolist()
            else:
                # Revert
                mat[r_idx, c_idx] = old_val
        
        if best_error == 0:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return None

    # Try to find a lower rank decomposition
    # Start from a lower bound. Rank must be at least max(n*m, m*p, p*n) ? No.
    # A simple bound is that the number of non-zero terms in the sum must cover all 54 outputs.
    # Let's try ranks from 30 to 53.
    # We prioritize lower ranks.
    
    # Note: For small dimensions, the rank is often the schoolbook rank or slightly lower.
    # We will try a few random restarts for each rank.
    
    candidate_ranks = list(range(30, 54))
    random.shuffle(candidate_ranks)
    
    best_result = None
    best_rank_found = 54
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        if rank >= best_rank_found:
            continue
            
        # Try a few random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            res = local_search(rank, max_iters=500, time_limit=deadline - 2)
            if res is not None:
                best_result = {"U": res[0], "V": res[1], "W": res[2]}
                best_rank_found = rank
                found = True
                break # Found a solution for this rank, try next (lower) rank
        
        if found:
            # If we found a solution, we can stop if we are at the theoretical minimum?
            # Theoretical min is unknown, but if we find one, it's better than 54.
            # We continue searching for even lower ranks if time permits.
            pass

    if best_result:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
