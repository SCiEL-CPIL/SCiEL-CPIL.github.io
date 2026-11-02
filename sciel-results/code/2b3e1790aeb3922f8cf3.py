import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, time_limit, seed=42):
    """
    Try to find a rank-`rank` decomposition.
    Uses random U, V with small integers, then solves for W.
    Returns (U, V, W) or None.
    """
    np.random.seed(seed)
    deadline = time.time() + time_limit
    
    # Number of random attempts based on available time
    # Each attempt is O(n*m*p*rank + n*p*rank^2) roughly
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random U and V with small integer entries
        # Using -1, 0, 1 is often sufficient for low rank decompositions
        # We can also try -2, -1, 0, 1, 2 for more diversity
        scale = 1 if rank > 40 else 2
        U = np.random.randint(-scale, scale + 1, size=(n*m, rank))
        V = np.random.randint(-scale, scale + 1, size=(m*p, rank))
        
        # Ensure we don't have all-zero columns in U or V
        if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
            continue
            
        # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        # Reshape U to (n, m, rank) and V to (m, p, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        # M[i, k, c] = sum_j U_reshaped[i, j, c] * V_reshaped[j, k, c]
        M = np.einsum('ijc,jkc->ikc', U_reshaped, V_reshaped)
        
        # Solve for W: for each (i, k), M[i, k, :] dot W[k*n+i, :] = 1
        W = np.zeros((p*n, rank))
        valid = True
        for i in range(n):
            for k in range(p):
                m_vec = M[i, k, :]
                norm_sq = np.dot(m_vec, m_vec)
                if norm_sq < 1e-10:
                    valid = False
                    break
                W[k*n+i, :] = m_vec / norm_sq
            if not valid:
                break
        
        if not valid:
            continue
            
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    schoolbook_rank = n * m * p  # 96
    
    # We want to find the minimal rank.
    # Start from schoolbook - 1 and go down.
    # If we find a decomposition at rank r, we try r-1, etc.
    # However, searching from 95 down to 1 is expensive.
    # We will allocate time based on the rank.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Estimate time per attempt
    # Let's reserve 10 seconds for final verification and cleanup.
    total_time_budget = 140  # Leave 10s margin for 150s limit
    
    # Strategy:
    # 1. Try ranks from schoolbook-1 down to some lower bound.
    # 2. Allocate more time to higher ranks (closer to schoolbook) as they are more likely to exist?
    #    Actually, lower ranks are harder to find but more valuable.
    #    However, if rank 95 exists, we should find it quickly.
    #    Let's iterate downwards.
    
    # To be efficient, we can try a few ranks in parallel or sequentially with time slices.
    # Given the 150s limit, let's do a sequential search downwards.
    
    # We will try ranks from 95 down to 1.
    # But we need to be smart about time allocation.
    # If we spend too much time on 95 and fail, we might not reach 50.
    # Let's allocate a fixed time slice per rank, or dynamically.
    
    # Let's try ranks in descending order.
    # For each rank, give it a time slice.
    # If we find a solution, we update best_rank and continue to lower ranks?
    # The task asks for "rank-minimal". So we must find the absolute minimum.
    # This implies we need to check all ranks down to 1?
    # Or we can stop if we hit a theoretical lower bound?
    # The lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # So we need to search from 95 down to 32.
    
    # That's 64 ranks to check.
    # Total time 140s. ~2.1s per rank.
    # This is very tight.
    
    # Optimization:
    # If we find a solution at rank R, we know R is achievable.
    # We still need to check R-1, R-2... to see if we can do better.
    
    # Let's prioritize:
    # 1. Check if 95 is possible (likely yes).
    # 2. Check 94, 93...
    
    # To maximize the chance of finding a low rank, we should spend time on lower ranks too.
    # But if we don't find 95, we can't know if 94 exists without checking.
    
    # Let's use a time-slicing approach.
    # Allocate time proportional to the "difficulty" or just equal slices.
    # Since lower ranks are harder, maybe give them more time?
    # But if we miss the easy ones, we might return a suboptimal high rank.
    # Actually, if we find 95, we MUST check 94 to see if it's better.
    
    # Let's just iterate from 95 down to 32.
    # Time per rank = (total_time_budget) / (schoolbook_rank - 32)
    
    ranks_to_check = range(schoolbook_rank - 1, 31, -1)
    num_ranks = len(ranks_to_check)
    time_per_rank = total_time_budget / num_ranks
    
    current_rank = schoolbook_rank
    
    for rank in ranks_to_check:
        if time.time() - start_time > total_time_budget:
            break
            
        # Allocate a bit of time for this rank
        # If we have found a solution already, we are trying to improve it.
        # If not, we are trying to find any.
        
        # Dynamic time allocation:
        # If we haven't found anything yet, we might want to spend more time on higher ranks
        # because they are easier.
        # If we have found rank R, we are trying R-1.
        
        # Let's just use a fixed slice for simplicity, but maybe larger for lower ranks?
        # No, fixed slice is safer to ensure we cover the range.
        
        result = _try_rank(n, m, p, rank, time_per_rank, seed=int(rank))
        
        if result is not None:
            U, V, W = result
            current_rank = rank
            best_result = (U, V, W)
            # We continue to check lower ranks to see if we can do even better.
            # But we should be careful with time.
            
    if best_result is not None:
        U, V, W = best_result
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
    return {"U": U_list, "V": V_list, "W": W_list}
