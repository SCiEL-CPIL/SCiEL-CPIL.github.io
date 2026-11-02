import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_over_f2_fast(U, V, n, m, p, r):
    """
    Solve for W over F2 using bitmask Gaussian elimination.
    Returns W (integer array) if a valid solution exists, else None.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            eqs = []
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                prod = u_row & v_row
                mask = 0
                # Use a loop for clarity, but could be vectorized if r is large enough
                # r is small (<=36), so loop is fine
                for t in range(r):
                    if prod[t]:
                        mask |= (1 << t)
                eqs.append((mask, 1))
            
            # Gaussian Elimination over F2
            # System: A x = b, where A is m x r (conceptually), b is m x 1
            # We represent rows as (mask, rhs)
            rank = 0
            pivot_cols = []
            
            # We need to handle the case where m > r or m < r
            # Standard GE
            for col in range(r):
                pivot = -1
                for rr in range(rank, m):
                    if (eqs[rr][0] >> col) & 1:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                
                # Swap
                eqs[rank], eqs[pivot] = eqs[pivot], eqs[rank]
                pivot_cols.append(col)
                
                # Eliminate other rows
                for rr in range(m):
                    if rr != rank and ((eqs[rr][0] >> col) & 1):
                        eqs[rr] = (eqs[rr][0] ^ eqs[rank][0], eqs[rr][1] ^ eqs[rank][1])
                
                rank += 1
                if rank == m:
                    break
            
            # Check for inconsistency (0 = 1)
            for rr in range(rank, m):
                if eqs[rr][0] == 0 and eqs[rr][1] == 1:
                    return None
            
            # Back substitution (set free variables to 0)
            x = [0] * r
            for rr in range(rank):
                col = pivot_cols[rr]
                x[col] = eqs[rr][1]
            
            W[k * n + i, :] = x
    
    # Verify over integers (entries are 0 or 1, so product is 0 or 1)
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return None
    
    return W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Trivial decomposition as fallback (Rank 36)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    # Strategy:
    # 1. Try to find a solution for rank 24 (suspected minimal or close to it).
    # 2. If not found, try 25, 26...
    # 3. Use diverse random initializations for U and V.
    # 4. The key insight from inspirations is that a structured or semi-structured search
    #    might be better, but random search with F2 solver is robust.
    #    We will enhance the random search with:
    #    - Varying sparsity levels
    #    - Restarting with different seeds
    #    - Prioritizing lower ranks
    
    random.seed(42)
    
    # Ranks to try: 24 is a strong candidate. 3x3x3 is 23. 3x3x4 is likely 24 or 25.
    # Let's try 24, 25, 26, 27, 28, 29, 30.
    # If we find a solution at rank R, we return it immediately because we iterate upwards.
    # This guarantees we find the *first* valid rank in our search sequence.
    # Is 24 the true minimum?
    # Lower bound for 3x3x4:
    # R(3,3,4) >= R(3,3,3) = 23? Not necessarily, but often close.
    # Actually, R(3,3,4) is known to be 24? Or is it 25?
    # Let's just search.
    
    min_rank_to_try = 24
    max_rank_to_try = 32 # Don't go too high, trivial is 36
    
    for target_rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank
        # If we have lots of time, try more attempts
        # Simple heuristic: try a fixed number of attempts per rank, or until time is up
        # Since lower ranks are harder, maybe give them more time?
        # But if 24 is impossible, we waste time.
        # Let's try a moderate number of attempts.
        
        max_attempts = 20000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            
            # Random sparse patterns
            # Tune sparsity
            # For rank ~24, n*m=9, m*p=12.
            # Average density: 9/24 ~ 0.375, 12/24 ~ 0.5.
            # Let's sample 2-4 non-zeros per row.
            
            for i in range(n * m):
                # Vary sparsity slightly
                num_nonzero = random.randint(2, 4)
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    U[i, c] = 1
            
            for i in range(m * p):
                num_nonzero = random.randint(2, 4)
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    V[i, c] = 1
            
            W = _solve_w_over_f2_fast(U, V, n, m, p, target_rank)
            if W is not None:
                best_result = (target_rank, U, V, W)
                # Return immediately as we are iterating ranks in increasing order
                return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
        
        # If we exhausted attempts for this rank without success, try next rank
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
