import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_over_f2(U, V, n, m, p, r):
    """
    Solve for W over F2 using bitmask Gaussian elimination.
    Returns W if a valid integer solution exists, else None.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            # Build system: for each j, sum_t (U[i*m+j,t] & V[j*p+k,t]) * W[k*n+i,t] = 1 mod 2
            rows = []
            for j in range(m):
                mask = 0
                for t in range(r):
                    if U[i * m + j, t] & V[j * p + k, t]:
                        mask |= (1 << t)
                rows.append(mask)
            
            # Gaussian elimination over F2
            # We have m equations in r unknowns
            aug = [(rows[j], 1) for j in range(m)]
            rank = 0
            pivot_cols = []
            for col in range(r):
                # Find pivot
                pivot = -1
                for rr in range(rank, m):
                    if (aug[rr][0] >> col) & 1:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                aug[rank], aug[pivot] = aug[pivot], aug[rank]
                pivot_cols.append(col)
                for rr in range(m):
                    if rr != rank and ((aug[rr][0] >> col) & 1):
                        aug[rr] = (aug[rr][0] ^ aug[rank][0], aug[rr][1] ^ aug[rank][1])
                rank += 1
                if rank == m:
                    break
            
            # Check consistency
            for rr in range(rank, m):
                if aug[rr][0] == 0 and aug[rr][1] == 1:
                    return None
            
            # Back-substitute (set free variables to 0)
            x = [0] * r
            for rr in range(rank):
                x[pivot_cols[rr]] = aug[rr][1]
            
            W[k * n + i, :] = x
    
    # Verify over integers (entries are 0 or 1, so product is 0 or 1)
    U_int = U.astype(np.int64)
    V_int = V.astype(np.int64)
    W_int = W.astype(np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should equal 1
                prod = U_int[i * m + j, :] * V_int[j * p + k, :] * W_int[k * n + i, :]
                if prod.sum() != 1:
                    return None
    
    return W

def _try_rank_f2(target_rank, n, m, p, deadline, seed_offset=0):
    """Try to find a rank decomposition using F2 search."""
    random.seed(42 + seed_offset)
    attempts = 0
    max_attempts = 200000
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        U = np.zeros((n * m, target_rank), dtype=np.int64)
        V = np.zeros((m * p, target_rank), dtype=np.int64)
        
        # Vary sparsity for diversity
        for i in range(n * m):
            num_nonzero = random.randint(1, 4)
            cols = random.sample(range(target_rank), num_nonzero)
            for c in cols:
                U[i, c] = 1
        
        for i in range(m * p):
            num_nonzero = random.randint(1, 4)
            cols = random.sample(range(target_rank), num_nonzero)
            for c in cols:
                V[i, c] = 1
        
        W = _solve_w_over_f2(U, V, n, m, p, target_rank)
        if W is not None:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
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
    
    # Try ranks from 23 upward, with more attempts for lower ranks
    # Use a shorter deadline for search to leave time for verification
    search_deadline = start_time + 130  # 20s safety margin
    
    for target_rank in [23, 24, 25, 26, 27, 28]:
        if time.time() > search_deadline:
            break
        
        result = _try_rank_f2(target_rank, n, m, p, search_deadline, seed_offset=target_rank)
        if result is not None:
            U, V, W = result
            best_result = (target_rank, U, V, W)
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
