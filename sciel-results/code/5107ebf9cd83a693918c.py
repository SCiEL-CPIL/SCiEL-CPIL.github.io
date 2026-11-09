"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Precompute indices for faster checking
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    index_to_pos = {(i, j, k): idx for idx, (i, j, k) in enumerate(indices)}
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i, j, k in indices:
            val = 0
            for c in range(rank):
                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if val != 1:
                return False
        return True
    
    def greedy_construct(max_rank=95):
        """Try to construct a decomposition greedily with improved heuristics."""
        current_sum = np.zeros((n, m, p), dtype=int)
        U_cols = []
        V_cols = []
        W_cols = []
        
        uncovered = set(indices)
        
        for c in range(max_rank):
            if time.time() > deadline:
                break
            if not uncovered:
                break
            
            best_col = None
            best_count = 0
            best_penalty = 0
            
            # Try various patterns for this column
            # Use a mix of random and structured approaches
            for _ in range(80):
                if time.time() > deadline:
                    break
                
                # Choose density based on remaining coverage
                remaining = len(uncovered)
                total = n * m * p
                density = max(0.1, min(0.6, 2.0 * remaining / total))
                
                # Randomly select some (i,j) for U
                u_active = set()
                for i in range(n):
                    for j in range(m):
                        if random.random() < density:
                            u_active.add((i, j))
                
                # Randomly select some (j,k) for V
                v_active = set()
                for j in range(m):
                    for k in range(p):
                        if random.random() < density:
                            v_active.add((j, k))
                
                # Randomly select some (k,i) for W
                w_active = set()
                for k in range(p):
                    for i in range(n):
                        if random.random() < density:
                            w_active.add((k, i))
                
                # Count how many uncovered entries this column would cover
                count = 0
                for (i, j, k) in uncovered:
                    if (i, j) in u_active and (j, k) in v_active and (k, i) in w_active:
                        count += 1
                
                # Penalize columns that would over-cover already covered entries
                # (not directly possible with current_sum=1, but track for future)
                if count > best_count or (count == best_count and count > 0):
                    best_count = count
                    best_col = (u_active, v_active, w_active)
            
            if best_col is None or best_count == 0:
                # Try a more targeted approach: find an uncovered entry and build around it
                if uncovered:
                    sample = next(iter(uncovered))
                    i, j, k = sample
                    u_active = {(i, j)}
                    v_active = {(j, k)}
                    w_active = {(k, i)}
                    best_col = (u_active, v_active, w_active)
                    best_count = 1
                else:
                    break
            
            u_active, v_active, w_active = best_col
            
            # Build the column vectors
            u_col = np.zeros(nm, dtype=int)
            v_col = np.zeros(mp, dtype=int)
            w_col = np.zeros(pn, dtype=int)
            
            for (i, j) in u_active:
                u_col[i*m + j] = 1
            for (j, k) in v_active:
                v_col[j*p + k] = 1
            for (k, i) in w_active:
                w_col[k*n + i] = 1
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
            # Update current sum and uncovered
            for (i, j, k) in list(uncovered):
                if (i, j) in u_active and (j, k) in v_active and (k, i) in w_active:
                    current_sum[i, j, k] += 1
                    if current_sum[i, j, k] == 1:
                        uncovered.discard((i, j, k))
        
        if not uncovered and len(U_cols) > 0:
            rank = len(U_cols)
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            return U, V, W, rank
        
        return None
    
    # Try greedy construction multiple times with different random seeds
    for trial in range(5):
        if time.time() > deadline:
            break
        random.seed(trial * 42 + 7)
        result = greedy_construct()
        if result is not None:
            U, V, W, rank = result
            if check_decomposition(U, V, W, rank) and rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
