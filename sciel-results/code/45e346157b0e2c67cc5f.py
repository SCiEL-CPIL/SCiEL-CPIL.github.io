"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization.
    3. Exploit the fact that we only need to verify 36 active triples + ensure
       all other combinations sum to 0.
    4. Use local search / hill climbing to refine candidates.
    5. Leverage the specific structure of the 3x3x4 case by searching for a 
       decomposition of rank 18 (the theoretical lower bound for some subspaces)
       or lower by representing the multiplication as a bilinear map over a finite field.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    
    # Precompute active/inactive masks for fast verification
    active_mask = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples])
    all_triples_arr = np.array(all_triples)
    
    def verify_fast(U, V, W, r):
        """Fast verification using NumPy. U, V, W are 2D lists."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # For each triple (a,b,c), compute sum over columns of U[a,col]*V[b,col]*W[c,col]
        # We can do this efficiently by iterating over columns
        tensor = np.zeros(len(all_triples), dtype=np.int64)
        for ci in range(r):
            u_col = U_np[:, ci]
            v_col = V_np[:, ci]
            w_col = W_np[:, ci]
            # For each triple, add u_col[a]*v_col[b]*w_col[c]
            for idx, (a, b, c) in enumerate(all_triples):
                tensor[idx] += u_col[a] * v_col[b] * w_col[c]
        
        return np.all(tensor == active_mask)
    
    def error_count_fast(U, V, W, r):
        """Count errors using NumPy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        tensor = np.zeros(len(all_triples), dtype=np.int64)
        for ci in range(r):
            u_col = U_np[:, ci]
            v_col = V_np[:, ci]
            w_col = W_np[:, ci]
            for idx, (a, b, c) in enumerate(all_triples):
                tensor[idx] += u_col[a] * v_col[b] * w_col[c]
        
        return int(np.sum(tensor != active_mask))
    
    def active_error_count(U, V, W, r):
        """Count only active triple errors (faster check)."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        errors = 0
        for (a, b, c) in active_triples:
            s = 0
            for ci in range(r):
                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
            if s != 1:
                errors += 1
        return errors
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            strategy = attempt % 5
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for ci in range(r):
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
            
            elif strategy == 3:
                # Binary sparse: ±1 with controlled density
                density = max(1, r // 4)
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(density, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(density, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(density, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            else:
                # Mixed: some rows dense, some sparse
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    if random.random() < 0.5:
                        U[a] = [random.choice([-1, 0, 1]) for _ in range(r)]
                    else:
                        num_nonzero = random.randint(1, min(2, r))
                        cols = random.sample(range(r), num_nonzero)
                        for ci in cols:
                            U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    if random.random() < 0.5:
                        V[b] = [random.choice([-1, 0, 1]) for _ in range(r)]
                    else:
                        num_nonzero = random.randint(1, min(2, r))
                        cols = random.sample(range(r), num_nonzero)
                        for ci in cols:
                            V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    if random.random() < 0.5:
                        W[c] = [random.choice([-1, 0, 1]) for _ in range(r)]
                    else:
                        num_nonzero = random.randint(1, min(2, r))
                        cols = random.sample(range(r), num_nonzero)
                        for ci in cols:
                            W[c][ci] = random.choice([-1, 1])
            
            # Quick check: verify active triples first
            if active_error_count(U, V, W, r) > 0:
                continue
            
            # Full verification
            if verify_fast(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def local_search(U, V, W, r, max_steps=100):
        """Local search to refine a near-solution."""
        if time.time() > deadline:
            return None
        
        current_errors = error_count_fast(U, V, W, r)
        if current_errors == 0:
            return {"U": U, "V": V, "W": W}
        
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        best_errors = current_errors
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if best_errors == 0:
                break
            
            # Try flipping a single entry in U, V, or W
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Randomly choose which matrix and which entry to flip
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    idx = random.randint(0, n * m - 1)
                    ci = random.randint(0, r - 1)
                    old_val = U[idx][ci]
                    U[idx][ci] = -old_val if old_val != 0 else random.choice([-1, 1])
                    errors = error_count_fast(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                    else:
                        U[idx][ci] = old_val
                elif mat_choice == 'V':
                    idx = random.randint(0, m * p - 1)
                    ci = random.randint(0, r - 1)
                    old_val = V[idx][ci]
                    V[idx][ci] = -old_val if old_val != 0 else random.choice([-1, 1])
                    errors = error_count_fast(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                    else:
                        V[idx][ci] = old_val
                else:
                    idx = random.randint(0, p * n - 1)
                    ci = random.randint(0, r - 1)
                    old_val = W[idx][ci]
                    W[idx][ci] = -old_val if old_val != 0 else random.choice([-1, 1])
                    errors = error_count_fast(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                    else:
                        W[idx][ci] = old_val
            
            if not improved and step > 20:
                # Reset to best and try different strategy
                U = [row[:] for row in best_U]
                V = [row[:] for row in best_V]
                W = [row[:] for row in best_W]
        
        if best_errors == 0:
            return {"U": best_U, "V": best_V, "W": best_W}
        return None
    
    # Try to find lower rank decompositions
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        if r <= 22:
            max_attempts = min(500, int(time_remaining * 10))
        elif r <= 26:
            max_attempts = min(300, int(time_remaining * 5))
        else:
            max_attempts = min(100, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we found a solution, try to improve it with local search at same rank
    if best_rank < n * m * p and time.time() < deadline - 10:
        # Try local search to see if we can find a lower rank solution
        for r in range(18, best_rank):
            if time.time() > deadline - 5:
                break
            
            # Use the found solution as a starting point for local search
            U_start = [[0] * r for _ in range(n * m)]
            V_start = [[0] * r for _ in range(m * p)]
            W_start = [[0] * r for _ in range(p * n)]
            
            # Initialize with sparse random
            for a in range(n * m):
                num_nonzero = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nonzero)
                for ci in cols:
                    U_start[a][ci] = random.choice([-1, 1])
            for b in range(m * p):
                num_nonzero = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nonzero)
                for ci in cols:
                    V_start[b][ci] = random.choice([-1, 1])
            for c in range(p * n):
                num_nonzero = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nonzero)
                for ci in cols:
                    W_start[c][ci] = random.choice([-1, 1])
            
            ls_result = local_search(U_start, V_start, W_start, r, max_steps=50)
            if ls_result is not None:
                best_result = ls_result
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
