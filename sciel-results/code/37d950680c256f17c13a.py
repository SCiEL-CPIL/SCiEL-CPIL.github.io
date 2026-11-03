import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i, j, k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # 1. Generate a known valid decomposition using Strassen's algorithm
    # We need to decompose 3x4 * 4x6.
    # We can pad to 4x4 * 4x4 and use Strassen (rank 7), then extract.
    # Or, more directly, use the standard block decomposition.
    # Let's use the standard schoolbook first as a fallback, but we want better.
    
    # Let's try to construct a rank-66 decomposition manually or via a known method?
    # Actually, for 3x4x6, the lower bound is not 66. 
    # Wait, the current program returns schoolbook (72). 
    # Strassen for 4x4x4 is rank 7. 
    # If we pad 3x4 to 4x4 and 4x6 to 4x8? No.
    
    # Let's use a different strategy: 
    # The task asks for rank-minimal. 
    # Known bounds: 
    # R(3,4,6) <= ?
    # Strassen's algorithm gives R(n,n,n) <= 7^(log2 n).
    # For 3x4x6, we can view it as a sub-tensor of 4x4x6 or 3x4x8.
    
    # Let's try to find a rank-56 or similar decomposition using a "structured" random search 
    # that is guided by the algebraic constraints, rather than pure random.
    
    # However, a more robust "structural" change is to use **Integer Linear Programming (ILP)** 
    # or **Satisfiability (SAT)** if available, but we only have NumPy/SciPy.
    
    # Alternative Structural Change: 
    # Use the **Landsberg-Oppermann** bound or known decompositions.
    # Actually, let's look at the "Inspiration 3" which suggests ALS. 
    # The prompt asks for a DIFFERENT strategy. 
    # The current strategy is Random Search. 
    # Inspiration 3 is ALS (Numerical). 
    # I will implement a **Hybrid Deterministic + Local Search** approach.
    
    # 1. Start with the Schoolbook decomposition (Rank 72).
    # 2. Attempt to reduce rank by "merging" terms.
    #    A term c is redundant if its contribution can be expressed as a linear combination of others.
    #    This is hard to do directly.
    
    # Let's try a **Greedy Rank Reduction** heuristic:
    # 1. Start with a valid decomposition of rank R.
    # 2. Try to set one column of U, V, W to zero.
    # 3. If the error becomes non-zero, try to adjust the remaining columns (via least squares) 
    #    to compensate, while keeping them sparse/integer if possible.
    
    # But "keeping them integer" is the hard part.
    
    # Let's try a **Randomized Local Search** with a **Repair Mechanism**.
    # 1. Generate a random sparse decomposition of rank R.
    # 2. If it fails, identify the "worst" error term (i,j,k).
    # 3. Adjust the entries in U, V, W for that specific term to try to fix it, 
    #    while minimizing the increase in error for other terms.
    
    # This is different from pure random restarts. It uses gradient-like information (the error itself).
    
    # Let's implement this "Error-Guided Local Search".
    
    def get_error(U, V, W):
        """Compute the error tensor E[i,j,k] = sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1"""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        E = np.ones((n, m, p))
        for c in range(rank):
            # Outer product of U[:,c] (n*m), V[:,c] (m*p), W[:,c] (p*n)
            # This is a bit expensive if done naively, but n,m,p are small.
            u = U_arr[:, c].reshape(n, m)
            v = V_arr[:, c].reshape(m, p)
            w = W_arr[:, c].reshape(p, n)
            
            # Contribution to E[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            # We can compute this via einsum or loops.
            # Since n,m,p are small (3,4,6), loops are fine.
            for i in range(n):
                for j in range(m):
                    if u[i,j] == 0: continue
                    for k in range(p):
                        E[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                        
        return E - 1.0 # We want sum - 1 = 0

    def verify_fast(U, V, W):
        E = get_error(U, V, W)
        return np.all(np.abs(E) < 1e-9)

    # Start with Schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    random.seed(42)
    
    # Strategy: Try to find a decomposition with rank < 72.
    # We will use a "Simulated Annealing" style local search on the space of sparse integer matrices.
    
    # State: U, V, W
    # Energy: Sum of squared errors + penalty for non-integer/non-sparse (we enforce integer by rounding)
    
    # Since we need EXACT integer/half-integer, we restrict our search space to integers.
    
    # Let's try ranks from 71 down to 50.
    for target_rank in range(71, 49, -1):
        if time.time() > deadline:
            break
        
        # Initialize with random sparse integers
        vals = [0, 0, 0, 1, 1, -1, 2, -2]
        
        found = False
        max_iters = 5000
        current_temp = 1.0
        min_temp = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate initial guess if it is the first iteration for this rank
            if it == 0:
                U_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
                V_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
                W_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
                
                # Check if it's already valid
                if verify_fast(U_cand, V_cand, W_cand):
                    found = True
                    break
            
            # Local Search Step:
            # Pick a random matrix (U, V, or W), a random row, and a random column.
            # Modify the value by -1, 0, or 1 (or a small random integer).
            
            mat_type = random.choice(['U', 'V', 'W'])
            if mat_type == 'U':
                mat = U_cand
                rows = n * m
            elif mat_type == 'V':
                mat = V_cand
                rows = m * p
            else:
                mat = W_cand
                rows = p * n
                
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, target_rank - 1)
            
            old_val = mat[r_idx][c_idx]
            # Try a small change
            change = random.choice([-2, -1, 1, 2])
            new_val = old_val + change
            
            # To keep numbers small, if new_val is large, maybe reset?
            # Let's just apply the change.
            mat[r_idx][c_idx] = new_val
            
            # Calculate Error
            E = get_error(U_cand, V_cand, W_cand)
            error_sq = np.sum(E * E)
            
            # Acceptance Criterion (Simulated Annealing)
            # We don't have a "previous" error stored easily in this simple loop without overhead.
            # Let's store the previous state.
            
            # Actually, let's just do a "Hill Climbing" with random restarts if stuck.
            # This is simpler and often works for small discrete spaces.
            
            # If error is 0, we found it.
            if error_sq < 1e-9:
                found = True
                break
            
            # If we are getting worse consistently, we might need to restart.
            # For now, let's just keep going.
            
            # Cooling
            current_temp *= 0.99
            
            # If we've done many iterations without improvement, restart?
            # Let's just let it run.
        
        if found:
            if target_rank < best_rank:
                best = {"U": U_cand, "V": V_cand, "W": W_cand}
                best_rank = target_rank
            # Continue to see if we can go lower
        else:
            # If we didn't find it, continue to next lower rank?
            # No, if we can't find rank 71, we might not find 70 either with this method.
            # But it's possible.
            pass
        
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # 1. Generate a known valid decomposition using Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # 2. Define a fast error calculation
    def get_error(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        E = np.ones((n, m, p))
        for c in range(rank):
            u = U_arr[:, c].reshape(n, m)
            v = V_arr[:, c].reshape(m, p)
            w = W_arr[:, c].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    if u[i,j] == 0: continue
                    for k in range(p):
                        E[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                        
        return E - 1.0

    def verify_fast(U, V, W):
        E = get_error(U, V, W)
        return np.all(np.abs(E) < 1e-9)

    random.seed(42)
    
    # 3. Error-Guided Local Search
    # We will attempt to find decompositions with rank < 72.
    
    for target_rank in range(71, 55, -1):
        if time.time() > deadline:
            break
        
        vals = [0, 0, 0, 1, 1, -1, 2, -2]
        found = False
        
        # Multiple restarts for each rank
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W_cand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check initial
            if verify_fast(U_cand, V_cand, W_cand):
                found = True
                break
            
            # Local Search
            current_error = np.sum(get_error(U_cand, V_cand, W_cand) ** 2)
            best_local_error = current_error
            best_local_state = (U_cand, V_cand, W_cand)
            
            max_steps = 1000
            stuck_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random variable to change
                mat_type = random.choice(['U', 'V', 'W'])
                if mat_type == 'U':
                    mat = U_cand
                    rows = n * m
                elif mat_type == 'V':
                    mat = V_cand
                    rows = m * p
                else:
                    mat = W_cand
                    rows = p * n
                    
                r_idx = random.randint(0, rows - 1)
                c_idx = random.randint(0, target_rank - 1)
                
                old_val = mat[r_idx][c_idx]
                # Try changes: -2, -1, 1, 2
                change = random.choice([-2, -1, 1, 2])
                new_val = old_val + change
                
                # Bounds: keep values small to avoid explosion
                if abs(new_val) > 5:
                    continue
                    
                mat[r_idx][c_idx] = new_val
                
                new_error = np.sum(get_error(U_cand, V_cand, W_cand) ** 2)
                
                if new_error < current_error:
                    current_error = new_error
                    stuck_count = 0
                    if new_error < 1e-9:
                        found = True
                        break
                else:
                    # Revert
                    mat[r_idx][c_idx] = old_val
                    stuck_count += 1
                    
                # Track best local
                if new_error < best_local_error:
                    best_local_error = new_error
                    best_local_state = (U_cand, V_cand, W_cand)
                
                # If stuck for too long, break and restart
                if stuck_count > 50:
                    break
            
            if found:
                break
        
        if found:
            if target_rank < best_rank:
                best = {"U": U_cand, "V": V_cand, "W": W_cand}
                best_rank = target_rank
            # Continue to lower ranks
        else:
            # If we failed at this rank, it might be hard to go lower, 
            # but let's try a couple more.
            pass

    return best
# EVOLVE-BLOCK-END
