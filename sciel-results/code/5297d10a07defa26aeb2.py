import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # Known Rank 28 Solution from Inspiration
    # This is a combination of Strassen for 2x2 blocks and direct mult for others.
    def get_rank_28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4) -> 2*5=10 terms
        # A is 2x3, B is 3x5.
        # A2 is col 2 (2x1). B2 is row 2 (1x5).
        # C2 = A2 B2.
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
        # Part 2: A1 B13 (j=0,1, k=4) -> 2*2=4 terms
        # A1 is cols 0,1 (2x2). B13 is row 0,1, col 4 (2x1).
        # C13 = A1 B13.
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
                
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen -> 7 terms
        # B11 is rows 0,1, cols 0,1.
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        W[1 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1
        V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1
        V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1
        W[1 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1
        V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen -> 7 terms
        # B12 is rows 0,1, cols 2,3.
        # M1
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M2
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1
        W[3 * n + 1][r] = -1
        r += 1
        
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1
        V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1
        V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1
        W[2 * n + 1][r] = 1
        r += 1
        
        # M5
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1
        W[3 * n + 0][r] = 1
        r += 1
        
        # M6
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1
        V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M7
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}

    # Verification function
    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Get the base solution
    best_sol = get_rank_28_solution()
    base_rank = 28
    
    # If we have time, try to find a lower rank (27)
    # We use a local search starting from a random initialization or a perturbed base.
    # Given the complexity, a pure random search for rank 27 is unlikely to succeed quickly.
    # However, we can try a few restarts with a time limit.
    
    def search_rank_27():
        rank = 27
        # We will use a simple local search
        # Objective: minimize sum (S_ijk - 1)^2
        # S_ijk = sum_s U[s,i,j] V[s,j,k] W[s,i,k]
        
        # Precompute indices for speed
        # U indices: s, i, j
        # V indices: s, j, k
        # W indices: s, i, k
        
        # To make search faster, we use numpy for error calc
        # U: (rank, n, m)
        # V: (rank, m, p)
        # W: (rank, n, p)
        
        max_time = 20 # seconds per attempt, max 2 attempts?
        attempts = 2
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(attempt * 100 + 42)
            
            # Initialize with random half-integers
            # Using a wider range to escape local minima
            vals = np.arange(-3, 3.5, 0.5)
            
            U = np.random.choice(vals, size=(rank, n, m))
            V = np.random.choice(vals, size=(rank, m, p))
            W = np.random.choice(vals, size=(rank, n, p))
            
            S = np.einsum('sij,sjk,sik->ijk', U, V, W)
            error = np.sum((S - 1.0)**2)
            
            # Local search
            # We try to change one entry at a time
            # If error decreases, accept.
            
            max_iters = 50000
            iter_count = 0
            
            # Helper to update S and error quickly
            # We maintain S and error incrementally
            
            while error > 1e-9 and iter_count < max_iters:
                if time.time() > deadline - 10:
                    return None
                
                iter_count += 1
                
                # Randomly pick a tensor and index
                t_id = np.random.randint(3)
                
                if t_id == 0: # U
                    s, i, j = np.random.randint(rank), np.random.randint(n), np.random.randint(m)
                    old_val = U[s, i, j]
                    # Try +/- 0.5
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    # S[i, j, :] changes by delta * V[s, j, :] * W[s, i, :]
                    contrib = delta * (V[s, j, :] * W[s, i, :])
                    
                    old_S_row = S[i, j, :]
                    new_S_row = old_S_row + contrib
                    
                    old_err_part = np.sum((old_S_row - 1.0)**2)
                    new_err_part = np.sum((new_S_row - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        U[s, i, j] = new_val
                        S[i, j, :] = new_S_row
                        error += (new_err_part - old_err_part)
                        
                elif t_id == 1: # V
                    s, j, k = np.random.randint(rank), np.random.randint(m), np.random.randint(p)
                    old_val = V[s, j, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    # S[:, j, k] changes by delta * U[s, :, j] * W[s, :, k]
                    contrib = delta * (U[s, :, j] * W[s, :, k])
                    
                    old_S_col = S[:, j, k]
                    new_S_col = old_S_col + contrib
                    
                    old_err_part = np.sum((old_S_col - 1.0)**2)
                    new_err_part = np.sum((new_S_col - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        V[s, j, k] = new_val
                        S[:, j, k] = new_S_col
                        error += (new_err_part - old_err_part)
                        
                else: # W
                    s, i, k = np.random.randint(rank), np.random.randint(n), np.random.randint(p)
                    old_val = W[s, i, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    # S[i, :, k] changes by delta * U[s, i, :] * V[s, :, k]
                    contrib = delta * (U[s, i, :] * V[s, :, k])
                    
                    old_S_vec = S[i, :, k]
                    new_S_vec = old_S_vec + contrib
                    
                    old_err_part = np.sum((old_S_vec - 1.0)**2)
                    new_err_part = np.sum((new_S_vec - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        W[s, i, k] = new_val
                        S[i, :, k] = new_S_vec
                        error += (new_err_part - old_err_part)
            
            if error < 1e-9:
                # Convert to lists
                U_list = [[U[s, i, j] for s in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[V[s, j, k] for s in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[W[s, i, k] for s in range(rank)] for k in range(p) for i in range(n)]
                
                # Final verification
                if verify(U_list, V_list, W_list, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try to find rank 27
    if time.time() < deadline - 30:
        sol_27 = search_rank_27()
        if sol_27 is not None:
            return sol_27
            
    # If we have even more time, try rank 26? Unlikely to succeed, but maybe 1 attempt.
    # The local search for 27 is already aggressive.
    
    return best_sol
