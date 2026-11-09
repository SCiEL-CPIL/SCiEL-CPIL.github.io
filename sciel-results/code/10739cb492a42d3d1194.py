"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 20 second safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using randomized search
    # Strategy: pick random U, V with small entries, solve for W, verify
    np.random.seed(42)
    
    def try_rank(r, max_attempts, seed_offset=0):
        """Try to find a rank-r decomposition. Returns (U, V, W) if found, else None."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(42 + seed_offset)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U and V with denser, more structured entries
            # Use entries from {-1, 0, 1} but ensure rows have some structure
            U_rand = rng.randint(-1, 2, size=(n*m, r)).astype(float)
            V_rand = rng.randint(-1, 2, size=(m*p, r)).astype(float)
            
            # Ensure each row has at least one non-zero entry
            for idx in range(n*m):
                if np.all(U_rand[idx] == 0):
                    U_rand[idx, rng.randint(r)] = 1.0
            for idx in range(m*p):
                if np.all(V_rand[idx] == 0):
                    V_rand[idx, rng.randint(r)] = 1.0
            
            # For each (i, k), we need to find W[k*n+i, :] such that
            # for each j: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
            # This is a linear system: A @ W_row = b, where A[j,c] = U[i*m+j,c] * V[j*p+k,c], b[j]=1
            
            W_rand = np.zeros((p*n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the system for this (i,k)
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    b = np.ones(m)
                    
                    # Check if the system is consistent and solvable
                    # Use least squares and check residual
                    try:
                        W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
                        # Check if the solution is exact
                        residual = np.linalg.norm(A @ W_row - b)
                        if residual < 1e-10:
                            W_rand[k*n+i, :] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from low to high, with more attempts for lower ranks
    # Known lower bound for 3x5 x 5x5 is at least 25 (from trivial bounds)
    # Try a range of ranks with adaptive attempt counts
    rank_ranges = [
        (25, 50, 200),   # (start, end, max_attempts per rank)
        (30, 40, 300),
        (35, 50, 200),
        (40, 60, 150),
        (50, 75, 100),
    ]
    
    seed_offset = 0
    for r_start, r_end, max_att in rank_ranges:
        for r in range(r_start, min(r_end + 1, best_rank)):
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_att, seed_offset)
            seed_offset += 1
            
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                # If we found a good solution, we can try to polish it
                break
        
        if best_rank <= r_start:
            # Found a solution at or below the start of this range
            break
    
    # Polish pass: if we have a solution, try to reduce rank by perturbing
    if best_rank < rank_school:
        if time.time() < deadline - 5:
            # Try to find a slightly lower rank by local search
            current_U, current_V, current_W = best_result
            current_U_np = np.array(current_U, dtype=float)
            current_V_np = np.array(current_V, dtype=float)
            current_W_np = np.array(current_W, dtype=float)
            
            for r_lower in range(max(20, best_rank - 5), best_rank):
                if time.time() > deadline - 3:
                    break
                
                # Try to project the existing solution to lower rank
                # Use SVD to find a lower-rank approximation that might still work
                # This is heuristic - just try random low-rank matrices near the current ones
                
                rng = np.random.RandomState(1000 + r_lower)
                for attempt in range(50):
                    if time.time() > deadline - 3:
                        break
                    
                    # Start from current solution and perturb
                    U_try = current_U_np[:, :r_lower] + rng.randn(n*m, r_lower) * 0.1
                    V_try = current_V_np[:, :r_lower] + rng.randn(m*p, r_lower) * 0.1
                    
                    W_try = np.zeros((p*n, r_lower))
                    valid = True
                    
                    for i in range(n):
                        for k in range(p):
                            A = np.zeros((m, r_lower))
                            for j in range(m):
                                A[j, :] = U_try[i*m+j, :] * V_try[j*p+k, :]
                            b = np.ones(m)
                            try:
                                W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
                                if np.linalg.norm(A @ W_row - b) < 1e-10:
                                    W_try[k*n+i, :] = W_row
                                else:
                                    valid = False
                                    break
                            except:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        U_list = U_try.tolist()
                        V_list = V_try.tolist()
                        W_list = W_try.tolist()
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best_result = (U_list, V_list, W_list)
                            best_rank = r_lower
                            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
