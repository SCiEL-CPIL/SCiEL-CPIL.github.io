"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix
    by a 4x7 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 112).
    2. Attempt to reduce the rank by finding a valid decomposition with fewer terms.
    3. Use a hybrid approach:
       - Try known small-rank constructions if possible (unlikely for 4x4x7 to be much lower than 112 without specific structure, but we search).
       - Use randomized local search to find integer solutions for lower ranks.
       - The search space is constrained by the time limit (150s).
       
    Note: The rank of the matrix multiplication tensor for n x m by m x p is at least max(n*m, m*p, n*p) but can be higher.
    For 4x4 * 4x7:
    n=4, m=4, p=7.
    Schoolbook rank = 112.
    Lower bounds:
    - Max(n*m, m*p, n*p) = max(16, 28, 28) = 28.
    - Strassen-like reductions are possible for square matrices, but here dimensions are uneven.
    - It is highly unlikely that the rank is significantly lower than 112 for such small dimensions without specific algebraic structure that we haven't hardcoded.
    - However, we will attempt to verify if a rank lower than 112 exists via search, but realistically, the schoolbook solution is often optimal or near-optimal for small non-power-of-2 dimensions unless specific algorithms apply.
    
    Given the constraints and the difficulty of finding non-trivial low-rank decompositions for arbitrary small sizes via random search in <150s, we will:
    1. Check a few small ranks (e.g., 28 to 40) with aggressive random search.
    2. If no solution is found, return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-compute indices for speed
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)
    
    # Schoolbook solution as fallback
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
    
    best_solution = None
    best_rank = rank_school
    
    # Define a helper to check validity
    def is_valid(U, V, W, R):
        for i, j, k in indices:
            s = 0
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            for r in range(R):
                s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if s != 1:
                return False
        return True
    
    # Define a helper to calculate error count
    def get_errors(U, V, W, R):
        errors = []
        for i, j, k in indices:
            s = 0
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            for r in range(R):
                s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if s != 1:
                errors.append((i, j, k, s))
        return errors
    
    # Try ranks from a lower bound up to schoolbook rank
    # Lower bound is roughly max(n*m, m*p, n*p) = 28.
    # We will try a few ranks below 112 to see if we can find a solution.
    # Given the computational cost, we will limit the search to a few ranks and restarts.
    
    # We will try ranks in descending order from a conservative estimate to see if we can beat 112.
    # However, finding a solution for rank < 112 is extremely hard without specific structure.
    # Let's try a few ranks: 30, 40, 50, ..., up to 110.
    # To be efficient, we will only do extensive search for very low ranks if we have time,
    # otherwise we rely on the schoolbook.
    
    # Actually, random search for integer solutions in high-dimensional spaces is very inefficient.
    # A more structured approach is needed.
    # For now, we will attempt a limited search for ranks 28 to 40 with many restarts,
    # and if that fails, we return schoolbook.
    
    # Optimization: Use vectorized operations if possible? No, standard library + numpy is allowed.
    # Let's use numpy for faster inner product calculations during validation.
    import numpy as np
    
    # Convert indices to numpy arrays for faster access if we were doing batched ops,
    # but the loop structure is tricky. We'll stick to optimized Python loops for now.
    
    # Try a few specific ranks
    ranks_to_try = [28, 30, 32, 35, 40, 50, 60, 80, 100]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        if R >= best_rank:
            continue
            
        # Try multiple restarts for this rank
        max_restarts = 20 if R < 40 else 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random integers
            # Using 0, 1, -1 is often better for integer tensor decompositions
            U = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(p * n)]
            
            # Local search
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                errors = get_errors(U, V, W, R)
                
                if not errors:
                    # Valid solution found!
                    best_rank = R
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
                    
                # Pick a random error
                i, j, k, s = random.choice(errors)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Try to fix by adjusting one variable in a random term r
                r = random.randint(0, R - 1)
                
                # Current contribution of term r to this index:
                u_r = U[u_idx][r]
                v_r = V[v_idx][r]
                w_r = W[w_idx][r]
                contrib = u_r * v_r * w_r
                
                # We want the new sum to be 1.
                # Current sum is s.
                # New sum = s - contrib + new_contrib
                # 1 = s - contrib + new_contrib
                # new_contrib = 1 - s + contrib
                target_contrib = 1 - s + contrib
                
                # We can adjust U, V, or W.
                # If we adjust U[u_idx][r] to new_u:
                # new_u * v_r * w_r = target_contrib
                # If v_r * w_r != 0, new_u = target_contrib / (v_r * w_r)
                
                # Try U
                denom = v_r * w_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_u = target_contrib // denom
                        # Check bounds to keep numbers small
                        if -5 <= new_u <= 5:
                            U[u_idx][r] = new_u
                            continue
                
                # Try V
                denom = u_r * w_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_v = target_contrib // denom
                        if -5 <= new_v <= 5:
                            V[v_idx][r] = new_v
                            continue
                
                # Try W
                denom = u_r * v_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_w = target_contrib // denom
                        if -5 <= new_w <= 5:
                            W[w_idx][r] = new_w
                            continue
                
                # If no simple fix was found, try a random perturbation
                # Pick a random variable and change it by +/- 1
                choice = random.randint(0, 2)
                if choice == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    U[idx][col] += random.choice([-1, 1])
                elif choice == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    V[idx][col] += random.choice([-1, 1])
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    W[idx][col] += random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            break

    if best_solution:
        return best_solution
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
