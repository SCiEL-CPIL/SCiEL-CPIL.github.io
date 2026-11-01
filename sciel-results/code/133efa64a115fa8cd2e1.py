import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Target Tensor T for matrix multiplication C = A B
    # Dimensions: A is n x m (2x4), B is m x p (4x4), C is n x p (2x4)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    # Indices:
    # u = i * m + j  (i in 0..n-1, j in 0..m-1)
    # v = j * p + k  (j in 0..m-1, k in 0..p-1)
    # w = k * n + i  (k in 0..p-1, i in 0..n-1) -- Note: problem states W[k*n+i, r]
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Construct Target Tensor T (sparse representation for efficiency)
    # T[u, v, w] = 1 if valid term, 0 otherwise
    # We only need to check the non-zero entries for verification
    # Non-zero entries are for each (i, j, k)
    nz_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nz_entries.append((u_idx, v_idx, w_idx))
    
    def verify(U, V, W):
        # U, V, W are numpy arrays
        # Check if sum_r U[u,r] V[v,r] W[w,r] == 1 for all nz entries
        # And 0 for all other entries? 
        # The problem says "exactly decompose". 
        # Usually this means the tensor is identical.
        # Let's compute the full tensor for small sizes.
        # 8x16x8 = 1024 entries. Rank ~20-30. 
        # np.einsum is fast enough.
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        T_target = np.zeros((size_u, size_v, size_w))
        for u, v, w in nz_entries:
            T_target[u, v, w] = 1.0
        
        # Check for exact match (within tolerance for float errors, but we want exact integers/half-integers)
        if not np.allclose(T_calc, T_target, atol=1e-6):
            return False
        
        # Check if entries are integer or half-integer
        # The problem states: "with integer/half-integer entries"
        # We should ensure the returned matrices satisfy this.
        # If the optimization finds real numbers, we must round them.
        # Rounding might break the decomposition.
        # Strategy: Find a solution with small integer coefficients if possible, or round and verify.
        
        return True

    def is_half_integer(X):
        # Check if all entries are close to k/2 for integer k
        # Multiply by 2, check if close to integer
        X2 = X * 2.0
        return np.allclose(X2, np.round(X2), atol=1e-4)

    def round_to_half_integer(X):
        return np.round(X * 2.0) / 2.0

    # Fallback: Rank 32 (Schoolbook)
    U32 = np.zeros((size_u, 32))
    V32 = np.zeros((size_v, 32))
    W32 = np.zeros((size_w, 32))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U32[u_idx, col] = 1
                V32[v_idx, col] = 1
                W32[w_idx, col] = 1
                col += 1
                
    best_U, best_V, best_W = U32, V32, W32
    best_rank = 32

    # Strategy: Random Search + Local Optimization for lower ranks
    # We will try ranks from 20 up to 31.
    # For each rank, we initialize randomly and try to minimize the error.
    # We use a simple gradient-free method or local search.
    # Given the non-convexity, random restarts are key.
    
    # To ensure half-integer constraint, we can search in the space of half-integers directly?
    # No, that's a discrete search, too large.
    # Better: Find a real solution, then round to nearest half-integer and verify.
    # If verification fails, we might try to adjust.
    # However, known decompositions (Strassen etc.) have integer coefficients.
    # So rounding should be stable if the solution is close to integers.
    
    # Let's implement a local search / simulated annealing style approach.
    # Objective: Minimize sum (T_calc - T_target)^2
    # Penalty for non-half-integers? No, just round at the end.
    
    def error_func(U, V, W):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        T_target = np.zeros((size_u, size_v, size_w))
        for u, v, w in nz_entries:
            T_target[u, v, w] = 1.0
        return np.sum((T_calc - T_target)**2)

    # To speed up error_func, we can precompute T_target
    T_target = np.zeros((size_u, size_v, size_w))
    for u, v, w in nz_entries:
        T_target[u, v, w] = 1.0

    def fast_error(U, V, W):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        return np.sum((T_calc - T_target)**2)

    # Local Search: 
    # 1. Initialize U, V, W randomly.
    # 2. Iteratively pick a random element in U, V, or W, perturb it, accept if error decreases.
    # 3. Periodically try to round and verify.
    
    np.random.seed(42)
    
    # Try ranks from 20 to 31
    # Rank 20 is theoretically possible (R(2,2,4)=10 -> 2*10=20).
    # Rank 25 is known lower bound.
    # Rank 28 is easy via Strassen blocks.
    
    for target_rank in range(20, 32):
        if time.time() > deadline - 5:
            break
            
        # If we already found a better rank, we might stop early?
        # The task asks for rank-minimal. We should try to find the lowest possible.
        # But we must return the best found within time.
        # So we should iterate ranks from low to high, and keep the first valid one found?
        # Or try all and keep the best?
        # Since lower rank is better, we should try low ranks first.
        # If we find a valid rank 20, we can return it immediately?
        # But maybe rank 19 exists? Unlikely for 2x4x4.
        # Known R(2,4,4) = 25? Or 20?
        # Let's assume 20 is the target. If we find 20, we stop.
        
        found = False
        
        # Number of restarts depends on time
        # Rank 20: 8*20 + 16*20 + 8*20 = 640 variables.
        # Random search in 640D is hard.
        # We need a smarter initialization.
        
        # Initialization Strategy:
        # Use the structure of the problem.
        # For 2x4x4, we can split into two 2x2x4 problems.
        # A = [A1, A2], B = [B1; B2].
        # C = A1 B1 + A2 B2.
        # Each 2x2x4 has rank 10.
        # Total rank 20.
        # We can initialize U, V, W based on known Rank 10 decomposition for 2x2x4.
        
        # Do I have Rank 10 for 2x2x4?
        # R(2,2,4) = 10.
        # Let's try to find it online or derive it.
        # Actually, I can just use random restarts and hope.
        # But 640D random search is unlikely to converge in 150s with simple local search.
        
        # Alternative: Use the Rank 28 Strassen construction as a starting point for rank 28.
        # And try to reduce it to 25 or 20?
        
        # Let's construct Rank 28 explicitly first to have a solid baseline better than 32.
        # Rank 28 Construction:
        # Split 2x4x4 into two 2x2x4? No.
        # Split 2x4x4 into four 2x2x2?
        # A = [[A11, A12], [A21, A22]]? No, A is 2x4.
        # A = [A_L, A_R] where A_L, A_R are 2x2.
        # B = [[B_TL, B_TR], [B_BL, B_BR]] where blocks are 2x2.
        # C = A B = [A_L B_TL + A_R B_BL, A_L B_TR + A_R B_BR].
        # C = [C_L, C_R] where C_L, C_R are 2x2.
        # C_L = A_L B_TL + A_R B_BL.
        # C_R = A_L B_TR + A R B_BR.
        
        # This is two sums of products.
        # We can use Strassen for 2x2x2.
        # But A_L is shared in C_L and C_R.
        # B_TL, B_BL are for C_L. B_TR, B_BR are for C_R.
        
        # Let's use the standard Block Strassen approach.
        # It is known that R(2,4,4) <= 28.
        # Construction:
        # Use 7 bilinear forms for each of the 4 blocks? No.
        
        # Let's just use the random search for Rank 28, 25, 20.
        # But to make it effective, we need a good initialization.
        
        # Let's try a "Greedy" construction for Rank 28.
        # Or just rely on the fact that R(2,4,4)=25 is likely, and 28 is easy.
        
        # I will implement a specific Rank 28 construction using Strassen on 2x2 blocks.
        # Reference: "On the bilinear complexity of matrix multiplication"
        # For 2x4x4, we can view it as multiplying a 2x4 by 4x4.
        # This is equivalent to two 2x2x4 multiplications?
        # No, it's one 2x4x4.
        
        # Let's try to find a Rank 28 solution via random search with a time limit.
        # If we find it, we return it.
        # If not, we return 32.
        
        # To improve search, we can use the fact that the tensor is sparse.
        
        # Let's try Rank 28 first.
        if target_rank == 28:
            # We know 28 is possible.
            # Let's try to find it.
            pass
            
        # Actually, let's just run a generic search for the current target_rank.
        
        # Number of restarts
        max_restarts = 5
        if target_rank > 25:
            max_restarts = 10
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            U = np.random.randn(size_u, target_rank) * 0.1
            V = np.random.randn(size_v, target_rank) * 0.1
            W = np.random.randn(size_w, target_rank) * 0.1
            
            # Local Search
            current_error = fast_error(U, V, W)
            
            # Simple hill climbing
            # We will do a fixed number of steps
            steps = 2000
            step_size = 0.1
            
            for s in range(steps):
                if time.time() > deadline - 10:
                    break
                    
                # Pick a random matrix and element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row = np.random.randint(size_u)
                    col = np.random.randint(target_rank)
                    old_val = U[row, col]
                    U[row, col] += np.random.randn() * step_size
                elif mat_idx == 1:
                    row = np.random.randint(size_v)
                    col = np.random.randint(target_rank)
                    old_val = V[row, col]
                    V[row, col] += np.random.randn() * step_size
                else:
                    row = np.random.randint(size_w)
                    col = np.random.randint(target_rank)
                    old_val = W[row, col]
                    W[row, col] += np.random.randn() * step_size
                
                new_error = fast_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                else:
                    # Revert
                    if mat_idx == 0:
                        U[row, col] = old_val
                    elif mat_idx == 1:
                        V[row, col] = old_val
                    else:
                        W[row, col] = old_val
                
                # Check for convergence
                if current_error < 1e-6:
                    break
            
            if current_error < 1e-4:
                # Round to half-integers
                U_r = round_to_half_integer(U)
                V_r = round_to_half_integer(V)
                W_r = round_to_half_integer(W)
                
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    # Since we are iterating ranks from low to high, if we find a valid one,
                    # we might want to continue searching for lower ranks?
                    # No, we are iterating 20, 21, ...
                    # So if we find rank 20, we should stop?
                    # But maybe we can find rank 19? Unlikely.
                    # Let's assume the first valid rank found is the best we can get quickly.
                    # However, to be safe, we should try to find the absolute minimum.
                    # But time is limited.
                    # If we find rank 20, that's excellent.
                    # If we find rank 28, that's good.
                    # We should break out of the rank loop if we find a "good" rank?
                    # Let's break if we find rank <= 25.
                    if target_rank <= 25:
                        break
                    else:
                        # Continue to see if we can find a lower rank?
                        # No, we are iterating upwards.
                        # If we found 28, we already tried 20-27 and failed.
                        # So we can break.
                        break
                        
        if found:
            break
            
    # If no better rank found, return 32
    # Ensure best_U, best_V, best_W are lists of lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
