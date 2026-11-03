import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Schoolbook decomposition (Rank 60)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # The tensor T for matrix multiplication of (n x m) and (m x p) matrices is defined such that:
    # C_{i, k} = sum_{j} A_{i, j} B_{j, k}
    # In the tensor decomposition framework:
    # T_{i, j, k} = 1 for all i, j, k is NOT correct.
    # The tensor is actually a 3-way tensor T with indices (i, j, k) where:
    # T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to the product.
    # Wait, the standard tensor decomposition of matrix multiplication M_{n,m,p} is:
    # T_{i, j, k} = 1 for all i in n, j in m, k in p?
    # No. The tensor T is such that <T, A \otimes B> = vec(C).
    # The term A_{i, j} B_{j, k} appears in C_{i, k}.
    # So the coefficient of A_{i, j} B_{j, k} in the expansion of the linear map is 1.
    # This means T_{i, j, k} = 1 for all valid indices?
    # If T_{i, j, k} = 1, then the rank is at least n*m*p?
    # No, the rank of the tensor T defined by T_{i, j, k} = 1 is not n*m*p.
    # Let's check the decomposition:
    # sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} = 1 ?
    # If this holds for all i, j, k, then the decomposition is valid.
    # For the schoolbook algorithm, we have rank R = n*m*p.
    # U_{i, j, r} is 1 if r corresponds to (i, j, k'), 0 otherwise.
    # This works because sum_r ... = 1 for the specific r where indices match.
    
    # Can we do better?
    # For 2x2 matrix multiplication, the rank is 7 (Strassen), which is less than 8 (schoolbook).
    # For 3x4 by 4x5, we can potentially use block decomposition.
    # However, implementing Strassen for non-square matrices and combining blocks is complex.
    # A simpler heuristic: Random search for a lower rank decomposition.
    
    # We will attempt to find a decomposition of rank R < 60.
    # We'll try R = 59, 58, ... down to some limit.
    # For each R, we initialize U, V, W with random integers and use a local search or solver.
    # Given the time limit, we can only try a few ranks with a limited number of iterations.
    
    # Let's try a simple gradient descent on the loss function:
    # L = sum_{i, j, k} (sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} - 1)^2
    
    # Note: The indices in the task are:
    # U: (n*m, R) -> U[i*m + j, r]
    # V: (m*p, R) -> V[j*p + k, r]
    # W: (p*n, R) -> W[k*n + i, r]
    # The contraction is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i, j, k.
    
    # Let's verify the schoolbook solution with this formula.
    # For schoolbook, R = n*m*p.
    # Let r = i*m*p + j*p + k.
    # U[i*m+j, r] = 1 if (i, j) matches, 0 else.
    # V[j*p+k, r] = 1 if (j, k) matches, 0 else.
    # W[k*n+i, r] = 1 if (k, i) matches, 0 else.
    # For a fixed (i, j, k), only the term r corresponding to (i, j, k) is non-zero in all three.
    # So the sum is 1*1*1 = 1. Correct.
    
    # Now, let's try to optimize for lower ranks.
    # We'll use a simple random restart + local search approach.
    
    def try_rank(R, max_iters=1000):
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, (n*m, R))
        V = np.random.randint(-1, 2, (m*p, R))
        W = np.random.randint(-1, 2, (p*n, R))
        
        # Compute the tensor T_rec
        # T_rec[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We can compute this efficiently using matrix multiplications.
        # Let A = U.T (R x n*m), B = V (m*p x R), C = W.T (R x p*n)
        # This is a 3-way contraction.
        # We can flatten i, j, k.
        
        # Actually, we can just loop for small dimensions.
        # n=3, m=4, p=5. Total terms 60.
        # R is around 50-60.
        # 60 * R multiplications. Very fast.
        
        def compute_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1)**2
            return err
        
        current_err = compute_error()
        
        # Local search: flip random entries
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and a random entry
            mat_choice = np.random.choice(3)
            if mat_choice == 0:
                idx = np.random.randint(0, n*m)
                r = np.random.randint(0, R)
                old_val = U[idx, r]
                U[idx, r] = -U[idx, r] if old_val != 0 else 1 # Flip sign or set to 1
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    U[idx, r] = old_val
            elif mat_choice == 1:
                idx = np.random.randint(0, m*p)
                r = np.random.randint(0, R)
                old_val = V[idx, r]
                V[idx, r] = -V[idx, r] if old_val != 0 else 1
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    V[idx, r] = old_val
            else:
                idx = np.random.randint(0, p*n)
                r = np.random.randint(0, R)
                old_val = W[idx, r]
                W[idx, r] = -W[idx, r] if old_val != 0 else 1
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    W[idx, r] = old_val
            
            if current_err == 0:
                # Found an exact solution
                return U, V, W
        
        return None

    # Try ranks from 59 down to 50
    # We might find a solution with rank < 60.
    # Given the time, we'll try a few ranks.
    
    for R in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(R, max_iters=500)
            if result is not None:
                U, V, W = result
                # Convert to list of lists of integers
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
    # If no better solution found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
