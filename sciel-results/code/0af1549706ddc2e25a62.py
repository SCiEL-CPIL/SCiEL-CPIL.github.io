import time
import random
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42
    
    # Trivial decomposition as baseline
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = trivial_rank
    best_result = {"U": U, "V": V, "W": W}
    
    def check_decomposition(U_r, V_r, W_r, rank):
        """Check if decomposition is valid: sum_t U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                    if s != 1:
                        return False
        return True

    def try_merge(U_r, V_r, W_r, rank, t1, t2):
        """
        Try to merge column t2 into column t1 (or another column) by adjusting coefficients.
        This is a simplified local search step.
        We try to find a column t3 such that we can set V_r[:, t2] = 0 by adding a multiple of V_r[:, t3].
        This requires V_r[:, t2] to be a multiple of V_r[:, t3].
        """
        # Check if V_r[:, t2] is a scalar multiple of any other V_r[:, t]
        # If so, we can potentially eliminate t2.
        for t3 in range(rank):
            if t3 == t2:
                continue
            
            # Check if V_r[:, t2] = c * V_r[:, t3] for some c
            # Find c from the first non-zero element
            c = None
            valid = True
            for idx in range(m * p):
                if V_r[idx][t3] != 0:
                    if V_r[idx][t2] % V_r[idx][t3] != 0:
                        valid = False
                        break
                    c = V_r[idx][t2] // V_r[idx][t3]
                    break
                elif V_r[idx][t2] != 0:
                    valid = False
                    break
            
            if not valid or c is None:
                continue
            
            # Verify c for all elements
            for idx in range(m * p):
                if V_r[idx][t2] != c * V_r[idx][t3]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If valid, we can eliminate t2 by adding -c * U_r[:, t3] to U_r[:, t2] 
            # and -c * W_r[:, t3] to W_r[:, t2]? 
            # No, the term is U_t2 * V_t2 * W_t2.
            # If V_t2 = c * V_t3, then U_t2 * V_t2 * W_t2 = U_t2 * c * V_t3 * W_t2.
            # We want to absorb this into the t3 term: U_t3 * V_t3 * W_t3.
            # This requires U_t2 * c * W_t2 to be added to U_t3 * W_t3? 
            # No, the structure is fixed. 
            # We can only eliminate t2 if U_t2 * W_t2 (outer product) can be added to U_t3 * W_t3 (outer product) 
            # such that the total tensor remains the same.
            # Specifically, the contribution of term t2 is (U_t2 * W_t2) outer V_t2.
            # The contribution of term t3 is (U_t3 * W_t3) outer V_t3.
            # If V_t2 = c * V_t3, then contribution t2 = c * (U_t2 * W_t2) outer V_t3.
            # We can add c * (U_t2 * W_t2) to the "U*W" part of term t3?
            # But U and W are separate matrices. We can modify U_t3 and W_t3?
            # No, we can modify U_t3 and W_t3 independently.
            # Let U_new = U_t3 + c * U_t2 ? No, that changes the U*W product.
            # (U_t3 + c*U_t2) * (W_t3 + d*W_t2) ... this is getting complicated.
            
            # Simpler: If V_t2 is a multiple of V_t3, we can just scale U_t2 and W_t2?
            # No, we want to REMOVE t2.
            # This approach is flawed for general tensors.
            
            pass
        
        return None

    def try_rank(target_rank, max_attempts=1000):
        """Try to find a valid decomposition of given rank"""
        if time.time() > deadline - 5:
            return None
        
        reduction = trivial_rank - target_rank
        
        # Strategy: Start from trivial, randomly zero out 'reduction' columns, 
        # then try to fix the remaining columns to compensate.
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Select columns to keep
            kept_indices = random.sample(range(trivial_rank), target_rank)
            
            U_r = [[0] * target_rank for _ in range(n * m)]
            V_r = [[0] * target_rank for _ in range(m * p)]
            W_r = [[0] * target_rank for _ in range(p * n)]
            
            # Copy kept columns
            for new_col, old_col in enumerate(kept_indices):
                for i in range(n * m):
                    U_r[i][new_col] = U[i][old_col]
                for j in range(m * p):
                    V_r[j][new_col] = V[j][old_col]
                for k in range(p * n):
                    W_r[k][new_col] = W[k][old_col]
            
            # Now, try to adjust the kept columns to satisfy the equations.
            # The current sum is 1 for all (i,j,k) where the corresponding trivial term was kept.
            # It is 0 for the removed ones.
            # We need to change the values so that ALL (i,j,k) sum to 1.
            
            # This is a system of nonlinear equations.
            # We use a simple gradient-free search: randomly perturb entries and accept if error decreases.
            
            def calculate_error(U_curr, V_curr, W_curr, rank_curr):
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for t in range(rank_curr):
                                s += U_curr[i * m + j][t] * V_curr[j * p + k][t] * W_curr[k * n + i][t]
                            error += (s - 1) ** 2
                return error
            
            current_error = calculate_error(U_r, V_r, W_r, target_rank)
            
            # If error is 0, we are done
            if current_error == 0:
                return {"U": U_r, "V": V_r, "W": W_r}
            
            # Local search
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Randomly pick a matrix, row, col
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, target_rank-1)
                    delta = random.choice([-1, 1])
                    U_r[row][col] += delta
                elif mat_choice == 'V':
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, target_rank-1)
                    delta = random.choice([-1, 1])
                    V_r[row][col] += delta
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, target_rank-1)
                    delta = random.choice([-1, 1])
                    W_r[row][col] += delta
                
                new_error = calculate_error(U_r, V_r, W_r, target_rank)
                
                if new_error == 0:
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # Accept if better, or with small probability (simulated annealing lite)
                if new_error < current_error or random.random() < 0.05:
                    current_error = new_error
                else:
                    # Revert
                    if mat_choice == 'U':
                        U_r[row][col] -= delta
                    elif mat_choice == 'V':
                        V_r[row][col] -= delta
                    else:
                        W_r[row][col] -= delta
        
        return None

    # Search for better ranks
    # Start from just below trivial and go down
    for target_rank in range(trivial_rank - 1, max(20, trivial_rank - 15), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, max_attempts=200)
        if result is not None:
            best_rank = target_rank
            best_result = result
            # Found a better rank, continue searching lower
            continue
    
    return best_result
