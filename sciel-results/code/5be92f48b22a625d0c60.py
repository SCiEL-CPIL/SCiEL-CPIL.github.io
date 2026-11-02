"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    import numpy as np

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        if rank == 0:
            return False
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        """Check if entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    def verify_full(U, V, W):
        """Full verification: correct tensor and integer/half-integer entries."""
        if not verify_fast(U, V, W):
            return False
        if not verify_integer(U, V, W):
            return False
        return True

    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy 1: Greedy removal from schoolbook
    def greedy_reduce(U_init, V_init, W_init, max_time=30):
        """Try to reduce rank by removing dependent terms."""
        U_np = np.array(U_init, dtype=float)
        V_np = np.array(V_init, dtype=float)
        W_np = np.array(W_init, dtype=float)
        current_rank = U_np.shape[1]
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            if current_rank <= 28:
                break
            
            found_removal = False
            # Try each possible term to remove, in random order
            order = list(range(current_rank))
            random.shuffle(order)
            
            for c_remove in order:
                if time.time() - t_start > max_time:
                    break
                
                mask = [i for i in range(current_rank) if i != c_remove]
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]
                
                U_r = U_test.reshape(n, m, -1)
                V_r = V_test.reshape(m, p, -1)
                W_r = W_test.reshape(p, n, -1)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                if np.allclose(T, 1.0, atol=1e-9):
                    if verify_integer(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                        U_np = U_test
                        V_np = V_test
                        W_np = W_test
                        current_rank -= 1
                        found_removal = True
                        break
            
            if not found_removal:
                break
        
        return U_np.tolist(), V_np.tolist(), W_np.tolist(), current_rank

    # Apply greedy reduction
    if time.time() < deadline - 60:
        U_reduced, V_reduced, W_reduced, rank_reduced = greedy_reduce(
            best_result["U"], best_result["V"], best_result["W"], max_time=25
        )
        if rank_reduced < best_rank:
            best_result = {"U": U_reduced, "V": V_reduced, "W": W_reduced}
            best_rank = rank_reduced

    # Strategy 2: Structured random search
    # Key insight: for matrix multiplication, many entries in U, V, W should be 0
    # Use sparse random matrices with structured non-zero patterns
    
    value_pool = [0, 1, -1, 0.5, -0.5]
    sparse_pool = [0, 0, 0, 1, -1]  # biased toward zeros

    lower_bound = max(n*m, m*p, p*n)  # 28
    
    for target_rank in range(lower_bound, min(best_rank + 1, 50)):
        if time.time() > deadline - 5:
            break
        
        # Limit attempts per rank
        max_attempts = 30 if target_rank <= 35 else 15
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate sparse random matrices
            # For U: row i*m+j should have non-zeros only in certain columns
            # For V: row j*p+k should have non-zeros only in certain columns  
            # For W: row k*n+i should have non-zeros only in certain columns
            
            # Use structured sparsity: each row has 1-3 non-zeros
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n * m):
                num_nz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), num_nz)
                for c in cols:
                    U[i][c] = random.choice([1, -1, 0.5, -0.5])
            
            for j in range(m * p):
                num_nz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), num_nz)
                for c in cols:
                    V[j][c] = random.choice([1, -1, 0.5, -0.5])
            
            for k in range(p * n):
                num_nz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), num_nz)
                for c in cols:
                    W[k][c] = random.choice([1, -1, 0.5, -0.5])
            
            if verify_full(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    # Strategy 3: Local search around best found solution
    if best_rank < 84 and time.time() < deadline - 10:
        U_best = np.array(best_result["U"], dtype=float)
        V_best = np.array(best_result["V"], dtype=float)
        W_best = np.array(best_result["W"], dtype=float)
        r = best_rank
        
        # Try perturbing individual entries
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry to perturb
            mat_idx = random.choice([0, 1, 2])
            row_idx = random.randint(0, (n*m if mat_idx == 0 else m*p if mat_idx == 1 else p*n) - 1)
            col_idx = random.randint(0, r - 1)
            
            U_test = U_best.copy()
            V_test = V_best.copy()
            W_test = W_best.copy()
            
            if mat_idx == 0:
                old_val = U_test[row_idx, col_idx]
                U_test[row_idx, col_idx] = random.choice(value_pool)
            elif mat_idx == 1:
                old_val = V_test[row_idx, col_idx]
                V_test[row_idx, col_idx] = random.choice(value_pool)
            else:
                old_val = W_test[row_idx, col_idx]
                W_test[row_idx, col_idx] = random.choice(value_pool)
            
            if verify_full(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                # Try to reduce rank
                U_list = U_test.tolist()
                V_list = V_test.tolist()
                W_list = W_test.tolist()
                U_r2, V_r2, W_r2, rank_r2 = greedy_reduce(U_list, V_list, W_list, max_time=5)
                if rank_r2 < best_rank:
                    best_result = {"U": U_r2, "V": V_r2, "W": W_r2}
                    best_rank = rank_r2
                    U_best = np.array(U_r2, dtype=float)
                    V_best = np.array(V_r2, dtype=float)
                    W_best = np.array(W_r2, dtype=float)
                    r = best_rank

    return best_result
# EVOLVE-BLOCK-END
