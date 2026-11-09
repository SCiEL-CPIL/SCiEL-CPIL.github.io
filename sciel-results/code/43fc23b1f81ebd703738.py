"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def build_strassen_partition(j_blocks, k_blocks):
        """Build a Strassen-based decomposition with given j and k block sizes.
        j_blocks: list of (start_j, size_j) pairs
        k_blocks: list of (start_k, size_k) pairs
        Each block must have size_j <= 2 and size_k <= 2 for Strassen to apply.
        """
        rank = 0
        for _, sj in j_blocks:
            for _, sk in k_blocks:
                rank += 7 * sj * sk  # Each 2x2x2 block costs 7
        # Actually need to be more careful - let's just build it directly
        U = []
        V = []
        W = []
        for i in range(n):
            U.append([0] * rank)
        for j in range(m):
            V.append([0] * rank)
        for k in range(p):
            W.append([0] * rank)
        
        col = 0
        for (j1, sj) in j_blocks:
            for (k1, sk) in k_blocks:
                if sj == 2 and sk == 2:
                    # Strassen block for 2x2x2
                    for u_pat, v_pat, w_pat in strassen_terms:
                        for (di, dj), coeff in u_pat.items():
                            U[(di) * m + (j1 + dj)][col] += coeff
                        for (dj, dk), coeff in v_pat.items():
                            V[(j1 + dj) * p + (k1 + dk)][col] += coeff
                        for (di, dk), coeff in w_pat.items():
                            W[(k1 + dk) * n + (di)][col] += coeff
                        col += 1
                elif sj == 2 and sk == 1:
                    # Schoolbook 2x2x1
                    for di in range(2):
                        for dj in range(2):
                            U[di * m + (j1 + dj)][col] = 1
                            V[(j1 + dj) * p + k1][col] = 1
                            W[k1 * n + di][col] = 1
                            col += 1
                elif sj == 1 and sk == 2:
                    # Schoolbook 1x2x2
                    for dj in range(2):
                        for dk in range(2):
                            U[0 * m + (j1)][col] = 1
                            V[(j1) * p + (k1 + dk)][col] = 1
                            W[(k1 + dk) * n + 0][col] = 1
                            col += 1
                elif sj == 1 and sk == 1:
                    # Single element
                    U[0 * m + j1][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k1 * n + 0][col] = 1
                    col += 1
                else:
                    raise ValueError("Block sizes must be <= 2")
        
        return U, V, W, col
    
    def check_decomposition(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = sum(U[i*m+j1][r] * V[j2*p+k][r] * W[k*n+i][r] for r in range(rank))
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(val - target) > tol:
                            return False
        return True
    
    def to_half_int_list(arr):
        """Convert to list of lists with half-integer values."""
        result = []
        for i in range(len(arr)):
            row = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-9:
                    if abs(rounded - int(rounded)) < 1e-9:
                        row.append(int(rounded))
                    else:
                        row.append(rounded)
                else:
                    row.append(rounded)
            result.append(row)
        return result
    
    # Build the standard rank-36 decomposition
    # Partition: j = [0,1],[2,3] and k = [0,1],[2,3],[4]
    j_blocks = [(0, 2), (2, 2)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]
    U, V, W, rank = build_strassen_partition(j_blocks, k_blocks)
    
    # Verify
    assert check_decomposition(U, V, W, rank), "Initial decomposition failed"
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by removing columns
    def try_remove_columns(U, V, W, rank, max_removals=5):
        """Try to remove columns one at a time, checking validity after each removal."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        current_rank = rank
        removed = 0
        
        while removed < max_removals and current_rank > 1:
            if time.time() - start_time > deadline - 10:
                break
            
            found = False
            for col_to_remove in range(current_rank):
                if time.time() - start_time > deadline - 10:
                    break
                
                U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
                V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
                W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
                
                if check_decomposition(U_new, V_new, W_new, current_rank - 1):
                    U = U_new
                    V = V_new
                    W = W_new
                    current_rank -= 1
                    removed += 1
                    found = True
                    break
            
            if not found:
                break
        
        return U, V, W, current_rank
    
    # Try column removal from the rank-36 decomposition
    U_reduced, V_reduced, W_reduced, rank_reduced = try_remove_columns(U, V, W, rank, max_removals=10)
    
    if rank_reduced < best_rank:
        best_rank = rank_reduced
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Try alternative partitionings
    alternative_partitions = [
        # Different j/k splits
        ([(0, 2), (2, 2)], [(0, 1), (1, 1), (2, 2), (4, 1)]),
        ([(0, 1), (1, 1), (2, 2)], [(0, 2), (2, 2), (4, 1)]),
        ([(0, 2), (2, 2)], [(0, 2), (2, 1), (3, 1), (4, 1)]),
        ([(0, 1), (1, 2), (3, 1)], [(0, 2), (2, 2), (4, 1)]),
        ([(0, 2), (2, 1), (3, 1)], [(0, 2), (2, 2), (4, 1)]),
    ]
    
    for j_b, k_b in alternative_partitions:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            U_alt, V_alt, W_alt, rank_alt = build_strassen_partition(j_b, k_b)
            if check_decomposition(U_alt, V_alt, W_alt, rank_alt):
                U_r, V_r, W_r, rank_r = try_remove_columns(U_alt, V_alt, W_alt, rank_alt, max_removals=5)
                if rank_r < best_rank:
                    best_rank = rank_r
                    best_U, best_V, best_W = U_r, V_r, W_r
        except Exception:
            pass
    
    # Try local optimization to find lower-rank decompositions
    # Use least_squares to try to fit a rank-r decomposition
    def try_optimize_rank(target_rank, num_seeds=5):
        """Try to find a valid decomposition of the given rank via optimization."""
        if time.time() - start_time > deadline - 15:
            return None, None, None, None
        
        for seed in range(num_seeds):
            if time.time() - start_time > deadline - 15:
                break
            
            np.random.seed(seed * 7919 + target_rank * 104729)
            
            U0 = np.random.randn(n*m, target_rank) * 0.3
            V0 = np.random.randn(m*p, target_rank) * 0.3
            W0 = np.random.randn(p*n, target_rank) * 0.3
            
            def residuals(x):
                U = x[:n*m*target_rank].reshape(n*m, target_rank)
                V = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                val = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                                target = 1.0 if j1 == j2 else 0.0
                                res.append(val - target)
                return np.array(res)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=3000)
                if result.cost < 1e-8:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers and check
                    U_rounded = np.round(U_opt * 2) / 2.0
                    V_rounded = np.round(V_opt * 2) / 2.0
                    W_rounded = np.round(W_opt * 2) / 2.0
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if check_decomposition(U_list, V_list, W_list, target_rank):
                        return U_list, V_list, W_list, target_rank
            except Exception:
                pass
        
        return None, None, None, None
    
    # Try ranks from best_rank-1 downward
    for target_rank in range(best_rank - 1, max(best_rank - 5, 20), -1):
        if time.time() - start_time > deadline - 20:
            break
        
        U_opt, V_opt, W_opt, r_opt = try_optimize_rank(target_rank, num_seeds=3)
        if U_opt is not None and r_opt < best_rank:
            best_rank = r_opt
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            break  # Found a lower rank, stop
    
    # Final polish: try to remove more columns from the best found
    if time.time() - start_time < deadline - 5:
        U_final, V_final, W_final, rank_final = try_remove_columns(
            [row[:] for row in best_U], 
            [row[:] for row in best_V], 
            [row[:] for row in best_W], 
            best_rank, 
            max_removals=3
        )
        if rank_final < best_rank:
            best_rank = rank_final
            best_U, best_V, best_W = U_final, V_final, W_final
    
    return {
        "U": to_half_int_list(best_U),
        "V": to_half_int_list(best_V),
        "W": to_half_int_list(best_W)
    }
# EVOLVE-BLOCK-END
