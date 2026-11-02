"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Precompute index mappings for fast verification
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    
    # Fallback: trivial decomposition with rank = n*m*p
    def trivial_decomposition():
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
    
    def fast_verify(U, V, W, rank):
        """Fast vectorized verification using precomputed index arrays."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        
        # Precompute all triples
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    prod_sum = np.sum(U[u_row, :] * V[v_row, :] * W[w_row, :])
                    errors.append(prod_sum - 1.0)
        
        max_err = max(abs(e) for e in errors)
        return max_err < 1e-10, max_err
    
    def compute_error(U, V, W):
        """Compute total squared error for a candidate decomposition."""
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    prod_sum = np.sum(U[u_row, :] * V[v_row, :] * W[w_row, :])
                    total_err += (prod_sum - 1.0) ** 2
        return total_err
    
    def try_rank_search(rank, time_limit, seed_base=0):
        """Try to find a valid decomposition with given rank using enhanced search."""
        start = time.time()
        np.random.seed(seed_base + rank * 1000)
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Strategy 1: Try structured initializations based on known patterns
        # For matrix multiplication, we can try to exploit the tensor structure
        
        # Try several different initialization strategies
        strategies = [
            'random',       # Random small integers
            'sparse',       # Mostly zeros with some 1s and -1s
            'structured',   # Based on Kronecker-like structure
        ]
        
        for strategy in strategies:
            for attempt in range(300):
                if time.time() - start > time_limit:
                    break
                
                if strategy == 'random':
                    U = np.random.choice([-2, -1, 0, 1, 2], size=(n * m, rank))
                    V = np.random.choice([-2, -1, 0, 1, 2], size=(m * p, rank))
                    W = np.random.choice([-2, -1, 0, 1, 2], size=(p * n, rank))
                elif strategy == 'sparse':
                    # Mostly zeros, with ~30% non-zero entries
                    def sparse_matrix(rows, cols, nnz_frac=0.3):
                        mat = np.zeros((rows, cols), dtype=np.int32)
                        for r in range(rows):
                            for c in range(cols):
                                if np.random.random() < nnz_frac:
                                    mat[r, c] = np.random.choice([-2, -1, 1, 2])
                        return mat
                    U = sparse_matrix(n * m, rank, 0.3)
                    V = sparse_matrix(m * p, rank, 0.3)
                    W = sparse_matrix(p * n, rank, 0.3)
                else:  # structured
                    # Try a pattern based on the fact that matrix multiplication
                    # has a natural tensor structure
                    U = np.zeros((n * m, rank), dtype=np.int32)
                    V = np.zeros((m * p, rank), dtype=np.int32)
                    W = np.zeros((p * n, rank), dtype=np.int32)
                    
                    # Fill with a structured pattern
                    for r in range(rank):
                        # Assign to random rows/columns
                        u_idx = np.random.randint(0, n * m)
                        v_idx = np.random.randint(0, m * p)
                        w_idx = np.random.randint(0, p * n)
                        U[u_idx, r] = np.random.choice([-1, 1])
                        V[v_idx, r] = np.random.choice([-1, 1])
                        W[w_idx, r] = np.random.choice([-1, 1])
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err < 1e-10:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        # If we found a reasonably good candidate, try local search
        if best_U is not None and best_err < 50:
            # Local search: try changing individual entries
            for _ in range(500):
                if time.time() - start > time_limit:
                    break
                
                # Try changing a random entry in U, V, or W
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row, col = np.random.randint(0, n * m), np.random.randint(0, rank)
                    old_val = best_U[row, col]
                    for new_val in [-3, -2, -1, 0, 1, 2, 3]:
                        if new_val == old_val:
                            continue
                        best_U[row, col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err:
                            best_err = err
                        else:
                            best_U[row, col] = old_val
                elif mat_choice == 1:
                    row, col = np.random.randint(0, m * p), np.random.randint(0, rank)
                    old_val = best_V[row, col]
                    for new_val in [-3, -2, -1, 0, 1, 2, 3]:
                        if new_val == old_val:
                            continue
                        best_V[row, col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err:
                            best_err = err
                        else:
                            best_V[row, col] = old_val
                else:  # W
                    row, col = np.random.randint(0, p * n), np.random.randint(0, rank)
                    old_val = best_W[row, col]
                    for new_val in [-3, -2, -1, 0, 1, 2, 3]:
                        if new_val == old_val:
                            continue
                        best_W[row, col] = new_val
                        err = compute_error(best_U, best_V, best_W)
                        if err < best_err:
                            best_err = err
                        else:
                            best_W[row, col] = old_val
                
                if best_err < 1e-10:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        if best_U is not None and best_err < 1e-10:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try ranks from lower to higher
    # Theoretical lower bound for M_{3,3,7} is at least max(n*m, m*p, p*n) = 21
    # Known results suggest rank might be around 25-35
    
    # Use more time for lower ranks
    rank_list = list(range(21, 45)) + [46, 48, 50, 52, 55, 58, 60, 63]
    
    for rank in rank_list:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - 5  # 5s safety margin
        
        if remaining < 2:
            break
        
        # Allocate time proportionally to rank (lower ranks get more time)
        time_for_rank = min(remaining * 0.4, max(2.0, 8.0 - (rank - 21) * 0.15))
        
        # Use a stable seed to avoid overflow and ensure reproducibility
        # We use a simple hash-like combination that stays within 32-bit range
        seed_val = (42 + rank * 1000) % (2**32)
        
        result = try_rank_search(rank, time_limit=time_for_rank, seed_base=seed_val)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to trivial decomposition
    return trivial_decomposition()
# EVOLVE-BLOCK-END
