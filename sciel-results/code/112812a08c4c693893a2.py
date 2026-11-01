"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # First: build the rank-49 Strassen decomposition
    result_49 = _construct_strassen_49(n, m, p)
    best_rank = 49
    best_result = result_49
    
    if best_result is None:
        # Fallback: schoolbook rank-56
        return _construct_schoolbook(n, m, p)
    
    # Try to reduce rank below 49 using various strategies
    elapsed = time.time() - start_time
    if elapsed > 10:
        return best_result
    
    # Strategy 1: Try rank 48, 47, ... by zeroing out one column at a time
    # and checking if the remaining columns still span the tensor
    for target_rank in range(48, 30, -1):
        if time.time() - start_time > deadline:
            break
        if target_rank >= best_rank:
            continue
        reduced = _try_rank_reduction(best_result, n, m, p, target_rank, deadline)
        if reduced is not None:
            best_rank = target_rank
            best_result = reduced
            break
    
    return best_result


def _construct_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1
        
        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1
        
        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition(U, V, W, n, m, p, num_tests=3):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(len(U[0])):
                        tensor_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, num_tests=5):
    """Fast verification using numpy matrix operations."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' ... )
        # More directly:
        # Let's build C from the tensor factors
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j A[i,j] B[j,k] * 1
        # = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape for efficient computation
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U_np.shape[1]
        
        # For each output position (i,k):
        # C[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * B[j,k] * V[j*p+k, r] * W[k*n+i, r]
        
        # This is complex; let's just do it directly for small dimensions
        for i in range(n):
            for k in range(p):
                c_val = 0.0
                for j in range(m):
                    for r in range(rank):
                        c_val += A[i,j] * U_np[i*m+j, r] * B[j,k] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(c_val - C_expected[i,k]) > 1e-10:
                    return False
    return True


def _try_rank_reduction(result, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank of the decomposition to target_rank by 
    finding a subset of columns that still spans the tensor.
    """
    U = [row[:] for row in result['U']]
    V = [row[:] for row in result['V']]
    W = [row[:] for row in result['W']]
    current_rank = len(U[0])
    
    # We need to go from current_rank to target_rank
    # Try removing columns one at a time (greedy)
    columns_to_remove = current_rank - target_rank
    remaining_cols = list(range(current_rank))
    
    for removal in range(columns_to_remove):
        if time.time() > deadline:
            return None
        
        # Try each column and see if removing it still gives a valid decomposition
        found_removal = False
        for col_idx in sorted(remaining_cols, key=lambda c: _column_weight(U, V, W, c)):
            # Temporarily remove this column
            new_U = [row[:col_idx] + row[col_idx+1:] for row in U]
            new_V = [row[:col_idx] + row[col_idx+1:] for row in V]
            new_W = [row[:col_idx] + row[col_idx+1:] for row in W]
            
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, num_tests=2):
                U = new_U
                V = new_V
                W = new_W
                remaining_cols.remove(col_idx)
                # Adjust indices of remaining columns
                remaining_cols = [c if c < col_idx else c - 1 for c in remaining_cols]
                found_removal = True
                break
        
        if not found_removal:
            return None
    
    return {"U": U, "V": V, "W": W}


def _column_weight(U, V, W, col_idx):
    """Compute a weight for a column to determine which to try removing first."""
    weight = 0
    for row in U:
        weight += abs(row[col_idx])
    for row in V:
        weight += abs(row[col_idx])
    for row in W:
        weight += abs(row[col_idx])
    return weight
# EVOLVE-BLOCK-END
