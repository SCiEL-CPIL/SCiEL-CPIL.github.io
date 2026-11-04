"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    """
    Return (U_local, V_local, W_local) for 2x2x2 Strassen decomposition.
    
    U_local: shape (4, 7) for (i,a) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    V_local: shape (4, 7) for (b,k) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    W_local: shape (4, 7) for (k,i) in {0,0:0, 0,1:1, 1,0:2, 1,1:3} i.e. k*2+i
    
    The constraint: for all i,a,b,k in {0,1}:
      sum_c W_local[k*2+i, c] * U_local[i*2+a, c] * V_local[b*2+k, c] = delta[a,b]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M1 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2: M2 = (A00)(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # c=3: M3 = (A11)(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # c=4: M4 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # c=5: M5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # c=6: M6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the factor matrices correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    err = abs(s - expected)
                    if err > max_err:
                        max_err = err
                    if err > tol:
                        return False, max_err
    return True, max_err


def _build_decomposition(I_blocks, J_blocks, K_blocks, U_str, V_str, W_str):
    """Build decomposition from block partitions."""
    n, m, p = DIMENSIONS
    
    total_rank = 0
    block_info = []
    
    for (i_start, i_size) in I_blocks:
        for (j_start, j_size) in J_blocks:
            for (k_start, k_size) in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    rank_contrib = 7
                    method = 'strassen'
                else:
                    rank_contrib = i_size * j_size * k_size
                    method = 'schoolbook'
                total_rank += rank_contrib
                block_info.append((i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method))
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for (i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method) in block_info:
        if method == 'strassen':
            for local_i in range(2):
                for local_a in range(2):
                    global_i = i_start + local_i
                    global_a = j_start + local_a
                    global_u_row = global_i * m + global_a
                    for c in range(7):
                        U[global_u_row, col_offset + c] = U_str[local_i * 2 + local_a, c]
            
            for local_b in range(2):
                for local_k in range(2):
                    global_b = j_start + local_b
                    global_k = k_start + local_k
                    global_v_row = global_b * p + global_k
                    for c in range(7):
                        V[global_v_row, col_offset + c] = V_str[local_b * 2 + local_k, c]
            
            for local_k in range(2):
                for local_i in range(2):
                    global_k = k_start + local_k
                    global_i = i_start + local_i
                    global_w_row = global_k * n + global_i
                    for c in range(7):
                        W[global_w_row, col_offset + c] = W_str[local_k * 2 + local_i, c]
            
            col_offset += 7
        else:
            for local_i in range(i_size):
                for local_a in range(j_size):
                    for local_k in range(k_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        global_k = k_start + local_k
                        
                        global_u_row = global_i * m + global_a
                        global_v_row = global_a * p + global_k
                        global_w_row = global_k * n + global_i
                        
                        U[global_u_row, col_offset] = 1
                        V[global_v_row, col_offset] = 1
                        W[global_w_row, col_offset] = 1
                        col_offset += 1
    
    return U, V, W, total_rank


def _generate_partitions(size):
    """Generate all possible partitions of a dimension into blocks of size 1 or 2."""
    if size == 0:
        return [[]]
    if size == 1:
        return [[(0, 1)]]
    if size == 2:
        return [[(0, 2)], [(0, 1), (1, 1)]]
    
    results = []
    # Try starting with block size 2
    if size >= 2:
        for rest in _generate_partitions(size - 2):
            results.append([(0, 2)] + [(start + 2, sz) for start, sz in rest])
    # Try starting with block size 1
    for rest in _generate_partitions(size - 1):
        results.append([(0, 1)] + [(start + 1, sz) for start, sz in rest])
    
    return results


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication
    by searching over block partition strategies.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Precompute Strassen factors
    U_str, V_str, W_str = _strassen_2x2x2_factors()
    
    # Generate all possible partitions
    I_partitions = _generate_partitions(n)
    J_partitions = _generate_partitions(m)
    K_partitions = _generate_partitions(p)
    
    best_rank = n * m * p  # Schoolbook fallback
    best_U, best_V, best_W = None, None, None
    
    # Try all combinations of partitions
    for I_blocks in I_partitions:
        if time.time() > deadline:
            break
        for J_blocks in J_partitions:
            if time.time() > deadline:
                break
            for K_blocks in K_partitions:
                if time.time() > deadline:
                    break
                
                U, V, W, rank = _build_decomposition(I_blocks, J_blocks, K_blocks, U_str, V_str, W_str)
                
                if rank < best_rank:
                    # Verify before accepting
                    valid, max_err = _verify_decomposition(U, V, W, n, m, p)
                    if valid:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                    else:
                        # If invalid, don't update best
                        pass
    
    # If no valid decomposition found, fall back to schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n * m, rank))
        best_V = np.zeros((m * p, rank))
        best_W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1
                    best_V[j * p + k, col] = 1
                    best_W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists with integer/half-integer entries
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                rounded = round(val)
                if abs(val - rounded) < 1e-9:
                    row_list.append(int(rounded))
                else:
                    half_rounded = round(val * 2) / 2
                    row_list.append(half_rounded)
            result.append(row_list)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
