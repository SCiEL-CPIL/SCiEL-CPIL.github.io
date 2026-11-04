"""Combined approach: block Strassen (2x2x2) with schoolbook fallback for 3x5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7) indexed by [i*2+a]
    V: (4, 7) indexed by [b*2+k]
    W: (4, 7) indexed by [k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0 (M0): (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1 (M1): (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # c=2 (M2): A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # c=3 (M3): A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = -1
    
    # c=4 (M4): (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    
    # c=5 (M5): (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # c=6 (M6): (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
      sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    """
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _generate_partitions(total, max_part=2):
    """Generate all partitions of total into parts of size 1 or max_part."""
    if total == 0:
        yield []
        return
    if total >= 1:
        for rest in _generate_partitions(total - 1, max_part):
            yield [1] + rest
    if total >= max_part:
        for rest in _generate_partitions(total - max_part, max_part):
            yield [max_part] + rest


def _count_strassen_blocks(I_parts, J_parts, K_parts):
    """Count number of (2,2,2) block combinations."""
    i2 = sum(1 for x in I_parts if x == 2)
    j2 = sum(1 for x in J_parts if x == 2)
    k2 = sum(1 for x in K_parts if x == 2)
    return i2 * j2 * k2


def _try_block_decomposition(n, m, p):
    """
    Construct a decomposition using block Strassen for 2x2x2 sub-blocks
    and schoolbook for remaining blocks.
    Searches over all valid partitions to maximize Strassen blocks.
    """
    # Precompute Strassen factors
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Find best partition
    best_rank = n * m * p  # schoolbook
    best_partition = None
    
    I_parts_list = list(_generate_partitions(n, 2))
    J_parts_list = list(_generate_partitions(m, 2))
    K_parts_list = list(_generate_partitions(p, 2))
    
    for I_parts in I_parts_list:
        for J_parts in J_parts_list:
            for K_parts in K_parts_list:
                n_strassen = _count_strassen_blocks(I_parts, J_parts, K_parts)
                # Total rank = n*m*p - n_strassen (each Strassen saves 1)
                total_rank = n * m * p - n_strassen
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_partition = (I_parts, J_parts, K_parts)
    
    if best_partition is None:
        return None
    
    I_parts, J_parts, K_parts = best_partition
    
    # Build block start positions
    I_starts = []
    pos = 0
    for s in I_parts:
        I_starts.append(pos)
        pos += s
    J_starts = []
    pos = 0
    for s in J_parts:
        J_starts.append(pos)
        pos += s
    K_starts = []
    pos = 0
    for s in K_parts:
        K_starts.append(pos)
        pos += s
    
    # Build U, V, W
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(I_parts)):
        for jj in range(len(J_parts)):
            for kk in range(len(K_parts)):
                i_size = I_parts[ii]
                j_size = J_parts[jj]
                k_size = K_parts[kk]
                i_start = I_starts[ii]
                j_start = J_starts[jj]
                k_start = K_starts[kk]
                
                is_strassen = (i_size == 2 and j_size == 2 and k_size == 2)
                
                if is_strassen:
                    # Place Strassen factors
                    for local_i in range(2):
                        for local_a in range(2):
                            gi = i_start + local_i
                            ga = j_start + local_a
                            u_idx = gi * m + ga
                            for c in range(7):
                                U[u_idx][col + c] = S_U[local_i * 2 + local_a][c]
                    for local_b in range(2):
                        for local_k in range(2):
                            gb = j_start + local_b
                            gk = k_start + local_k
                            v_idx = gb * p + gk
                            for c in range(7):
                                V[v_idx][col + c] = S_V[local_b * 2 + local_k][c]
                    for local_k in range(2):
                        for local_i in range(2):
                            gk = k_start + local_k
                            gi = i_start + local_i
                            w_idx = gk * n + gi
                            for c in range(7):
                                W[w_idx][col + c] = S_W[local_k * 2 + local_i][c]
                    col += 7
                else:
                    # Schoolbook for this block
                    for li in range(i_size):
                        for la in range(j_size):
                            for lk in range(k_size):
                                gi = i_start + li
                                ga = j_start + la
                                gk = k_start + lk
                                u_idx = gi * m + ga
                                v_idx = ga * p + gk
                                w_idx = gk * n + gi
                                U[u_idx][col] = 1
                                V[v_idx][col] = 1
                                W[w_idx][col] = 1
                                col += 1
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x7 matrix.
    
    Returns: {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try block Strassen decomposition first
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
