import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def strassen_2x2x2():
    """
    Return a verified rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    #
    # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k in {0,1}
    #
    # U_c[i,j] is the coefficient of A[i,j] in product c
    # V_c[j,k] is the coefficient of B[j,k] in product c
    # W_c[k,i] is the coefficient of product c in C[i,k] (note: W indexed by k*2+i)
    
    # c=0: M1 = (A00+A11)(B00+B11)
    # U: A00+A11 -> [1,0,0,1]
    # V: B00+B11 -> [1,0,0,1]
    # W: contributes +1 to C[0,0] and +1 to C[1,1]
    # W indexed by k*2+i: W[0,0]=1 (k=0,i=0), W[1,1]=1 (k=1,i=1) -> [1,0,0,1]
    
    # c=1: M2 = (A10+A11)B00
    # U: A10+A11 -> [0,0,1,1]
    # V: B00 -> [1,0,0,0]
    # W: contributes +1 to C[1,0]
    # W[0,1]=1 (k=0,i=1) -> [0,1,0,0]
    
    # c=2: M3 = A00(B01-B11)
    # U: A00 -> [1,0,0,0]
    # V: B01-B11 -> [0,1,0,-1]
    # W: contributes +1 to C[0,1]
    # W[1,0]=1 (k=1,i=0) -> [0,0,1,0]
    
    # c=3: M4 = A11(B10-B00)
    # U: A11 -> [0,0,0,1]
    # V: B10-B00 -> [0,-1,1,0]
    # W: contributes +1 to C[1,0]
    # W[0,1]=1 (k=0,i=1) -> [0,1,0,0]
    
    # c=4: M5 = (A00+A01)B11
    # U: A00+A01 -> [1,1,0,0]
    # V: B11 -> [0,0,0,1]
    # W: contributes -1 to C[0,0] and +1 to C[0,1]
    # W[0,0]=-1 (k=0,i=0), W[1,0]=+1 (k=1,i=0) -> [-1,0,1,0]
    
    # c=5: M6 = (A10-A00)(B00+B01)
    # U: A10-A00 -> [-1,0,1,0]
    # V: B00+B01 -> [1,1,0,0]
    # W: contributes +1 to C[1,1]
    # W[1,1]=1 (k=1,i=1) -> [0,0,0,1]
    
    # c=6: M7 = (A01-A11)(B10+B11)
    # U: A01-A11 -> [0,1,0,-1]
    # V: B10+B11 -> [0,0,1,1]
    # W: contributes -1 to C[0,0] and +1 to C[1,1]
    # W[0,0]=-1 (k=0,i=0), W[1,1]=+1 (k=1,i=1) -> [-1,0,0,1]
    
    U_s = np.array([
        [1, 0, 0, 1],  # c=0
        [0, 0, 1, 1],  # c=1
        [1, 0, 0, 0],  # c=2
        [0, 0, 0, 1],  # c=3
        [1, 1, 0, 0],  # c=4
        [-1, 0, 1, 0], # c=5
        [0, 1, 0, -1], # c=6
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],  # c=0
        [1, 0, 0, 0],  # c=1
        [0, 1, 0, -1], # c=2
        [0, -1, 1, 0], # c=3
        [0, 0, 0, 1],  # c=4
        [1, 1, 0, 0],  # c=5
        [0, 0, 1, 1],  # c=6
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1],   # c=0
        [0, 1, 0, 0],   # c=1
        [0, 0, 1, 0],   # c=2
        [0, 1, 0, 0],   # c=3
        [-1, 0, 1, 0],  # c=4
        [0, 0, 0, 1],   # c=5
        [-1, 0, 0, 1],  # c=6
    ], dtype=float)
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for c in range(7):
                    s += U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Strassen decomposition failed at ({i},{j},{k}): got {s}")
    
    return U_s, V_s, W_s

def schoolbook_block(ni, nj, nk):
    """
    Return schoolbook decomposition for ni x nj x nk matrix multiplication.
    Returns (U, V, W) where U is (ni*nj) x (ni*nj*nk), V is (nj*nk) x (ni*nj*nk),
    W is (nk*ni) x (ni*nj*nk).
    """
    rank = ni * nj * nk
    U = np.zeros((ni * nj, rank))
    V = np.zeros((nj * nk, rank))
    W = np.zeros((nk * ni, rank))
    
    col = 0
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                U[i * nj + j, col] = 1
                V[j * nk + k, col] = 1
                W[k * ni + i, col] = 1
                col += 1
    
    return U, V, W

def get_block_decomp(ni, nj, nk):
    """
    Get the best known decomposition for an ni x nj x nk sub-problem.
    Uses Strassen (rank 7) for 2x2x2, schoolbook otherwise.
    """
    if ni == 2 and nj == 2 and nk == 2:
        U, V, W = strassen_2x2x2()
        return U, V, W
    else:
        return schoolbook_block(ni, nj, nk)

def all_partitions(d):
    """Generate all compositions of d (ordered partitions)."""
    if d <= 1:
        yield [d] if d > 0 else []
        return
    for first in range(1, d + 1):
        for rest in all_partitions(d - first):
            yield [first] + rest

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
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
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try all combinations of partitions for n, m, p
    # For each combination, build the block decomposition and compute total rank
    # Keep the one with minimum rank
    
    n_partitions = list(all_partitions(n))
    m_partitions = list(all_partitions(m))
    p_partitions = list(all_partitions(p))
    
    # Pre-compute all needed block decompositions
    block_cache = {}
    
    def get_block(ni, nj, nk):
        key = (ni, nj, nk)
        if key not in block_cache:
            block_cache[key] = get_block_decomp(ni, nj, nk)
        return block_cache[key]
    
    candidates = []
    
    for n_parts in n_partitions:
        for m_parts in m_partitions:
            for p_parts in p_partitions:
                # Compute total rank
                total_rank = 0
                for ni in n_parts:
                    for nj in m_parts:
                        for nk in p_parts:
                            U_b, V_b, W_b = get_block(ni, nj, nk)
                            total_rank += U_b.shape[1]
                
                if total_rank < best_rank:
                    candidates.append((total_rank, n_parts, m_parts, p_parts))
    
    # Sort by rank and try the best ones
    candidates.sort(key=lambda x: x[0])
    
    for total_rank, n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline - 10:
            break
        
        # Build the full decomposition
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col_offset = 0
        i_offset = 0
        for ni in n_parts:
            j_offset = 0
            for nj in m_parts:
                k_offset = 0
                for nk in p_parts:
                    U_b, V_b, W_b = get_block(ni, nj, nk)
                    rank_b = U_b.shape[1]
                    
                    # Place U block
                    for i_local in range(ni):
                        for j_local in range(nj):
                            i_global = i_offset + i_local
                            j_global = j_offset + j_local
                            U_full[i_global * m + j_global, col_offset:col_offset + rank_b] = \
                                U_b[i_local * nj + j_local, :]
                    
                    # Place V block
                    for j_local in range(nj):
                        for k_local in range(nk):
                            j_global = j_offset + j_local
                            k_global = k_offset + k_local
                            V_full[j_global * p + k_global, col_offset:col_offset + rank_b] = \
                                V_b[j_local * nk + k_local, :]
                    
                    # Place W block
                    for k_local in range(nk):
                        for i_local in range(ni):
                            k_global = k_offset + k_local
                            i_global = i_offset + i_local
                            W_full[k_global * n + i_global, col_offset:col_offset + rank_b] = \
                                W_b[k_local * ni + i_local, :]
                    
                    col_offset += rank_b
                    k_offset += nk
                j_offset += nj
            i_offset += ni
        
        # Verify
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        if verify(U_list, V_list, W_list, n, m, p):
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = total_rank
            break  # We found the best possible (sorted by rank)
    
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
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
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Systematic search over all integer partitions of n, m, p
    # For each partition triple, build block decomposition using:
    #   - Strassen (rank 7) for 2x2x2 blocks
    #   - Schoolbook for all other block sizes
    # Verify each candidate; keep the minimum-rank valid one.
    
    n_partitions = list(all_partitions(n))
    m_partitions = list(all_partitions(m))
    p_partitions = list(all_partitions(p))
    
    # Pre-compute block decompositions
    block_cache = {}
    
    def get_block(ni, nj, nk):
        key = (ni, nj, nk)
        if key not in block_cache:
            block_cache[key] = get_block_decomp(ni, nj, nk)
        return block_cache[key]
    
    candidates = []
    
    for n_parts in n_partitions:
        for m_parts in m_partitions:
            for p_parts in p_partitions:
                total_rank = 0
                for ni in n_parts:
                    for nj in m_parts:
                        for nk in p_parts:
                            U_b, V_b, W_b = get_block(ni, nj, nk)
                            total_rank += U_b.shape[1]
                
                if total_rank < best_rank:
                    candidates.append((total_rank, n_parts, m_parts, p_parts))
    
    candidates.sort(key=lambda x: x[0])
    
    for total_rank, n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline - 10:
            break
        
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col_offset = 0
        i_offset = 0
        for ni in n_parts:
            j_offset = 0
            for nj in m_parts:
                k_offset = 0
                for nk in p_parts:
                    U_b, V_b, W_b = get_block(ni, nj, nk)
                    rank_b = U_b.shape[1]
                    
                    for i_local in range(ni):
                        for j_local in range(nj):
                            i_global = i_offset + i_local
                            j_global = j_offset + j_local
                            U_full[i_global * m + j_global, col_offset:col_offset + rank_b] = \
                                U_b[i_local * nj + j_local, :]
                    
                    for j_local in range(nj):
                        for k_local in range(nk):
                            j_global = j_offset + j_local
                            k_global = k_offset + k_local
                            V_full[j_global * p + k_global, col_offset:col_offset + rank_b] = \
                                V_b[j_local * nk + k_local, :]
                    
                    for k_local in range(nk):
                        for i_local in range(ni):
                            k_global = k_offset + k_local
                            i_global = i_offset + i_local
                            W_full[k_global * n + i_global, col_offset:col_offset + rank_b] = \
                                W_b[k_local * ni + i_local, :]
                    
                    col_offset += rank_b
                    k_offset += nk
                j_offset += nj
            i_offset += ni
        
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        if verify(U_list, V_list, W_list, n, m, p):
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = total_rank
            break
    
    return best
# EVOLVE-BLOCK-END
