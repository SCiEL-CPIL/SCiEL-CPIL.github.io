import time
import numpy as np
from itertools import product
from fractions import Fraction

n, m, p = 3, 5, 5
_START_TIME = time.time()
DEADLINE = 140.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _validate_numpy(U, V, W, n, m, p):
    """Fast validation using numpy."""
    if not U or not V or not W:
        return False
    try:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
    except Exception:
        return False
    if U_np.ndim != 2 or V_np.ndim != 2 or W_np.ndim != 2:
        return False
    rank = U_np.shape[1]
    if rank == 0:
        return False
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != V_np.shape[1] or V_np.shape[1] != W_np.shape[1]:
        return False
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return False
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns the 7-term Strassen decomposition for 2x2 * 2x2.
    Shapes: U (4,7), V (4,7), W (4,7)
    """
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
    # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
    # W rows: C11, C12, C21, C22 (indices 0,1,2,3)
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 1, 0, 0, 1, 0, -1],  # A12
        [0, 1, 0, 1, 0, 1, 0],   # A21
        [1, 0, 0, 1, 0, 0, -1]   # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, -1, 1, 0, 1, 0],  # B12
        [1, 0, 1, 0, 0, 0, 1],   # B21
        [1, 0, 0, 0, 1, 0, 1]    # B22
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # C11
        [0, 0, 0, 0, 1, 0, 0],   # C12
        [0, 1, 1, 0, 0, 0, 0],   # C21
        [0, 0, 0, 1, 0, 0, 0]    # C22
    ]
    return U, V, W

def _block_decomposition(n_parts, m_parts, p_parts, strassen_blocks=False):
    """
    Constructs a block decomposition.
    n_parts: list of row block sizes for A
    m_parts: list of col block sizes for A / row block sizes for B
    p_parts: list of col block sizes for B / row block sizes for C
    
    If strassen_blocks is True, uses Strassen for 2x2x2 blocks where possible.
    Otherwise, uses schoolbook.
    """
    U_blocks = []
    V_blocks = []
    W_blocks = []
    total_rank = 0
    
    # Precompute Strassen matrices
    if strassen_blocks:
        S_U, S_V, S_W = _strassen_2x2x2()
    
    for i_idx, ni in enumerate(n_parts):
        for j_idx, mj in enumerate(m_parts):
            for k_idx, pk in enumerate(p_parts):
                # Determine if we can use Strassen
                if strassen_blocks and ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    rank_block = 7
                    U_block = S_U
                    V_block = S_V
                    W_block = S_W
                else:
                    # Schoolbook
                    rank_block = ni * mj * pk
                    U_block = np.eye(ni * mj, rank_block)
                    V_block = np.eye(mj * pk, rank_block)
                    W_block = np.eye(pk * ni, rank_block)
                
                # We need to expand these to the full block sizes if they were smaller?
                # No, the blocks are exactly ni, mj, pk.
                # But Strassen is specifically for 2x2x2.
                
                U_blocks.append((i_idx, j_idx, U_block, rank_block))
                V_blocks.append((j_idx, k_idx, V_block, rank_block))
                W_blocks.append((k_idx, i_idx, W_block, rank_block))
                total_rank += rank_block
                
    # Construct final matrices
    U = np.zeros((sum(n_parts) * sum(m_parts), total_rank))
    V = np.zeros((sum(m_parts) * sum(p_parts), total_rank))
    W = np.zeros((sum(p_parts) * sum(n_parts), total_rank))
    
    # This construction is tricky because blocks overlap in indices.
    # It's easier to build by iterating over the global indices.
    
    # Let's restart the construction logic more carefully.
    
    U_final = []
    V_final = []
    W_final = []
    
    # We will build column by column (rank term by rank term).
    # For each block (i,j,k), we have a set of rank terms.
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_idx, ni in enumerate(n_parts):
        for j_idx, mj in enumerate(m_parts):
            U_row = [0] * total_rank
            for k_idx, pk in enumerate(p_parts):
                # Find the offset for this block
                # This is getting complicated with offsets.
                pass
            
    # Alternative: Direct construction
    # Let's use a simpler approach:
    # 1. Determine total rank.
    # 2. For each (i,j,k) block, assign a range of rank indices [start, end).
    # 3. Fill in U, V, W for that range using the appropriate pattern (Strassen or Schoolbook).
    
    rank_ranges = {}
    current_rank = 0
    for i_idx, ni in enumerate(n_parts):
        for j_idx, mj in enumerate(m_parts):
            for k_idx, pk in enumerate(p_parts):
                if strassen_blocks and ni == 2 and mj == 2 and pk == 2:
                    r = 7
                else:
                    r = ni * mj * pk
                rank_ranges[(i_idx, j_idx, k_idx)] = (current_rank, current_rank + r)
                current_rank += r
    
    total_rank = current_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Compute global offsets
    n_offsets = [0]
    for i in range(len(n_parts)-1):
        n_offsets.append(n_offsets[-1] + n_parts[i])
    m_offsets = [0]
    for j in range(len(m_parts)-1):
        m_offsets.append(m_offsets[-1] + m_parts[j])
    p_offsets = [0]
    for k in range(len(p_parts)-1):
        p_offsets.append(p_offsets[-1] + p_parts[k])
        
    for (i_idx, j_idx, k_idx), (start_r, end_r) in rank_ranges.items():
        ni = n_parts[i_idx]
        mj = m_parts[j_idx]
        pk = p_parts[k_idx]
        
        n_start = n_offsets[i_idx]
        m_start = m_offsets[j_idx]
        p_start = p_offsets[k_idx]
        
        if strassen_blocks and ni == 2 and mj == 2 and pk == 2:
            S_U, S_V, S_W = _strassen_2x2x2()
            for r in range(7):
                # Fill U
                for i_loc in range(2):
                    for j_loc in range(2):
                        # Index in U: (n_start + i_loc)*m + (m_start + j_loc)
                        idx_u = (n_start + i_loc) * m + (m_start + j_loc)
                        U[idx_u][start_r + r] = S_U[i_loc*2+j_loc][r]
                # Fill V
                for j_loc in range(2):
                    for k_loc in range(2):
                        idx_v = (m_start + j_loc) * p + (p_start + k_loc)
                        V[idx_v][start_r + r] = S_V[j_loc*2+k_loc][r]
                # Fill W
                for k_loc in range(2):
                    for i_loc in range(2):
                        idx_w = (p_start + k_loc) * n + (n_start + i_loc)
                        W[idx_w][start_r + r] = S_W[k_loc*2+i_loc][r]
        else:
            # Schoolbook
            r_offset = 0
            for i_loc in range(ni):
                for j_loc in range(mj):
                    for k_loc in range(pk):
                        idx_u = (n_start + i_loc) * m + (m_start + j_loc)
                        idx_v = (m_start + j_loc) * p + (p_start + k_loc)
                        idx_w = (p_start + k_loc) * n + (n_start + i_loc)
                        
                        r_idx = start_r + r_offset
                        U[idx_u][r_idx] = 1
                        V[idx_v][r_idx] = 1
                        W[idx_w][r_idx] = 1
                        r_offset += 1
                        
    return {"U": U, "V": V, "W": W}

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # 2. Try block decompositions with Strassen
    
    # Partitions of n=3: [3], [1,2], [2,1], [1,1,1]
    # Partitions of m=5: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,3,1], [3,1,1], [2,2,1], [2,1,2], [1,2,2], [1,1,1,1,1]...
    # Partitions of p=5: same as m
    
    n_parts_list = [
        [3],
        [1, 2],
        [2, 1],
        [1, 1, 1]
    ]
    
    m_parts_list = [
        [5],
        [1, 4],
        [2, 3],
        [3, 2],
        [4, 1],
        [1, 1, 3],
        [1, 3, 1],
        [3, 1, 1],
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2]
    ]
    
    p_parts_list = m_parts_list # Same possibilities
    
    # We want to maximize the number of 2x2x2 blocks.
    # So we prefer partitions with many 2s.
    # Let's sort by number of 2s descending, then total rank ascending?
    
    def score_partition(parts):
        return parts.count(2)
        
    m_parts_list.sort(key=score_partition, reverse=True)
    p_parts_list.sort(key=score_partition, reverse=True)
    
    # Try combinations
    for n_parts in n_parts_list:
        if _time_left() < 5:
            break
        for m_parts in m_parts_list:
            if _time_left() < 5:
                break
            for p_parts in p_parts_list:
                if _time_left() < 2:
                    break
                    
                # Quick check: if no 2x2x2 blocks, skip Strassen (schoolbook is same)
                # But we can still try to see if the structure allows for something better?
                # No, without Strassen, it's just schoolbook.
                # So we only care if there are potential 2x2x2 blocks.
                has_strassen_potential = False
                for ni in n_parts:
                    if ni != 2:
                        continue
                    for mj in m_parts:
                        if mj != 2:
                            continue
                        for pk in p_parts:
                            if pk == 2:
                                has_strassen_potential = True
                                break
                        if has_strassen_potential:
                            break
                    if has_strassen_potential:
                        break
                            
                if not has_strassen_potential:
                    continue
                    
                try:
                    decomp = _block_decomposition(n_parts, m_parts, p_parts, strassen_blocks=True)
                    rank = len(decomp["U"][0])
                    if rank < best_rank:
                        if _validate_numpy(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                            best = decomp
                            best_rank = rank
                except Exception:
                    pass
                    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank achieved: {len(result['U'][0]) if result['U'] else 'unknown'}")
