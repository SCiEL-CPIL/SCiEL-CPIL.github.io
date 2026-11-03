"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 by 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the flattened 2x2 matrices.
    U is indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i.
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: shape (4, 7), indexed by (i,j) -> i*2+j
    # V: shape (4, 7), indexed by (j,k) -> j*2+k
    # W: shape (4, 7), indexed by (k,i) -> k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11 is at U[0], A22 is at U[3]
    # B11 is at V[0], B22 is at V[3]
    # C11 at W[0], C22 at W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21 + A22) * B11
    # A21 at U[2], A22 at U[3]
    # B11 at V[0]
    # C21 at W[2], C22 at W[3]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11 * (B12 - B22)
    # A11 at U[0]
    # B12 at V[1], B22 at V[3]
    # C12 at W[1], C22 at W[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22 * (B21 - B11)
    # A22 at U[3]
    # B21 at V[2], B11 at V[0]
    # C11 at W[0], C21 at W[2]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11 + A12) * B22
    # A11 at U[0], A12 at U[1]
    # B22 at V[3]
    # C11 at W[0], C12 at W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    # A21 at U[2], A11 at U[0]
    # B11 at V[0], B12 at V[1]
    # C22 at W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    # A12 at U[1], A22 at U[3]
    # B21 at V[2], B22 at V[3]
    # C11 at W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _construct_block_decomposition(n, m, p):
    """
    Construct a block decomposition using Strassen for 2x2x2 blocks where possible.
    For 3x5x6: 
    - Partition the 3 rows of A into blocks of sizes [2, 1]
    - Partition the 5 columns of A / rows of B into blocks of sizes [2, 2, 1]  
    - Partition the 6 columns of B / columns of C into blocks of sizes [2, 2, 2]
    
    This gives block structure:
    A: 3x5 -> blocks [2,1] x [2,2,1]
    B: 5x6 -> blocks [2,2,1] x [2,2,2]
    C: 3x6 -> blocks [2,1] x [2,2,2]
    
    For each pair of row-block (from A/C) and column-block (from B/C), and 
    for each shared middle block index, we need a decomposition of the 
    (row_block_size x mid_block_size) by (mid_block_size x col_block_size) product.
    
    If row_block_size = 2, mid_block_size = 2, col_block_size = 2: use Strassen (rank 7)
    Otherwise: use schoolbook (rank = r*m*c)
    """
    n, m, p = 3, 5, 6
    
    # Block partitions
    # Rows of A (and C): partition 3 into [2, 1]
    row_blocks = [2, 1]
    # Columns of A / Rows of B: partition 5 into [2, 2, 1]
    mid_blocks = [2, 2, 1]
    # Columns of B (and C): partition 6 into [2, 2, 2]
    col_blocks = [2, 2, 2]
    
    # Compute cumulative offsets
    row_offsets = [0]
    for s in row_blocks:
        row_offsets.append(row_offsets[-1] + s)
    mid_offsets = [0]
    for s in mid_blocks:
        mid_offsets.append(mid_offsets[-1] + s)
    col_offsets = [0]
    for s in col_blocks:
        col_offsets.append(col_offsets[-1] + s)
    
    # Build the decomposition
    U = [[0] * (n * m) for _ in range(n * m)]  # Will be resized
    V = [[0] * (n * m) for _ in range(m * p)]
    W = [[0] * (n * m) for _ in range(p * n)]
    
    # First, let's compute the total rank
    total_rank = 0
    block_decomps = []
    
    for ri, r_size in enumerate(row_blocks):
        for mi, m_size in enumerate(mid_blocks):
            for ci, c_size in enumerate(col_blocks):
                if r_size == 2 and m_size == 2 and c_size == 2:
                    rank = 7
                else:
                    rank = r_size * m_size * c_size
                total_rank += rank
                block_decomps.append((ri, mi, ci, r_size, m_size, c_size, rank))
    
    # Now build U, V, W with the correct dimensions
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (ri, mi, ci, r_size, m_size, c_size, rank) in block_decomps:
        if r_size == 2 and m_size == 2 and c_size == 2:
            # Use Strassen
            sU, sV, sW = _strassen_2x2x2()
            for ui in range(4):
                for ci_idx in range(7):
                    # U index: global row = row_offsets[ri] + (ui // 2), global col = mid_offsets[mi] + (ui % 2)
                    g_row = row_offsets[ri] + ui // 2
                    g_col = mid_offsets[mi] + ui % 2
                    U[g_row * m + g_col][col + ci_idx] += sU[ui][ci_idx]
            for vi in range(4):
                for ci_idx in range(7):
                    # V index: global row = mid_offsets[mi] + (vi // 2), global col = col_offsets[ci] + (vi % 2)
                    g_row = mid_offsets[mi] + vi // 2
                    g_col = col_offsets[ci] + vi % 2
                    V[g_row * p + g_col][col + ci_idx] += sV[vi][ci_idx]
            for wi in range(4):
                for ci_idx in range(7):
                    # W index: k*n+i where k is col offset, i is row offset
                    # wi // 2 = k_local, wi % 2 = i_local
                    g_k = col_offsets[ci] + wi // 2
                    g_i = row_offsets[ri] + wi % 2
                    W[g_k * n + g_i][col + ci_idx] += sW[wi][ci_idx]
        else:
            # Schoolbook for this block
            for i_local in range(r_size):
                for j_local in range(m_size):
                    for k_local in range(c_size):
                        g_i = row_offsets[ri] + i_local
                        g_j = mid_offsets[mi] + j_local
                        g_k = col_offsets[ci] + k_local
                        U[g_i * m + g_j][col] = 1
                        V[g_j * p + g_k][col] = 1
                        W[g_k * n + g_i][col] = 1
                        col += 1
        col += rank - (r_size * m_size * c_size if not (r_size == 2 and m_size == 2 and c_size == 2) else 7)
        # Actually, let me redo this more carefully
    
    # Let me redo the column tracking properly
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (ri, mi, ci, r_size, m_size, c_size, rank) in block_decomps:
        if r_size == 2 and m_size == 2 and c_size == 2:
            sU, sV, sW = _strassen_2x2x2()
            for ui in range(4):
                for ci_idx in range(7):
                    g_row = row_offsets[ri] + ui // 2
                    g_col = mid_offsets[mi] + ui % 2
                    U[g_row * m + g_col][col + ci_idx] += sU[ui][ci_idx]
            for vi in range(4):
                for ci_idx in range(7):
                    g_row = mid_offsets[mi] + vi // 2
                    g_col = col_offsets[ci] + vi % 2
                    V[g_row * p + g_col][col + ci_idx] += sV[vi][ci_idx]
            for wi in range(4):
                for ci_idx in range(7):
                    g_k = col_offsets[ci] + wi // 2
                    g_i = row_offsets[ri] + wi % 2
                    W[g_k * n + g_i][col + ci_idx] += sW[wi][ci_idx]
            col += 7
        else:
            for i_local in range(r_size):
                for j_local in range(m_size):
                    for k_local in range(c_size):
                        g_i = row_offsets[ri] + i_local
                        g_j = mid_offsets[mi] + j_local
                        g_k = col_offsets[ci] + k_local
                        U[g_i * m + g_j][col] = 1
                        V[g_j * p + g_k][col] = 1
                        W[g_k * n + g_i][col] = 1
                        col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}, total_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Start with schoolbook
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try block decomposition with Strassen
    try:
        if time.time() < deadline - 5:
            block_result, block_rank = _construct_block_decomposition(n, m, p)
            if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
                if block_rank < best_rank:
                    best_rank = block_rank
                    best_result = block_result
    except Exception:
        pass
    
    # Try alternative block partitions
    # Partition 3 into [1, 2], 5 into [1, 2, 2], 6 into [2, 2, 2]
    try:
        if time.time() < deadline - 5:
            # Manually construct with different partition
            row_blocks = [1, 2]
            mid_blocks = [1, 2, 2]
            col_blocks = [2, 2, 2]
            
            row_offsets = [0]
            for s in row_blocks:
                row_offsets.append(row_offsets[-1] + s)
            mid_offsets = [0]
            for s in mid_blocks:
                mid_offsets.append(mid_offsets[-1] + s)
            col_offsets = [0]
            for s in col_blocks:
                col_offsets.append(col_offsets[-1] + s)
            
            total_rank = 0
            block_decomps = []
            for ri, r_size in enumerate(row_blocks):
                for mi, m_size in enumerate(mid_blocks):
                    for ci, c_size in enumerate(col_blocks):
                        if r_size == 2 and m_size == 2 and c_size == 2:
                            rank = 7
                        else:
                            rank = r_size * m_size * c_size
                        total_rank += rank
                        block_decomps.append((ri, mi, ci, r_size, m_size, c_size, rank))
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            for (ri, mi, ci, r_size, m_size, c_size, rank) in block_decomps:
                if r_size == 2 and m_size == 2 and c_size == 2:
                    sU, sV, sW = _strassen_2x2x2()
                    for ui in range(4):
                        for ci_idx in range(7):
                            g_row = row_offsets[ri] + ui // 2
                            g_col = mid_offsets[mi] + ui % 2
                            U[g_row * m + g_col][col + ci_idx] += sU[ui][ci_idx]
                    for vi in range(4):
                        for ci_idx in range(7):
                            g_row = mid_offsets[mi] + vi // 2
                            g_col = col_offsets[ci] + vi % 2
                            V[g_row * p + g_col][col + ci_idx] += sV[vi][ci_idx]
                    for wi in range(4):
                        for ci_idx in range(7):
                            g_k = col_offsets[ci] + wi // 2
                            g_i = row_offsets[ri] + wi % 2
                            W[g_k * n + g_i][col + ci_idx] += sW[wi][ci_idx]
                    col += 7
                else:
                    for i_local in range(r_size):
                        for j_local in range(m_size):
                            for k_local in range(c_size):
                                g_i = row_offsets[ri] + i_local
                                g_j = mid_offsets[mi] + j_local
                                g_k = col_offsets[ci] + k_local
                                U[g_i * m + g_j][col] = 1
                                V[g_j * p + g_k][col] = 1
                                W[g_k * n + g_i][col] = 1
                                col += 1
            
            assert col == total_rank
            result = {"U": U, "V": V, "W": W}
            if _verify(U, V, W, n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = result
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
