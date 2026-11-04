"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        C_dec = np.zeros((n, p))
        for c in range(rank):
            for i in range(n):
                uA = 0.0
                for j in range(m):
                    uA += U[i * m + j][c] * A[i, j]
                for k in range(p):
                    vB = 0.0
                    for j2 in range(m):
                        vB += V[j2 * p + k][c] * B[j2, k]
                    C_dec[i, k] += W[k * n + i][c] * uA * vB
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def _to_list(mat):
    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else (round(v * 2) / 2 if abs(v * 2 - round(v * 2)) < 1e-9 else v) for v in row] for row in mat]

def _clean_matrix(mat):
    result = []
    for row in mat:
        new_row = []
        for v in row:
            if abs(v - round(v)) < 1e-9:
                new_row.append(int(round(v)))
            elif abs(v * 2 - round(v * 2)) < 1e-9:
                new_row.append(round(v * 2) / 2)
            else:
                new_row.append(v)
        result.append(new_row)
    return result

def _strassen_2x2x2():
    """Return (U, V, W) lists for rank-7 Strassen 2x2x2."""
    # Standard Strassen decomposition
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indexed by (i,j) where i=0,1 (row of A), j=0,1 (col of A)
    # V indexed by (j,k) where j=0,1 (row of B), k=0,1 (col of B)
    # W indexed by (k,i) where k=0,1 (col of C), i=0,1 (row of C)
    
    U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3: A00*(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, B11
    W[1][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4: A11*(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1; V[2][3] = -1  # B01, B10
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6: (A10-A00)*(B00+B10)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[2][5] = 1  # B00, B10
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)*(B01+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[1][6] = 1; V[3][6] = 1  # B01, B11
    W[0][6] = 1  # C00
    
    return U, V, W

def _schoolbook_2x2x2():
    """Return (U, V, W) lists for rank-8 schoolbook 2x2x2."""
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(block_decomps, block_offsets):
    """
    Combine multiple block decompositions into one global decomposition.
    
    block_decomps: list of (U_block, V_block, W_block) for each block
    block_offsets: list of (u_offset, v_offset, w_offset) for each block
                   u_offset: (i_start, j_start) in A
                   v_offset: (j_start, k_start) in B  
                   w_offset: (k_start, i_start) in C
    
    Returns (U, V, W) global lists.
    """
    total_rank = sum(len(U_b[0]) for U_b, _, _ in block_decomps)
    # We need to know dimensions
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Determine global sizes from the block structure
    # For 3x4x8: n=3, m=4, p=8
    n, m, p = 3, 4, 8
    
    U_global = [[0]*total_rank for _ in range(n*m)]
    V_global = [[0]*total_rank for _ in range(m*p)]
    W_global = [[0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for (U_b, V_b, W_b), (u_off, v_off, w_off) in zip(block_decomps, block_offsets):
        r_b = len(U_b[0])
        # U_b has shape (n_b * m_b, r_b)
        # V_b has shape (m_b * p_b, r_b)
        # W_b has shape (p_b * n_b, r_b)
        
        n_b = len(W_b) // (len(W_b[0]) // r_b) if r_b > 0 else 0
        # Actually, let's just use the offsets directly
        # u_off = (i_start, j_start), meaning U_b[i_b, j_b] maps to U_global[(i_start+i_b)*m + (j_start+j_b), ...]
        # v_off = (j_start, k_start), meaning V_b[j_b, k_b] maps to V_global[(j_start+j_b)*p + (k_start+k_b), ...]
        # w_off = (k_start, i_start), meaning W_b[k_b, i_b] maps to W_global[(k_start+k_b)*n + (i_start+i_b), ...]
        
        i_start, j_start = u_off
        j2_start, k_start = v_off
        k2_start, i2_start = w_off
        
        # For U: U_b row index = i_b * m_b + j_b where m_b is the block width in j
        # We need to infer m_b, p_b, n_b from the block sizes
        # Let's pass block dimensions explicitly
        # Actually, let me restructure: pass block dims too
        
        for i_b in range(len(U_b)):
            # i_b encodes (i_local, j_local) in the block
            # We need to know the block dimensions to decode
            pass
        
        # This is getting complicated. Let me use a different approach.
        break
    
    # Let me redo this more carefully
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Build a structured decomposition using blocks
    # Strategy: Partition A (3x4) and B (4x8) into blocks
    # A: rows 0-1 (2x4) and row 2 (1x4)
    # B: cols 0-3 (4x4) and cols 4-7 (4x4)
    
    # For the 2x4 x 4x4 subproblem (rows 0-1 of A, all of B's first 4 cols):
    # We can use 2x2x2 Strassen blocks: 4 blocks of 2x2x2
    # Each 2x2x2 block: A[i:i+2, j:j+2] * B[j:j+2, k:k+2] -> C[i:i+2, k:k+2]
    # With Strassen, each block has rank 7, total 4*7 = 28 for this subproblem
    
    # For the 1x4 x 4x4 subproblem (row 2 of A, first 4 cols of B):
    # This is 1x4x4, which can be done in rank 4 (schoolbook) or potentially less
    # Actually 1x4x4: C[0,k] = sum_j A[0,j]*B[j,k], rank = 4 (one for each j)
    # But we can't do better than 4 since each output depends on all 4 inputs
    
    # Let me build this step by step
    
    # Block decomposition:
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2], Strassen rank 7
    # Block (0,1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2], Strassen rank 7
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4], Strassen rank 7
    # Block (0,3): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4], Strassen rank 7
    # Block (1,0): A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2], schoolbook rank 4
    # Block (1,1): A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2], schoolbook rank 4
    # Block (1,2): A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4], schoolbook rank 4
    # Block (1,3): A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4], schoolbook rank 4
    # Blocks for cols 4-7:
    # Block (0,4): A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6], Strassen rank 7
    # Block (0,5): A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6], Strassen rank 7
    # Block (0,6): A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8], Strassen rank 7
    # Block (0,7): A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8], Strassen rank 7
    # Block (1,4): A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6], schoolbook rank 4
    # Block (1,5): A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6], schoolbook rank 4
    # Block (1,6): A[2:3, 0:2] * B[0:2, 6:8] -> C[2:3, 6:8], schoolbook rank 4
    # Block (1,7): A[2:3, 2:4] * B[2:4, 6:8] -> C[2:3, 6:8], schoolbook rank 4
    
    # Total rank: 8*7 + 8*4 = 56 + 32 = 88
    
    # Let me build this
    U_total = [[0]*88 for _ in range(nm)]
    V_total = [[0]*88 for _ in range(mp)]
    W_total = [[0]*88 for _ in range(pn)]
    
    rank_offset = 0
    
    # Helper to add a 2x2x2 block
    def add_2x2x2_block(i_off, j_off, k_off, use_strassen=True):
        nonlocal rank_offset
        if use_strassen:
            U_b, V_b, W_b = _strassen_2x2x2()
        else:
            U_b, V_b, W_b = _schoolbook_2x2x2()
        r_b = len(U_b[0])
        
        for i_b in range(2):
            for j_b in range(2):
                u_idx = (i_off + i_b) * m + (j_off + j_b)
                for c in range(r_b):
                    U_total[u_idx][rank_offset + c] = U_b[i_b * 2 + j_b][c]
        
        for j_b in range(2):
            for k_b in range(2):
                v_idx = (j_off + j_b) * p + (k_off + k_b)
                for c in range(r_b):
                    V_total[v_idx][rank_offset + c] = V_b[j_b * 2 + k_b][c]
        
        for k_b in range(2):
            for i_b in range(2):
                w_idx = (k_off + k_b) * n + (i_off + i_b)
                for c in range(r_b):
                    W_total[w_idx][rank_offset + c] = W_b[k_b * 2 + i_b][c]
        
        rank_offset += r_b
    
    # Helper to add a 1x2x2 block (schoolbook)
    def add_1x2x2_block(i_off, j_off, k_off):
        nonlocal rank_offset
        # C[i_off, k] = sum_j A[i_off, j] * B[j, k] for k in [k_off, k_off+2)
        # Rank 2: one term per j
        r_b = 2
        for j_b in range(2):
            for c in range(r_b):
                if c == j_b:
                    u_idx = i_off * m + (j_off + j_b)
                    U_total[u_idx][rank_offset + c] = 1
                    v_idx = (j_off + j_b) * p + k_off + c
                    V_total[v_idx][rank_offset + c] = 1
                    w_idx = (k_off + c) * n + i_off
                    W_total[w_idx][rank_offset + c] = 1
        rank_offset += r_b
    
    # Add all blocks
    # Rows 0-1 of A, all 4 cols of B, cols 0-7 of C
    # j ranges over 0-3 (4 cols of A), k ranges over 0-7 (8 cols of B)
    # For rows 0-1: use Strassen for 2x2x2 blocks
    for j_off in [0, 2]:  # j blocks of size 2
        for k_off in [0, 2, 4, 6]:  # k blocks of size 2
            add_2x2x2_block(0, j_off, k_off, use_strassen=True)
    
    # Row 2 of A
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_1x2x2_block(2, j_off, k_off)
    
    # Verify
    U_list = _clean_matrix(U_total)
    V_list = _clean_matrix(V_total)
    W_list = _clean_matrix(W_total)
    
    if _verify(U_list, V_list, W_list, n, m, p):
        best = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = 88
    
    # Try to improve: use a better algorithm for 1x4x4 (if exists)
    # Or try mixing: use Strassen for some, schoolbook for others
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    # Try random search for lower rank
    rng = np.random.default_rng(42)
    
    for r in range(24, min(best_rank, 89)):
        if time.time() > deadline - 5:
            break
        
        max_iters = 3
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_rand = rng.standard_normal((nm, r))
            V_rand = rng.standard_normal((mp, r))
            W_rand = rng.standard_normal((pn, r))
            
            U_l = _clean_matrix(U_rand)
            V_l = _clean_matrix(V_rand)
            W_l = _clean_matrix(W_rand)
            
            if _verify(U_l, V_l, W_l, n, m, p):
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    U_final = _clean_matrix(best["U"])
    V_final = _clean_matrix(best["V"])
    W_final = _clean_matrix(best["W"])
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
