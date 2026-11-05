import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2_block(i0, j0, k0, U, V, W, offset):
    """Add Strassen 7-multiplication for 2x2 * 2x2 subblock.
    A[2,2] * B[2,2] = C[2,2] using 7 multiplications."""
    # Strassen formulas for C = A*B:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for mult_idx in range(7):
        col = offset + mult_idx
        if mult_idx == 0:
            # M1 = (A00+A11)*(B00+B11)
            U[(i0+0)*m + (j0+0)][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+0)*p + (k0+0)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[(k0+0)*n + (i0+0)][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1
        elif mult_idx == 1:
            # M2 = (A10+A11)*B00
            U[(i0+1)*m + (j0+0)][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+0)*p + (k0+0)][col] = 1
            W[(k0+0)*n + (i0+1)][col] = 1
        elif mult_idx == 2:
            # M3 = A00*(B01-B11)
            U[(i0+0)*m + (j0+0)][col] = 1
            V[(j0+0)*p + (k0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = -1
            W[(k0+1)*n + (i0+0)][col] = 1
        elif mult_idx == 3:
            # M4 = A11*(B10-B00)
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+1)*p + (k0+0)][col] = 1
            V[(j0+0)*p + (k0+0)][col] = -1
            W[(k0+0)*n + (i0+1)][col] = 1
        elif mult_idx == 4:
            # M5 = (A00+A10)*B11
            U[(i0+0)*m + (j0+0)][col] = 1
            U[(i0+1)*m + (j0+0)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[(k0+1)*n + (i0+0)][col] = 1
        elif mult_idx == 5:
            # M6 = (A11-A00)*(B00+B10)
            U[(i0+1)*m + (j0+1)][col] = 1
            U[(i0+0)*m + (j0+0)][col] = -1
            V[(j0+0)*p + (k0+0)][col] = 1
            V[(j0+1)*p + (k0+0)][col] = 1
            W[(k0+0)*n + (i0+1)][col] = 1
        elif mult_idx == 6:
            # M7 = (A01-A11)*(B01+B11)
            U[(i0+0)*m + (j0+1)][col] = 1
            U[(i0+1)*m + (j0+1)][col] = -1
            V[(j0+0)*p + (k0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[(k0+1)*n + (i0+0)][col] = 1

def _add_2x1_block(i0, j0, k0, U, V, W, offset):
    """Add 2x1 * 1x5 block using 10 multiplications (schoolbook for this part)."""
    for di in range(2):
        for dk in range(5):
            col = offset + di*5 + dk
            U[(i0+di)*m + j0][col] = 1
            V[j0*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + (i0+di)][col] = 1

def _add_1x2_block(i0, j0, k0, U, V, W, offset):
    """Add 1x2 * 2x5 block using 10 multiplications (schoolbook for this part)."""
    for dj in range(2):
        for dk in range(5):
            col = offset + dj*5 + dk
            U[i0*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + i0][col] = 1

def _add_1x1_block(i0, j0, k0, U, V, W, offset):
    """Add 1x1 * 1x5 block using 5 multiplications (schoolbook for this part)."""
    for dk in range(5):
        col = offset + dk
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+dk)][col] = 1
        W[(k0+dk)*n + i0][col] = 1

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Build Strassen-based decomposition with 2+2+1 partitioning
    # A is 2x5, B is 5x5, C is 2x5
    # Partition A into rows: [2, 2, 1] -> but A only has 2 rows, so [2, 0, 0] or just handle differently
    # Actually for 2x5 * 5x5:
    # A has 2 rows, B has 5x5
    # Let's partition the column/row dimensions:
    # A: 2 rows, 5 cols -> partition cols into [2, 2, 1]
    # B: 5 rows, 5 cols -> partition rows into [2, 2, 1], cols into [2, 2, 1]
    # C: 2 rows, 5 cols -> partition cols into [2, 2, 1]
    
    # So we have 3x3 = 9 blocks:
    # Block (ri, cj) where ri in {0,1,2} for A row block (but A has only 2 rows, so ri=0 is rows 0-1, ri=1 is empty, ri=2 is empty)
    # Wait, A has 2 rows. So we can only partition A's rows into one block of size 2.
    # Let me reconsider: partition B's rows (5) into [2,2,1] and B's cols (5) into [2,2,1]
    # A's cols (5) into [2,2,1]
    # C's rows (2) is just one block, C's cols (5) into [2,2,1]
    
    # So the blocks are indexed by (a_col_block, b_row_block, b_col_block)
    # But since A has only 2 rows, the "row" dimension of A is fixed.
    
    # Let me think of it as: C[i, k] = sum_j A[i,j] * B[j,k]
    # Partition j into [2,2,1] and k into [2,2,1]
    # For each pair (j_block, k_block), we have a sub-computation:
    # C_sub[i, k_sub] = sum_{j in j_block} A[i, j_sub] * B[j_sub, k_sub]
    
    # Since A has 2 rows and 5 cols, and we partition cols into [2,2,1]:
    # A_block_0: 2x2 (rows 0-1, cols 0-1)
    # A_block_1: 2x2 (rows 0-1, cols 2-3)
    # A_block_2: 2x1 (rows 0-1, col 4)
    
    # B_block_(j_blk, k_blk):
    # B_00: 2x2 (rows 0-1, cols 0-1)
    # B_01: 2x2 (rows 0-1, cols 2-3)
    # B_02: 2x1 (rows 0-1, col 4)
    # B_10: 2x2 (rows 2-3, cols 0-1)
    # B_11: 2x2 (rows 2-3, cols 2-3)
    # B_12: 2x1 (rows 2-3, col 4)
    # B_20: 1x2 (row 4, cols 0-1)
    # B_21: 1x2 (row 4, cols 2-3)
    # B_22: 1x1 (row 4, col 4)
    
    # C_block_(k_blk):
    # C_0: 2x2 (rows 0-1, cols 0-1)
    # C_1: 2x2 (rows 0-1, cols 2-3)
    # C_2: 2x1 (rows 0-1, col 4)
    
    # Total blocks: 3 (j_blocks) * 3 (k_blocks) = 9 sub-problems
    # For each (j_blk, k_blk): A_sub (2 x |j_blk|) * B_sub (|j_blk| x |k_blk|) -> C_sub (2 x |k_blk|)
    
    # Let's compute the rank for each sub-problem:
    # (j_blk=0, k_blk=0): 2x2 * 2x2 -> 2x2, Strassen: 7
    # (j_blk=0, k_blk=1): 2x2 * 2x2 -> 2x2, Strassen: 7
    # (j_blk=0, k_blk=2): 2x2 * 2x1 -> 2x1, schoolbook: 4
    # (j_blk=1, k_blk=0): 2x2 * 2x2 -> 2x2, Strassen: 7
    # (j_blk=1, k_blk=1): 2x2 * 2x2 -> 2x2, Strassen: 7
    # (j_blk=1, k_blk=2): 2x2 * 2x1 -> 2x1, schoolbook: 4
    # (j_blk=2, k_blk=0): 2x1 * 1x2 -> 2x2, schoolbook: 4
    # (j_blk=2, k_blk=1): 2x1 * 1x2 -> 2x2, schoolbook: 4
    # (j_blk=2, k_blk=2): 2x1 * 1x1 -> 2x1, schoolbook: 2
    
    # Total: 7+7+4+7+7+4+4+4+2 = 46
    
    U = [[0] * 46 for _ in range(n * m)]
    V = [[0] * 46 for _ in range(m * p)]
    W = [[0] * 46 for _ in range(p * n)]
    
    offset = 0
    
    # Block (j_blk=0, k_blk=0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    _strassen_2x2_block(0, 0, 0, U, V, W, offset)
    offset += 7
    
    # Block (j_blk=0, k_blk=1): A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    _strassen_2x2_block(0, 2, 2, U, V, W, offset)
    offset += 7
    
    # Block (j_blk=0, k_blk=2): A[0:2, 4:5] * B[0:2, 4:5] -> C[0:2, 4:5]
    # 2x1 * 2x1 -> 2x1, schoolbook: 4
    for di in range(2):
        for dj in range(2):
            col = offset + di*2 + dj
            U[(0+di)*m + (4)][col] = 1
            V[(0+dj)*p + (4)][col] = 1
            W[(4)*n + (0+di)][col] = 1
    offset += 4
    
    # Block (j_blk=1, k_blk=0): A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    _strassen_2x2_block(0, 0, 0, U, V, W, offset)
    # But wait, this is wrong. The A block is the same (rows 0-1, cols 0-1 of A),
    # but the B block is rows 2-3, cols 0-1 of B.
    # I need to adjust the indices. Let me redo this more carefully.
    
    # Actually, I made an error. Let me restart the construction properly.
    
    U = [[0] * 46 for _ in range(n * m)]
    V = [[0] * 46 for _ in range(m * p)]
    W = [[0] * 46 for _ in range(p * n)]
    
    offset = 0
    
    # j_blocks: [0:2], [2:4], [4:5]
    # k_blocks: [0:2], [2:4], [4:5]
    
    for j_blk in range(3):
        j_start = [0, 2, 4][j_blk]
        j_end = [2, 4, 5][j_blk]
        j_size = j_end - j_start
        
        for k_blk in range(3):
            k_start = [0, 2, 4][k_blk]
            k_end = [2, 4, 5][k_blk]
            k_size = k_end - k_start
            
            # A_sub: rows 0-1, cols j_start to j_end (size 2 x j_size)
            # B_sub: rows j_start to j_end, cols k_start to k_end (size j_size x k_size)
            # C_sub: rows 0-1, cols k_start to k_end (size 2 x k_size)
            
            if j_size == 2 and k_size == 2:
                # Use Strassen: 7 multiplications
                # A_sub is 2x2, B_sub is 2x2
                _strassen_2x2_block(0, j_start, k_start, U, V, W, offset)
                offset += 7
            elif j_size == 2 and k_size == 1:
                # 2x2 * 2x1 -> 2x1, schoolbook: 4
                for di in range(2):
                    for dj in range(2):
                        col = offset + di*2 + dj
                        U[(0+di)*m + (j_start+dj)][col] = 1
                        V[(j_start+dj)*p + (k_start)][col] = 1
                        W[(k_start)*n + (0+di)][col] = 1
                offset += 4
            elif j_size == 1 and k_size == 2:
                # 2x1 * 1x2 -> 2x2, schoolbook: 4
                for di in range(2):
                    for dk in range(2):
                        col = offset + di*2 + dk
                        U[(0+di)*m + (j_start)][col] = 1
                        V[(j_start)*p + (k_start+dk)][col] = 1
                        W[(k_start+dk)*n + (0+di)][col] = 1
                offset += 4
            elif j_size == 1 and k_size == 1:
                # 2x1 * 1x1 -> 2x1, schoolbook: 2
                for di in range(2):
                    col = offset + di
                    U[(0+di)*m + (j_start)][col] = 1
                    V[(j_start)*p + (k_start)][col] = 1
                    W[(k_start)*n + (0+di)][col] = 1
                offset += 2
    
    # Verify the Strassen-based decomposition
    if _verify_exact(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = 46
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p
    
    # Try to find even lower rank using ALS if time permits
    if time.time() < deadline - 30.0:
        for target_rank in range(best_rank - 1, 40, -1):
            if time.time() > deadline - 5.0:
                break
            
            max_restarts = 10
            found = False
            for restart in range(max_restarts):
                if time.time() > deadline - 2.0:
                    break
                    
                np.random.seed(2000 + restart * target_rank)
                
                U_als = np.random.randn(n * m, target_rank) * 0.1
                V_als = np.random.randn(m * p, target_rank) * 0.1
                W_als = np.random.randn(p * n, target_rank) * 0.1
                
                error = float('inf')
                for it in range(30):
                    if time.time() > deadline - 1.0:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((p, target_rank))
                            for k in range(p):
                                A_mat[k, :] = V_als[j * p + k, :] * W_als[k * n + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                                U_als[i * m + j, :] = x
                            except np.linalg.LinAlgError:
                                pass

                    # Update V
                    for j in range(m):
                        for k in range(p):
                            B_mat = np.zeros((n, target_rank))
                            for i in range(n):
                                B_mat[i, :] = U_als[i * m + j, :] * W_als[k * n + i, :]
                            try:
                                y, _, _, _ = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                                V_als[j * p + k, :] = y
                            except np.linalg.LinAlgError:
                                pass

                    # Update W
                    for k in range(p):
                        for i in range(n):
                            C_mat = np.zeros((m, target_rank))
                            for j in range(m):
                                C_mat[j, :] = U_als[i * m + j, :] * V_als[j * p + k, :]
                            try:
                                z, _, _, _ = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)
                                W_als[k * n + i, :] = z
                            except np.linalg.LinAlgError:
                                pass
                    
                    # Check error
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_als[i * m + j, :], V_als[j * p + k, :] * W_als[k * n + i, :])
                                error += (s - 1.0)**2
                    
                    if error < 1e-12:
                        break
                
                if error < 1e-8:
                    # Try rounding
                    for denom in [1, 2, 4]:
                        U_r = np.round(U_als * denom) / denom
                        V_r = np.round(V_als * denom) / denom
                        W_r = np.round(W_als * denom) / denom
                        
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = Fraction(0)
                                    for c in range(target_rank):
                                        s += Fraction(U_r[i * m + j, c].item()) * Fraction(V_r[j * p + k, c].item()) * Fraction(W_r[k * n + i, c].item())
                                    if s != 1:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            best_U = U_r.tolist()
                            best_V = V_r.tolist()
                            best_W = W_r.tolist()
                            best_rank = target_rank
                            found = True
                            break
                    if found:
                        break
            
            if found:
                if time.time() > deadline - 10.0:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
