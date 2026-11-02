"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    
    # Target tensor: T[a*m+b, b*p+c, c*n+a] = 1 for all (a,b,c)
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    # T_computed[i,j,k] = sum_c U2[i,c] * V2[j,c] * W2[k,c]
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    
    # Since each factor is scaled by 2, product is scaled by 8
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True

def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Kronecker product of two 2x2 Strassen decompositions.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for macro_k in range(7):
        for micro_l in range(7):
            col = macro_k * 7 + micro_l
            # U: index = (macro_block_row * 2 + micro_row) * 4 + (macro_block_col * 2 + micro_col)
            # For A: global index = (block_row*2 + local_row)*4 + (block_col*2 + local_col)
            # U2[local_idx] where local_idx = local_row*2 + local_col
            # macro_k determines which blocks of A are involved
            # micro_l determines the local Strassen term
            
            # The 4x4 A matrix is partitioned into 2x2 blocks:
            # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
            # Global row index for A: (block_row*2 + local_row)*4 + (block_col*2 + local_col)
            
            # For macro term k, the blocks of A that contribute are determined by Strassen
            # M1: (A00+A11) -> blocks (0,0) and (1,1)
            # M2: (A10+A11) -> blocks (1,0) and (1,1)
            # M3: A00 -> block (0,0)
            # M4: A11 -> block (1,1)
            # M5: (A00+A01) -> blocks (0,0) and (0,1)
            # M6: (A10-A00) -> blocks (1,0) and (0,0)
            # M7: (A01-A11) -> blocks (0,1) and (1,1)
            
            a_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,1)],
                2: [(0,0,1)],
                3: [(1,1,1)],
                4: [(0,0,1), (0,1,1)],
                5: [(1,0,1), (0,0,-1)],
                6: [(0,1,1), (1,1,-1)],
            }[macro_k]
            
            b_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(0,0,1)],
                2: [(0,1,1), (1,1,-1)],
                3: [(1,0,1), (0,0,-1)],
                4: [(1,1,1)],
                5: [(0,0,1), (0,1,1)],
                6: [(1,0,1), (1,1,1)],
            }[macro_k]
            
            c_blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,-1)],
                2: [(0,1,1), (1,1,1)],
                3: [(0,0,1), (1,0,1)],
                4: [(0,0,-1), (0,1,1)],
                5: [(1,1,1)],
                6: [(0,0,1)],
            }[macro_k]
            
            for local_idx in range(4):
                u_coeff = U2[local_idx][micro_l]
                v_coeff = V2[local_idx][micro_l]
                w_coeff = W2[local_idx][micro_l]
                
                local_row, local_col = local_idx // 2, local_idx % 2
                
                for (blk_r, blk_c, coeff) in a_blocks:
                    if u_coeff != 0 and coeff != 0:
                        global_idx = (blk_r * 2 + local_row) * 4 + (blk_c * 2 + local_col)
                        U[global_idx][col] += u_coeff * coeff
                
                for (blk_r, blk_c, coeff) in b_blocks:
                    if v_coeff != 0 and coeff != 0:
                        global_idx = (blk_r * 2 + local_row) * 4 + (blk_c * 2 + local_col)
                        V[global_idx][col] += v_coeff * coeff
                
                for (blk_r, blk_c, coeff) in c_blocks:
                    if w_coeff != 0 and coeff != 0:
                        # C index: (blk_c * 2 + local_col) * 4 + (blk_r * 2 + local_row)
                        global_idx = (blk_c * 2 + local_col) * 4 + (blk_r * 2 + local_row)
                        W[global_idx][col] += w_coeff * coeff
    
    return U, V, W

def _build_5x5_decomposition():
    """
    Build a rank-110 decomposition for 5x5x5 matrix multiplication.
    Strategy: Split 5 = 4 + 1.
    - Use rank-49 Strassen for the 4x4 sub-block.
    - Handle cross terms between 4-part and 1-part naively.
    Total rank = 49 + 61 = 110.
    """
    U4, V4, W4 = _build_strassen_4x4()
    
    rank = 49 + 61
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    # Copy 4x4 Strassen decomposition into the first 16 rows/cols of each matrix
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
    
    col = 49
    # Cross terms: all (i,k,j) where not all of i,k,j are < 4
    for i in range(5):
        for k in range(5):
            for j in range(5):
                if i < 4 and k < 4 and j < 4:
                    continue
                # U index: row i of A, col k of A -> i*5 + k
                # V index: row k of B, col j of B -> k*5 + j
                # W index: row j of C, col i of C -> j*5 + i
                U[i*5 + k][col] = 1
                V[k*5 + j][col] = 1
                W[j*5 + i][col] = 1
                col += 1
    
    assert col == 110, f"Expected 110 columns, got {col}"
    return U, V, W

def _try_random_rank_reduction(U, V, W, n, m, p, target_rank, deadline, rng):
    """
    Attempt to reduce rank by finding linear dependencies.
    Returns (U, V, W) if successful with target_rank, else None.
    """
    if time.time() > deadline - 10:
        return None
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    current_rank = U_arr.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Build the target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    # Try removing one column at a time by checking if its contribution
    # is already covered by other columns
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Simple check: if the column is all zeros, remove it
        if np.allclose(U_arr[:, col_to_remove], 0) and np.allclose(V_arr[:, col_to_remove], 0) and np.allclose(W_arr[:, col_to_remove], 0):
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            return U_new, V_new, W_new
        
        # Check if removing this column still gives the correct tensor
        # by seeing if the remaining columns can reproduce the tensor
        if current_rank - 1 == target_rank:
            # Build tensor without this column
            U_rem = np.delete(U_arr, col_to_remove, axis=1)
            V_rem = np.delete(V_arr, col_to_remove, axis=1)
            W_rem = np.delete(W_arr, col_to_remove, axis=1)
            T_rem = np.einsum('ic,jc,kc->ijk', U_rem, V_rem, W_rem, optimize=True)
            
            # Check if T_rem equals target
            if np.max(np.abs(T_rem - target)) < 1e-9:
                U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
                V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
                W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
                return U_new, V_new, W_new
    
    return None

def _build_6x6_decomposition():
    """
    Build a decomposition for 6x6x6 matrix multiplication.
    Strategy: Split 6 = 4 + 2.
    - Use rank-49 Strassen for the 4x4 sub-block.
    - Use rank-7 Strassen for the 2x2 sub-block.
    - Handle cross terms naively.
    """
    U4, V4, W4 = _build_strassen_4x4()
    U2, V2, W2 = _build_strassen_2x2()
    
    # Cross terms: all (i,k,j) where not all in {0,1,2,3} AND not all in {4,5}
    # Naive cross terms count:
    # i in 0-3, k in 4-5, j in 0-3: 4*2*4 = 32
    # i in 0-3, k in 0-3, j in 4-5: 4*4*2 = 32
    # i in 4-5, k in 0-3, j in 0-3: 2*4*4 = 32
    # i in 4-5, k in 4-5, j in 0-3: 2*2*4 = 16
    # i in 4-5, k in 0-3, j in 4-5: 2*4*2 = 16
    # i in 0-3, k in 4-5, j in 4-5: 4*2*2 = 16
    # i in 4-5, k in 4-5, j in 4-5: handled by 2x2 Strassen (7 terms)
    # i in 0-3, k in 0-3, j in 0-3: handled by 4x4 Strassen (49 terms)
    
    cross_count = 32 + 32 + 32 + 16 + 16 + 16
    rank = 49 + 7 + cross_count
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Copy 4x4 Strassen into indices 0-3
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
    
    # Copy 2x2 Strassen into indices 4-5 (offset by 16 in each dimension)
    col_offset = 49
    for i in range(4):
        for c in range(7):
            U[16 + i][col_offset + c] = U2[i][c]
            V[16 + i][col_offset + c] = V2[i][c]
            W[16 + i][col_offset + c] = W2[i][c]
    
    col = 49 + 7
    # Cross terms
    for i in range(6):
        for k in range(6):
            for j in range(6):
                i4 = i < 4
                k4 = k < 4
                j4 = j < 4
                i2 = i >= 4
                k2 = k >= 4
                j2 = j >= 4
                
                # Skip if all in 4-block or all in 2-block
                if i4 and k4 and j4:
                    continue
                if i2 and k2 and j2:
                    continue
                
                U[i*6 + k][col] = 1
                V[k*6 + j][col] = 1
                W[j*6 + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _embed_5x5_in_6x6(U6, V6, W6, rank6):
    """
    Extract the 5x5x5 sub-problem from a 6x6x6 decomposition.
    We use indices 0-4 (first 5 of 6) for the 5x5 problem.
    This gives a valid decomposition for 5x5x5 with rank <= rank6.
    """
    n = 5
    U5 = [[0]*rank6 for _ in range(n*n)]
    V5 = [[0]*rank6 for _ in range(n*n)]
    W5 = [[0]*rank6 for _ in range(n*n)]
    
    # Map 5x5 indices to 6x6 indices: 5x5 index (a,b) -> 6x6 index (a,b) if a,b < 5
    # For U: 5x5 index i*5+k -> 6x6 index i*6+k (for i,k < 5)
    # For V: 5x5 index k*5+j -> 6x6 index k*6+j (for k,j < 5)
    # For W: 5x5 index j*5+i -> 6x6 index j*6+i (for j,i < 5)
    
    for i in range(n):
        for k in range(n):
            src_i6 = i * 6 + k
            for c in range(rank6):
                U5[i*5 + k][c] = U6[src_i6][c]
    
    for k in range(n):
        for j in range(n):
            src_k6 = k * 6 + j
            for c in range(rank6):
                V5[k*5 + j][c] = V6[src_k6][c]
    
    for j in range(n):
        for i in range(n):
            src_j6 = j * 6 + i
            for c in range(rank6):
                W5[j*5 + i][c] = W6[src_j6][c]
    
    return U5, V5, W5

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # Strategy 1: 4+1 split (rank 110)
    if time.time() < deadline - 30:
        U, V, W = _build_5x5_decomposition()
        if _verify(U, V, W, n, m, p):
            print(f"Found valid rank-110 decomposition (4+1 split)")
            best_rank = 110
            best_result = {"U": U, "V": V, "W": W}
    
    # Strategy 2: 6x6 embedding
    if time.time() < deadline - 30:
        U6, V6, W6 = _build_6x6_decomposition()
        rank6 = U6[0].__len__()
        U5, V5, W5 = _embed_5x5_in_6x6(U6, V6, W6, rank6)
        if _verify(U5, V5, W5, n, m, p):
            print(f"Found valid rank-{rank6} decomposition (6x6 embedding)")
            if rank6 < best_rank:
                best_rank = rank6
                best_result = {"U": U5, "V": V5, "W": W5}
    
    # Fallback to schoolbook
    if best_result is None:
        print("Falling back to schoolbook rank-125")
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = rank_sb
    
    # Try rank reduction
    rng = np.random.RandomState(42)
    current_U = best_result["U"]
    current_V = best_result["V"]
    current_W = best_result["W"]
    current_rank = best_rank
    
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        
        target_rank = current_rank - 1
        result = _try_random_rank_reduction(
            current_U, current_V, current_W,
            n, m, p, target_rank, deadline, rng
        )
        
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                print(f"Found valid rank-{target_rank} decomposition")
                current_rank = target_rank
                current_U, current_V, current_W = U_new, V_new, W_new
                best_result = {"U": U_new, "V": V_new, "W": W_new}
            else:
                break
        else:
            break
    
    print(f"Best rank found: {current_rank}")
    return best_result
# EVOLVE-BLOCK-END
