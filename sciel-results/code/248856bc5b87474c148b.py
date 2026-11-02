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
    
    # Try to find a column that can be eliminated by adjusting others
    # For each column to try removing, solve for coefficients that preserve the tensor
    
    # Build the target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    # Try removing one column at a time
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Check if this column is a linear combination of others in a way that
        # allows us to absorb its contribution
        
        # Simple check: if the column is all zeros, remove it
        if np.allclose(U_arr[:, col_to_remove], 0) and np.allclose(V_arr[:, col_to_remove], 0) and np.allclose(W_arr[:, col_to_remove], 0):
            # Remove this column
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            return U_new, V_new, W_new
    
    # More advanced: try to find two columns that can be merged
    # This is computationally expensive, so only try a few
    if current_rank - target_rank <= 2:
        for col_a in range(current_rank):
            for col_b in range(col_a + 1, current_rank):
                if time.time() > deadline - 5:
                    break
                
                # Check if we can merge these two columns
                # The sum of their contributions must be representable by a single rank-1 term
                # This is generally not possible, but let's check for special cases
                
                # For now, skip this as it's complex
                pass
    
    return None

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Build the rank-110 decomposition
    U, V, W = _build_5x5_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        print(f"Found valid rank-110 decomposition")
        best_rank = 110
        best_result = {"U": U, "V": V, "W": W}
    else:
        print("Rank-110 verification failed, falling back to schoolbook")
        best_rank = 125
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
    
    # Try rank reduction
    rng = np.random.RandomState(42)
    
    # Try to reduce rank by 1 at a time
    current_U = best_result["U"]
    current_V = best_result["V"]
    current_W = best_result["W"]
    current_rank = best_rank
    
    for _ in range(10):  # Try at most 10 reductions
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
