"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best result before 140s
    def deadline_ok():
        return (time.time() - start_time) < 140
    
    # Primary: rank-88 Strassen-based decomposition
    try:
        result = _construct_rank88(n, m, p)
        if result is not None:
            # Quick verification
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    if not deadline_ok():
        # Fallback: schoolbook rank 96 (always valid)
        return _construct_schoolbook(n, m, p)
    
    return _construct_schoolbook(n, m, p)


def _construct_rank88(n, m, p):
    """
    Rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Strategy:
    - Row 0: schoolbook 1x4x8 -> rank 32
    - Rows 1-2: Strassen on 2x4x8 -> split into 4 blocks of 2x2x2, 
      each using 7 Strassen multiplications -> 4*7 = 28
      But we need to handle the 4 columns of A properly.
      
    Actually, let me think more carefully:
    - A is 3x4, B is 4x8, C is 3x8
    - Split A into A_top (1x4) and A_bottom (2x4)
    - C_top = A_top * B (1x8) -> schoolbook: 4*8 = 32
    - C_bottom = A_bottom * B (2x8)
    
    For C_bottom (2x4 * 4x8 -> 2x8):
    - Split the 4 columns of A_bottom into two 2-column blocks
    - Split the 8 columns of B into four 2-column blocks
    - For each pair (A_col_block, B_col_block): 2x2 * 2x2 -> 2x2 via Strassen (rank 7)
    - Total: 2 * 4 * 7 = 56
    
    Total rank: 32 + 56 = 88
    """
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    # Part 1: Schoolbook for A[0,:] (1x4) * B (4x8) -> C[0,:] (1x8)
    # C[0,k] = sum_j A[0,j] * B[j,k]
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3,:] (2x4) * B (4x8) -> C[1:3,:] (2x8)
    # A_bottom has 2 rows, 4 cols. Split cols into [0,1] and [2,3]
    # B has 4 rows, 8 cols. Split rows into [0,1] and [2,3], cols into 4 blocks of 2
    #
    # For each (A_col_block, B_row_block, B_col_block):
    #   A_sub is 2x2, B_sub is 2x2, C_sub is 2x2
    #   Use Strassen (7 multiplications)
    
    # A col blocks: [0,1] and [2,3]
    a_col_blocks = [(0, 1), (2, 3)]
    # B row blocks: [0,1] and [2,3]
    b_row_blocks = [(0, 1), (2, 3)]
    # B col blocks: [0,1], [2,3], [4,5], [6,7]
    b_col_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for a_cb in a_col_blocks:
        for b_rb in b_row_blocks:
            for b_cb in b_col_blocks:
                # A_sub: rows [1,2] of A, cols a_cb
                # B_sub: rows b_rb of B, cols b_cb
                # C_sub: rows [1,2] of C, cols b_cb
                
                a_r1, a_r2 = 1, 2
                a_c1, a_c2 = a_cb
                b_r1, b_r2 = b_rb
                b_c1, b_c2 = b_cb
                c_r1, c_r2 = 1, 2
                c_c1, c_c2 = b_cb
                
                # U indices: A[i,j] -> i*m + j
                u_a = a_r1 * m + a_c1  # A[1, a_c1]
                u_b = a_r1 * m + a_c2  # A[1, a_c2]
                u_c = a_r2 * m + a_c1  # A[2, a_c1]
                u_d = a_r2 * m + a_c2  # A[2, a_c2]
                
                # V indices: B[j,k] -> j*p + k
                v_e = b_r1 * p + b_c1  # B[b_r1, b_c1]
                v_f = b_r1 * p + b_c2  # B[b_r1, b_c2]
                v_g = b_r2 * p + b_c1  # B[b_r2, b_c1]
                v_h = b_r2 * p + b_c2  # B[b_r2, b_c2]
                
                # W indices: C[i,k] -> k*n + i (trace convention W[k*n+i, r])
                w_c11 = b_c1 * n + c_r1  # C[1, b_c1]
                w_c12 = b_c2 * n + c_r1  # C[1, b_c2]
                w_c21 = b_c1 * n + c_r2  # C[2, b_c1]
                w_c22 = b_c2 * n + c_r2  # C[2, b_c2]
                
                # Strassen 7 multiplications:
                # M1 = (a+d)(e+h)
                add_rank_unit(
                    {u_a: 1, u_d: 1},
                    {v_e: 1, v_h: 1},
                    {w_c11: 1, w_c22: 1}
                )
                # M2 = (c+d)(e)
                add_rank_unit(
                    {u_c: 1, u_d: 1},
                    {v_e: 1},
                    {w_c21: 1, w_c22: -1}
                )
                # M3 = a(f-h)
                add_rank_unit(
                    {u_a: 1},
                    {v_f: 1, v_h: -1},
                    {w_c12: 1, w_c22: 1}
                )
                # M4 = d(g-e)
                add_rank_unit(
                    {u_d: 1},
                    {v_g: 1, v_e: -1},
                    {w_c11: 1, w_c21: 1}
                )
                # M5 = (a+b)h
                add_rank_unit(
                    {u_a: 1, u_b: 1},
                    {v_h: 1},
                    {w_c11: -1, w_c12: 1}
                )
                # M6 = (c-a)(e+f)
                add_rank_unit(
                    {u_c: 1, u_a: -1},
                    {v_e: 1, v_f: 1},
                    {w_c22: 1}
                )
                # M7 = (b-d)(g+h)
                add_rank_unit(
                    {u_b: 1, u_d: -1},
                    {v_g: 1, v_h: 1},
                    {w_c11: 1}
                )
    
    # Check that we used exactly 88 rank units
    if col != rank:
        # Try to adjust or return None
        return None
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p = 96"""
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


def _verify_decomposition(decomp, n, m, p, num_trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    rank = U.shape[1]
    
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # Reshape
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The contribution to C[i,k] from rank-r term is:
            # sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += np.sum(u_mat[i, :] * v_mat[:, k] * w_mat[k, i])
        
        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            return False
    
    return True
# EVOLVE-BLOCK-END
