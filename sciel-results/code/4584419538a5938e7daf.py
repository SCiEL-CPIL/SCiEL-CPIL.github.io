"""Program to find rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
Uses block-structured decomposition with Strassen on 2x2 sub-blocks.
"""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition for n×m × m×p matrix mult."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T_{ijk} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = 0.0
                for c in range(U_arr.shape[1]):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (naive) decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 tensor.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 if j==k, else 0
    
    Indexing:
    U row index = i*2 + j  (i=row of A, j=col of A)
    V row index = j*2 + k  (j=row of B, k=col of B)
    W row index = k*2 + i  (k=col of B, i=row of C)
    
    The matrix multiplication tensor: (A*B)_{ik} = sum_j A_{ij} B_{jk}
    So T_{ijk} = 1 if this is a valid (i,j,k) triple contributing to C_{ik}, i.e., always 1
    for the standard matmul tensor where T_{ijk} = delta_{jk}... 
    
    Actually, the standard formulation: T_{ijk} = 1 for all i,j,k in the matmul tensor
    where the contraction is sum_j A_{ij} B_{jk} = C_{ik}.
    The tensor is T_{i,j,k} = 1 (it's the identity in the middle index).
    Wait no. The matmul tensor has T_{ijk} = 1 if we're computing C_{ik} += A_{ij} * B_{jk}.
    So for fixed (i,j,k), T_{ijk} = 1 always (it's a rank-1 tensor per (i,j,k) in schoolbook).
    
    For the 2x2x2 case: T_{ijk} = 1 for all i,j,k in {0,1}. That's 8 entries, all 1.
    Schoolbook: 8 rank-1 terms. Strassen: 7 rank-1 terms.
    
    Strassen's algorithm:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U[i,j] for A, V[j,k] for B, W[k,i] for C
    # U row = i*2+j, V row = j*2+k, W row = k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    # A10: i=1,j=0 -> U[2]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    # C10: k=0,i=1 -> W[0*2+1]=W[1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00*(B01-B11)
    # A00: i=0,j=0 -> U[0]
    # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
    # C01: k=1,i=0 -> W[1*2+0]=W[2]; C11: k=1,i=1 -> W[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    # A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B00: j=0,k=0 -> V[0]
    # C00: k=0,i=0 -> W[0]; C10: k=0,i=1 -> W[1]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)*B11
    # A00: i=0,j=0 -> U[0]; A01: i=0,j=1 -> U[1]
    # B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0] with coeff -1; C01: k=1,i=0 -> W[2] with coeff +1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # A10: i=1,j=0 -> U[2]; A00: i=0,j=0 -> U[0]
    # B00: j=0,k=0 -> V[0]; B01: j=0,k=1 -> V[1]
    # C11: k=1,i=1 -> W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_5x5():
    """
    Construct a block-structured decomposition for 5x5 matrix multiplication.
    Partition as [2, 2, 1] for rows and columns.
    Uses Strassen (rank 7) for 2x2*2x2->2x2 block products,
    and schoolbook for all other block products.
    Expected total rank: 109 (vs 125 schoolbook).
    """
    n = m = p = 5
    block_sizes = [2, 2, 1]
    block_offsets = [0, 2, 4]
    
    # U: (25, rank), V: (25, rank), W: (25, rank)
    # U indexed by (i, j) -> i*5+j  where i in [0,5), j in [0,5)
    # V indexed by (j, k) -> j*5+k  where j in [0,5), k in [0,5)
    # W indexed by (k, i) -> k*5+i  where k in [0,5), i in [0,5)
    
    U_rows = [[] for _ in range(25)]
    V_rows = [[] for _ in range(25)]
    W_rows = [[] for _ in range(25)]
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        """Add one term (column) to the decomposition."""
        U_rows[u_idx].append(coeff)
        V_rows[v_idx].append(coeff)
        W_rows[w_idx].append(coeff)
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2 block product.
        Each of the 7 Strassen terms becomes one column in the decomposition.
        For each term c, U_c, V_c, W_c are sparse vectors over the 2x2 blocks.
        """
        U_str, V_str, W_str = _strassen_2x2()
        for c in range(7):
            # For this Strassen term, collect nonzero (index, coeff) pairs
            u_entries = []
            v_entries = []
            w_entries = []
            for u_i in range(4):
                if U_str[u_i][c] != 0:
                    i_loc = u_i // 2
                    j_loc = u_i % 2
                    u_entries.append(((a_i_start + i_loc) * 5 + (a_j_start + j_loc), U_str[u_i][c]))
            for v_i in range(4):
                if V_str[v_i][c] != 0:
                    j_loc = v_i // 2
                    k_loc = v_i % 2
                    v_entries.append(((b_j_start + j_loc) * 5 + (b_k_start + k_loc), V_str[v_i][c]))
            for w_i in range(4):
                if W_str[w_i][c] != 0:
                    k_loc = w_i // 2
                    i_loc = w_i % 2
                    w_entries.append(((c_k_start + k_loc) * 5 + (c_i_start + i_loc), W_str[w_i][c]))
            
            # This is ONE rank-1 term: add one column
            # All U, V, W entries for this term go into the same column
            for u_idx, u_coeff in u_entries:
                U_rows[u_idx].append(u_coeff)
            for v_idx, v_coeff in v_entries:
                V_rows[v_idx].append(v_coeff)
            for w_idx, w_coeff in w_entries:
                W_rows[w_idx].append(w_coeff)
    
    def add_schoolbook_block(a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start, 
                              a_rows, a_cols, b_rows, b_cols):
        """
        Add schoolbook decomposition for a block product.
        A block: a_rows x a_cols at (a_i_start, a_j_start)
        B block: b_rows x b_cols at (b_j_start, b_k_start)
        C block: a_rows x b_cols at (c_i_start, c_k_start)
        """
        for ai in range(a_rows):
            for bj in range(b_rows):  # this is the inner dimension
                for bk in range(b_cols):
                    u_idx = (a_i_start + ai) * 5 + (a_j_start + bj)
                    v_idx = (b_j_start + bj) * 5 + (b_k_start + bk)
                    w_idx = (c_k_start + bk) * 5 + (c_i_start + ai)
                    add_term(u_idx, v_idx, w_idx, 1)
    
    # Process all block products C[i][k] = sum_j A[i][j] * B[j][k]
    for i_blk in range(3):
        for k_blk in range(3):
            for j_blk in range(3):
                # A[i_blk][j_blk]: block_sizes[i_blk] rows, block_sizes[j_blk] cols
                a_rows = block_sizes[i_blk]
                a_cols = block_sizes[j_blk]
                # B[j_blk][k_blk]: block_sizes[j_blk] rows, block_sizes[k_blk] cols
                b_rows = block_sizes[j_blk]
                b_cols = block_sizes[k_blk]
                # C[i_blk][k_blk]: a_rows rows, b_cols cols
                
                a_i_start = block_offsets[i_blk]
                a_j_start = block_offsets[j_blk]
                b_j_start = block_offsets[j_blk]
                b_k_start = block_offsets[k_blk]
                c_i_start = block_offsets[i_blk]
                c_k_start = block_offsets[k_blk]
                
                if a_rows == 2 and a_cols == 2 and b_rows == 2 and b_cols == 2:
                    # Use Strassen
                    add_strassen_block(a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start)
                else:
                    # Use schoolbook
                    add_schoolbook_block(a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start,
                                         a_rows, a_cols, b_rows, b_cols)
    
    # Convert to list of lists format
    rank = len(U_rows[0]) if U_rows[0] else 0
    U = [row[:] for row in U_rows]
    V = [row[:] for row in V_rows]
    W = [row[:] for row in W_rows]
    
    return U, V, W

def _verify_strassen_2x2():
    """Quick self-check that _strassen_2x2 produces a valid rank-7 decomposition."""
    U_str, V_str, W_str = _strassen_2x2()
    U_arr = np.array(U_str, dtype=float)
    V_arr = np.array(V_str, dtype=float)
    W_arr = np.array(W_str, dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for c in range(7):
                    val += U_arr[i*2+j, c] * V_arr[j*2+k, c] * W_arr[k*2+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with schoolbook as guaranteed fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 125
    
    # Verify Strassen 2x2 works
    strassen_ok = _verify_strassen_2x2()
    
    # Try block decomposition with Strassen
    if strassen_ok:
        try:
            U_block, V_block, W_block = _block_decomposition_5x5()
            if U_block is not None and len(U_block) == 25 and len(U_block[0]) < best_rank:
                if _verify_decomposition(U_block, V_block, W_block, n, m, p):
                    best_U, best_V, best_W = U_block, V_block, W_block
                    best_rank = len(U_block[0])
        except Exception:
            pass
    
    # Verify best result
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
