"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks."""
import time

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 2+2+1 block partition. For each 2x2x2 sub-block product,
    uses Strassen's rank-7 decomposition instead of schoolbook rank 8.
    
    Total rank: 117 (vs 125 schoolbook)
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (U_dict, V_dict, W_dict)
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 column. Each is a list of (index, value)."""
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Block index groups
    # Group 0: indices [0,1]
    # Group 1: indices [2,3]
    # Group 2: index [4]
    blocks = [[0, 1], [2, 3], [4]]
    
    # Schoolbook for a block
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen for 2x2x2 block with given offset
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """Add 7 columns for the 2x2x2 Strassen decomposition at given offsets."""
        # Local indices 0,1 map to global offset+0, offset+1
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00, contributes to C10 and -C11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = A00(B01-B11), contributes to C01 and C11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A11(B10-B00), contributes to C00 and C10
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (A00+A01)B11, contributes to C01 and -C00
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = offset_i + li, offset_j + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = offset_j + lj, offset_k + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = offset_i + li, offset_k + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Build all blocks: A[i_block][j_block] * B[j_block][k_block]
    # i_block, j_block, k_block in {0, 1, 2}
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                # If all three blocks are size 2, use Strassen
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    strassen_2x2x2(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
    # Build the matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
