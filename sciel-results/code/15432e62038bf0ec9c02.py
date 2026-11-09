"""5x5x5 matrix multiplication: rank-117 decomposition via 2+2+1 block Strassen."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 10.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-117 exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 2+2+1 block partition. For each of the 8 (2,2,2) block products,
    applies Strassen's rank-7 algorithm instead of schoolbook rank 8,
    saving 1 rank per block for a total saving of 8 (125 - 8 = 117).
    
    The remaining 19 block products (involving at least one size-1 block)
    use the standard schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (u_dict, v_dict, w_dict)
    
    def add_column(u_entries, v_entries, w_entries):
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
    
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        """Standard rank-1 decomposition for each (i,j,k) in the block."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    def strassen_2x2x2(off_i, off_j, off_k):
        """
        Add 7 rank-1 terms implementing Strassen's algorithm for 2x2x2
        matrix multiplication at global offsets (off_i, off_j, off_k).
        
        Local indices 0,1 map to global off_i+li, off_j+lj, off_k+lk.
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
        # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
        # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
        # M5 = (A00+A01)B11       -> C01 += M5, C00 -= M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        
        strassen_terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # Each is a dict: {(local_i, local_j): coeff}
            (
                {(0,0): 1, (1,1): 1},   # A00+A11
                {(0,0): 1, (1,1): 1},   # B00+B11
                {(0,0): 1, (1,1): 1}    # C00+C11
            ),
            (
                {(1,0): 1, (1,1): 1},   # A10+A11
                {(0,0): 1},              # B00
                {(1,0): 1, (1,1): -1}   # C10-C11
            ),
            (
                {(0,0): 1},              # A00
                {(0,1): 1, (1,1): -1},   # B01-B11
                {(0,1): 1, (1,1): 1}    # C01+C11
            ),
            (
                {(1,1): 1},              # A11
                {(1,0): 1, (0,0): -1},   # B10-B00
                {(0,0): 1, (1,0): 1}    # C00+C10
            ),
            (
                {(0,0): 1, (0,1): 1},   # A00+A01
                {(1,1): 1},              # B11
                {(0,1): 1, (0,0): -1}   # C01-C00
            ),
            (
                {(1,0): 1, (0,0): -1},   # A10-A00
                {(0,0): 1, (0,1): 1},    # B00+B01
                {(1,1): 1}               # C11
            ),
            (
                {(0,1): 1, (1,1): -1},   # A01-A11
                {(1,0): 1, (1,1): 1},    # B10+B11
                {(0,0): 1}               # C00
            ),
        ]
        
        for a_c, b_c, c_c in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in a_c.items():
                gi, gj = off_i + li, off_j + lj
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in b_c.items():
                gj, gk = off_j + lj, off_k + lk
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in c_c.items():
                gi, gk = off_i + li, off_k + lk
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)
    
    # Iterate over all 3x3x3 = 27 block products
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    strassen_2x2x2(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
    # Build output matrices
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
