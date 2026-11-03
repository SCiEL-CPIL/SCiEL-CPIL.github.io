import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 108
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for M_{2,2,2}
    def add_strassen(i_start, j_start, k_start, col_offset):
        nonlocal col
        # Local indices:
        # U: i (2) x j (2) -> 4 entries
        # V: j (2) x k (2) -> 4 entries
        # W: k (2) x i (2) -> 4 entries (index k*2+i)
        
        # M1: (A00+A11)(B00+B11)
        # U: 1 at (0,0) and (1,1) -> idx 0, 3
        # V: 1 at (0,0) and (1,1) -> idx 0, 3
        # W: Coeffs for C
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4 + M7
        # C10 = M2 + M3 - M1 + M6
        # C11 = M1 - M3 + M2 - M6
        
        # We add 7 columns.
        
        # Term 1: M1
        U[i_start * m + j_start][col_offset] = 1
        U[(i_start+1) * m + (j_start+1)][col_offset] = 1
        V[j_start * p + k_start][col_offset] = 1
        V[(j_start+1) * p + (k_start+1)][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 1       # C00 coeff 1
        W[k_start * n + (i_start+1)][col_offset] = 0   # C01 coeff 0
        W[(k_start+1) * n + i_start][col_offset] = 0   # C10 coeff 0
        W[(k_start+1) * n + (i_start+1)][col_offset] = 1 # C11 coeff 1
        col_offset += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[(i_start+1) * m + j_start][col_offset] = 1
        U[(i_start+1) * m + (j_start+1)][col_offset] = 1
        V[j_start * p + k_start][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 0
        W[k_start * n + (i_start+1)][col_offset] = 0
        W[(k_start+1) * n + i_start][col_offset] = 1   # C10 coeff 1
        W[(k_start+1) * n + (i_start+1)][col_offset] = 1 # C11 coeff 1
        col_offset += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[i_start * m + j_start][col_offset] = 1
        V[j_start * p + (k_start+1)][col_offset] = 1
        V[(j_start+1) * p + (k_start+1)][col_offset] = -1
        W[k_start * n + i_start][col_offset] = 0
        W[k_start * n + (i_start+1)][col_offset] = 1   # C01 coeff 1
        W[(k_start+1) * n + i_start][col_offset] = 1   # C10 coeff 1
        W[(k_start+1) * n + (i_start+1)][col_offset] = -1 # C11 coeff -1
        col_offset += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[(i_start+1) * m + (j_start+1)][col_offset] = 1
        V[(j_start+1) * p + k_start][col_offset] = 1
        V[j_start * p + k_start][col_offset] = -1
        W[k_start * n + i_start][col_offset] = -1      # C00 coeff -1
        W[k_start * n + (i_start+1)][col_offset] = 1   # C01 coeff 1
        W[(k_start+1) * n + i_start][col_offset] = 0
        W[(k_start+1) * n + (i_start+1)][col_offset] = 0
        col_offset += 1
        
        # Term 5: M5 = (A00+A10)B11
        U[i_start * m + j_start][col_offset] = 1
        U[(i_start+1) * m + j_start][col_offset] = 1
        V[(j_start+1) * p + (k_start+1)][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 1       # C00 coeff 1
        W[k_start * n + (i_start+1)][col_offset] = 0
        W[(k_start+1) * n + i_start][col_offset] = 0
        W[(k_start+1) * n + (i_start+1)][col_offset] = -1 # C11 coeff -1
        col_offset += 1
        
        # Term 6: M6 = (A01-A00)(B00+B10)
        U[i_start * m + (j_start+1)][col_offset] = 1
        U[i_start * m + j_start][col_offset] = -1
        V[j_start * p + k_start][col_offset] = 1
        V[(j_start+1) * p + k_start][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 0
        W[k_start * n + (i_start+1)][col_offset] = 0
        W[(k_start+1) * n + i_start][col_offset] = 1   # C10 coeff 1
        W[(k_start+1) * n + (i_start+1)][col_offset] = -1 # C11 coeff -1
        col_offset += 1
        
        # Term 7: M7 = (A11-A10)(B01+B11)
        U[(i_start+1) * m + (j_start+1)][col_offset] = 1
        U[(i_start+1) * m + j_start][col_offset] = -1
        V[j_start * p + (k_start+1)][col_offset] = 1
        V[(j_start+1) * p + (k_start+1)][col_offset] = 1
        W[k_start * n + i_start][col_offset] = -1      # C00 coeff -1
        W[k_start * n + (i_start+1)][col_offset] = 1   # C01 coeff 1
        W[(k_start+1) * n + i_start][col_offset] = 0
        W[(k_start+1) * n + (i_start+1)][col_offset] = 1 # C11 coeff 1
        col_offset += 1
        
        return col_offset

    # Helper to add Outer Product terms for M_{2,1,2}
    def add_outer(i_start, j_start, k_start, col_offset):
        nonlocal col
        # A is 2x1, B is 1x2
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        # Term 1: C00
        U[i_start * m + j_start][col_offset] = 1
        V[j_start * p + k_start][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 1
        col_offset += 1
        
        # Term 2: C01
        U[i_start * m + j_start][col_offset] = 1
        V[j_start * p + (k_start+1)][col_offset] = 1
        W[k_start * n + (i_start+1)][col_offset] = 1 # Wait, C01 is i=0, k=1.
        # W index is k*n + i.
        # C01 -> k=1, i=0 -> W[1*n + 0]
        # My previous line set W[k_start * n + (i_start+1)] which is k=0, i=1. That is C10.
        # Correct:
        # C00: k=0, i=0
        # C01: k=1, i=0
        # C10: k=0, i=1
        # C11: k=1, i=1
        
        # Let's re-do W assignments carefully.
        
        # Reset and re-implement add_outer correctly
        pass

    # Re-implement add_outer correctly
    def add_outer(i_start, j_start, k_start, col_offset):
        nonlocal col
        # Term 1: A0 * B0 -> C00 (i=0, k=0)
        U[i_start * m + j_start][col_offset] = 1
        V[j_start * p + k_start][col_offset] = 1
        W[k_start * n + i_start][col_offset] = 1
        col_offset += 1
        
        # Term 2: A0 * B1 -> C01 (i=0, k=1)
        U[i_start * m + j_start][col_offset] = 1
        V[j_start * p + (k_start+1)][col_offset] = 1
        W[(k_start+1) * n + i_start][col_offset] = 1
        col_offset += 1
        
        # Term 3: A1 * B0 -> C10 (i=1, k=0)
        U[(i_start+1) * m + j_start][col_offset] = 1
        V[j_start * p + k_start][col_offset] = 1
        W[k_start * n + (i_start+1)][col_offset] = 1
        col_offset += 1
        
        # Term 4: A1 * B1 -> C11 (i=1, k=1)
        U[(i_start+1) * m + j_start][col_offset] = 1
        V[j_start * p + (k_start+1)][col_offset] = 1
        W[(k_start+1) * n + (i_start+1)][col_offset] = 1
        col_offset += 1
        
        return col_offset

    # j blocks: [0,1], [2,3], [4]
    # k blocks: [0,1], [2,3], [4,5]
    
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_block in [0, 2]:
        i_start = i_block
        for jb in j_blocks:
            j_start, j_dim = jb
            for kb in k_blocks:
                k_start, k_dim = kb
                if j_dim == 2 and k_dim == 2:
                    col = add_strassen(i_start, j_start, k_start, col)
                elif j_dim == 1 and k_dim == 2:
                    col = add_outer(i_start, j_start, k_start, col)
                else:
                    raise ValueError("Unsupported block sizes")
                    
    return {"U": U, "V": V, "W": W}

