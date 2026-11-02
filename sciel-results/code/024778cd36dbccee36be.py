import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # Block 1: 2x4x4 -> 4 * (2x2x2 Strassen) = 4 * 7 = 28
    # Block 2: 2x1x4 -> Schoolbook = 2 * 1 * 4 = 8
    # Block 3: 1x4x4 -> Schoolbook = 1 * 4 * 4 = 16
    # Block 4: 1x1x4 -> Schoolbook = 1 * 1 * 4 = 4
    # Total Rank = 28 + 8 + 16 + 4 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col_offset
        # Local indices 0,1
        # i_off, j_off, k_off are global offsets
        
        # M1: (A00+A11)(B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1
        U[i_off * m + j_off][col_offset] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_offset] = 1
        V[j_off * p + k_off][col_offset] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_offset] = 1
        W[k_off * n + i_off][col_offset] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_offset] = 1
        col_offset += 1
        
        # M2: (A10+A11)B00
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10=1
        U[(i_off + 1) * m + j_off][col_offset] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_offset] = 1
        V[j_off * p + k_off][col_offset] = 1
        W[k_off * n + (i_off + 1)][col_offset] = 1
        col_offset += 1
        
        # M3: A00(B01-B11)
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01=1
        U[i_off * m + j_off][col_offset] = 1
        V[j_off * p + (k_off + 1)][col_offset] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_offset] = -1
        W[(k_off + 1) * n + i_off][col_offset] = 1
        col_offset += 1
        
        # M4: A11(B10-B00)
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C00=1, C10=1
        U[(i_off + 1) * m + (j_off + 1)][col_offset] = 1
        V[(j_off + 1) * p + k_off][col_offset] = 1
        V[j_off * p + k_off][col_offset] = -1
        W[k_off * n + i_off][col_offset] = 1
        W[k_off * n + (i_off + 1)][col_offset] = 1
        col_offset += 1
        
        # M5: (A00+A01)B11
        # u: A00=1, A01=1
        # v: B11=1
        # w: C01=1, C11=-1
        U[i_off * m + j_off][col_offset] = 1
        U[i_off * m + (j_off + 1)][col_offset] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_offset] = 1
        W[(k_off + 1) * n + i_off][col_offset] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_offset] = -1
        col_offset += 1
        
        # M6: (A10-A00)(B00+B01)
        # u: A10=1, A00=-1
        # v: B00=1, B01=1
        # w: C11=1
        U[(i_off + 1) * m + j_off][col_offset] = 1
        U[i_off * m + j_off][col_offset] = -1
        V[j_off * p + k_off][col_offset] = 1
        V[j_off * p + (k_off + 1)][col_offset] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_offset] = 1
        col_offset += 1
        
        # M7: (A01-A11)(B10+B11)
        # u: A01=1, A11=-1
        # v: B10=1, B11=1
        # w: C00=1, C10=-1
        U[i_off * m + (j_off + 1)][col_offset] = 1
        U[(i_off + 1) * m + (j_off + 1)][col_offset] = -1
        V[(j_off + 1) * p + k_off][col_offset] = 1
        V[(j_off + 1) * p + (k_off + 1)][col_offset] = 1
        W[k_off * n + i_off][col_offset] = 1
        W[k_off * n + (i_off + 1)][col_offset] = -1
        col_offset += 1

    # Helper to add a schoolbook block
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col_offset
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col_offset] = 1
                    V[j * p + k][col_offset] = 1
                    W[k * n + i][col_offset] = 1
                    col_offset += 1

    # Block 1: 2x4x4 (i:0-1, j:0-3, k:0-3)
    # Composed of four 2x2x2 Strassen blocks
    add_strassen_block(0, 0, 0) # i:0-1, j:0-1, k:0-1
    add_strassen_block(0, 0, 2) # i:0-1, j:0-1, k:2-3
    add_strassen_block(0, 2, 0) # i:0-1, j:2-3, k:0-1
    add_strassen_block(0, 2, 2) # i:0-1, j:2-3, k:2-3
    
    # Block 2: 2x1x4 (i:0-1, j:4, k:0-3)
    add_schoolbook_block(0, 2, 4, 5, 0, 4)
    
    # Block 3: 1x4x4 (i:2, j:0-3, k:0-3)
    add_schoolbook_block(2, 3, 0, 4, 0, 4)
    
    # Block 4: 1x1x4 (i:2, j:4, k:0-3)
    add_schoolbook_block(2, 3, 4, 5, 0, 4)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-check
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.4f}s")
    print(f"Rank: {len(res['U'][0])}")
