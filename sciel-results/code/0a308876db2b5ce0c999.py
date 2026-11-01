import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 6 Strassen blocks (rank 7 each) + 2 Rank-4 blocks (rank 4 each)
    rank = 6 * 7 + 2 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices i in {0,1}, j in {0,1}, k in {0,1}
        # Global i = i_off + i, j = j_off + j, k = k_off + k
        
        # 7 terms
        # Term 1: M1
        # A' = A00 + A11, B' = B00 + B11
        # C00 += 1, C11 += 1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 2: M2
        # A' = A10 + A11, B' = B00
        # C10 += 1
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 3: M3
        # A' = A00, B' = B10 - B11
        # C01 += 1, C11 += 1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = -1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: M4
        # A' = A11, B' = B01 - B00
        # C00 += 1, C10 += 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 5: M5
        # A' = A00 + A01, B' = B11
        # C00 -= 1, C01 += 1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # Term 6: M6
        # A' = A10 - A00, B' = B00 + B01
        # C11 += 1
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 7: M7
        # A' = A01 - A11, B' = B10 + B11
        # C00 += 1, C11 -= 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1

    # Helper to add Rank-4 block
    def add_rank4(i_off, j_off, k_off):
        nonlocal col
        # Local i in {0,1}, j in {0,1}, k in {0}
        # Global i = i_off + i, j = j_off + j, k = k_off + k
        
        # Term 1: A00 B00 -> C00
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1
        
        # Term 2: A01 B10 -> C00
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1
        
        # Term 3: A10 B00 -> C10
        U[(i_off+1)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: A11 B10 -> C10
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1

    # Block 1: A1 B11 (i=0, j=0, k=0)
    add_strassen(0, 0, 0)
    # Block 2: A1 B12 (i=0, j=0, k=2)
    add_strassen(0, 0, 2)
    # Block 3: A1 B13 (i=0, j=0, k=4)
    add_strassen(0, 0, 4)
    # Block 4: A1 B14 (i=0, j=0, k=6)
    add_rank4(0, 0, 6)
    
    # Block 5: A2 B21 (i=0, j=2, k=0)
    add_strassen(0, 2, 0)
    # Block 6: A2 B22 (i=0, j=2, k=2)
    add_strassen(0, 2, 2)
    # Block 7: A2 B23 (i=0, j=2, k=4)
    add_strassen(0, 2, 4)
    # Block 8: A2 B24 (i=0, j=2, k=6)
    add_rank4(0, 2, 6)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
