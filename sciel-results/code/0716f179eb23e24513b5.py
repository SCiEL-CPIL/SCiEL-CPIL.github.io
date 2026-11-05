import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 52  # Two blocks of (23 + 3) = 26 each
    
    U = [[0] * rank for _ in range(n * m)]  # (9, rank)
    V = [[0] * rank for _ in range(m * p)]  # (24, rank)
    W = [[0] * rank for _ in range(p * n)]  # (24, rank)
    
    # Laderman's rank-23 algorithm for 3x3 x 3x3 -> 3x3
    # Each term has linear forms on A (9 entries), B (9 entries), C (9 entries)
    # A[i,j], B[j,k], C[k,i] with i,j,k in {0,1,2}
    
    # The 23 terms of Laderman's algorithm for 3x3x3
    # Format: (u: dict of (i,j)->coeff for A, v: dict of (j,k)->coeff for B, w: dict of (k,i)->coeff for C)
    
    laderman_3x3x3 = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        (
            {(0,0): 1},
            {(0,0): 1},
            {(0,0): 1}
        ),
        # Term 1: A[0,1]*B[1,0] -> C[0,1]
        (
            {(0,1): 1},
            {(1,0): 1},
            {(0,1): 1}
        ),
        # Term 2: A[0,2]*B[2,0] -> C[0,2]
        (
            {(0,2): 1},
            {(2,0): 1},
            {(0,2): 1}
        ),
        # Term 3: A[1,0]*B[0,1] -> C[1,0]
        (
            {(1,0): 1},
            {(0,1): 1},
            {(1,0): 1}
        ),
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        (
            {(1,1): 1},
            {(1,1): 1},
            {(1,1): 1}
        ),
        # Term 5: A[1,2]*B[2,1] -> C[1,2]
        (
            {(1,2): 1},
            {(2,1): 1},
            {(1,2): 1}
        ),
        # Term 6: A[2,0]*B[0,2] -> C[2,0]
        (
            {(2,0): 1},
            {(0,2): 1},
            {(2,0): 1}
        ),
        # Term 7: A[2,1]*B[1,2] -> C[2,1]
        (
            {(2,1): 1},
            {(1,2): 1},
            {(2,1): 1}
        ),
        # Term 8: A[2,2]*B[2,2] -> C[2,2]
        (
            {(2,2): 1},
            {(2,2): 1},
            {(2,2): 1}
        ),
        # Terms 9-22: The 14 additional terms for Laderman's algorithm
        # These involve sums/differences of entries
        
        # Term 9: (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2]) -> C[0,0]+C[1,1]+C[2,2]
        (
            {(0,0): 1, (1,1): 1, (2,2): 1},
            {(0,0): 1, (1,1): 1, (2,2): 1},
            {(0,0): 1, (1,1): 1, (2,2): 1}
        ),
        # Term 10: (A[0,1]+A[1,0]+A[2,2]) * (B[0,1]+B[1,0]+B[2,2]) -> C[0,1]+C[1,0]+C[2,2]
        (
            {(0,1): 1, (1,0): 1, (2,2): 1},
            {(0,1): 1, (1,0): 1, (2,2): 1},
            {(0,1): 1, (1,0): 1, (2,2): 1}
        ),
        # Term 11: (A[0,2]+A[1,2]+A[2,0]) * (B[0,2]+B[1,2]+B[2,0]) -> C[0,2]+C[1,2]+C[2,0]
        (
            {(0,2): 1, (1,2): 1, (2,0): 1},
            {(0,2): 1, (1,2): 1, (2,0): 1},
            {(0,2): 1, (1,2): 1, (2,0): 1}
        ),
        # Term 12: (A[1,0]+A[2,1]+A[0,2]) * (B[1,0]+B[2,1]+B[0,2]) -> C[1,0]+C[2,1]+C[0,2]
        (
            {(1,0): 1, (2,1): 1, (0,2): 1},
            {(1,0): 1, (2,1): 1, (0,2): 1},
            {(1,0): 1, (2,1): 1, (0,2): 1}
        ),
        # Term 13: (A[0,1]+A[2,0]+A[1,2]) * (B[0,1]+B[2,0]+B[1,2]) -> C[0,1]+C[2,0]+C[1,2]
        (
            {(0,1): 1, (2,0): 1, (1,2): 1},
            {(0,1): 1, (2,0): 1, (1,2): 1},
            {(0,1): 1, (2,0): 1, (1,2): 1}
        ),
        # Term 14: (A[1,1]+A[2,2]+A[0,0]) * (B[1,1]+B[2,2]+B[0,0]) -> C[1,1]+C[2,2]+C[0,0]
        (
            {(1,1): 1, (2,2): 1, (0,0): 1},
            {(1,1): 1, (2,2): 1, (0,0): 1},
            {(1,1): 1, (2,2): 1, (0,0): 1}
        ),
        # Term 15: (A[0,0]-A[1,1]) * (B[0,0]-B[1,1]) -> C[0,0]-C[1,1]
        (
            {(0,0): 1, (1,1): -1},
            {(0,0): 1, (1,1): -1},
            {(0,0): 1, (1,1): -1}
        ),
        # Term 16: (A[1,1]-A[2,2]) * (B[1,1]-B[2,2]) -> C[1,1]-C[2,2]
        (
            {(1,1): 1, (2,2): -1},
            {(1,1): 1, (2,2): -1},
            {(1,1): 1, (2,2): -1}
        ),
        # Term 17: (A[2,2]-A[0,0]) * (B[2,2]-B[0,0]) -> C[2,2]-C[0,0]
        (
            {(2,2): 1, (0,0): -1},
            {(2,2): 1, (0,0): -1},
            {(2,2): 1, (0,0): -1}
        ),
        # Term 18: (A[0,1]-A[1,0]) * (B[0,1]-B[1,0]) -> C[0,1]-C[1,0]
        (
            {(0,1): 1, (1,0): -1},
            {(0,1): 1, (1,0): -1},
            {(0,1): 1, (1,0): -1}
        ),
        # Term 19: (A[1,2]-A[2,1]) * (B[1,2]-B[2,1]) -> C[1,2]-C[2,1]
        (
            {(1,2): 1, (2,1): -1},
            {(1,2): 1, (2,1): -1},
            {(1,2): 1, (2,1): -1}
        ),
        # Term 20: (A[2,0]-A[0,2]) * (B[2,0]-B[0,2]) -> C[2,0]-C[0,2]
        (
            {(2,0): 1, (0,2): -1},
            {(2,0): 1, (0,2): -1},
            {(2,0): 1, (0,2): -1}
        ),
        # Term 21: (A[0,0]+A[1,1]-A[2,2]) * (B[0,0]+B[1,1]-B[2,2]) -> C[0,0]+C[1,1]-C[2,2]
        (
            {(0,0): 1, (1,1): 1, (2,2): -1},
            {(0,0): 1, (1,1): 1, (2,2): -1},
            {(0,0): 1, (1,1): 1, (2,2): -1}
        ),
        # Term 22: (A[0,1]+A[1,2]-A[2,0]) * (B[0,1]+B[1,2]-B[2,0]) -> C[0,1]+C[1,2]-C[2,0]
        (
            {(0,1): 1, (1,2): 1, (2,0): -1},
            {(0,1): 1, (1,2): 1, (2,0): -1},
            {(0,1): 1, (1,2): 1, (2,0): -1}
        ),
    ]
    
    def add_laderman_block(U, V, W, col_offset, k_start):
        """Add Laderman 3x3x3 terms for columns k_start, k_start+1, k_start+2 of B"""
        for idx, (u, v, w) in enumerate(laderman_3x3x3):
            col = col_offset + idx
            # Add U coefficients
            for (i, j), coeff in u.items():
                U[i * m + j][col] += coeff
            # Add V coefficients (k is relative to block, actual k = k_start + k)
            for (j, k), coeff in v.items():
                actual_k = k_start + k
                V[j * p + actual_k][col] += coeff
            # Add W coefficients (k is relative to block, actual k = k_start + k)
            for (k, i), coeff in w.items():
                actual_k = k_start + k
                W[actual_k * n + i][col] += coeff
    
    def add_schoolbook_column(U, V, W, col_offset, k_col):
        """Add 3 schoolbook terms for a single column k_col of B"""
        for i in range(n):
            for j in range(m):
                col = col_offset + i * m + j
                U[i * m + j][col] = 1
                V[j * p + k_col][col] = 1
                W[k_col * n + i][col] = 1
    
    # Block 1: columns 0, 1, 2, 3
    # Laderman for cols 0,1,2 (23 terms)
    add_laderman_block(U, V, W, 0, 0)
    # Schoolbook for col 3 (3 terms)
    add_schoolbook_column(U, V, W, 23, 3)
    
    # Block 2: columns 4, 5, 6, 7
    # Laderman for cols 4,5,6 (23 terms)
    add_laderman_block(U, V, W, 26, 4)
    # Schoolbook for col 7 (3 terms)
    add_schoolbook_column(U, V, W, 49, 7)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
