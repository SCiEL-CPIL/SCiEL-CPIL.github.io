import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Rank 72 decomposition:
    # Part 1: 4x4x4 using Strassen (8 blocks of 2x2x2, each rank 7 = 56)
    # Part 2: 4x4x1 using schoolbook (rank 16)
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]   # (16, 72)
    V = [[0] * rank for _ in range(m * p)]   # (20, 72)
    W = [[0] * rank for _ in range(p * n)]   # (20, 72)
    
    col = 0
    
    # Strassen decomposition for 2x2x2 block
    # A blocks at (ar, ac), B blocks at (br, bc), C blocks at (cr, cc)
    # Each block is 2x2 in the respective matrix
    
    products = [
        (0, 0, 0, 0, 0, 0),  # A00 B00 -> C00
        (0, 1, 1, 0, 0, 0),  # A01 B10 -> C00
        (0, 0, 0, 1, 0, 1),  # A00 B01 -> C01
        (0, 1, 1, 1, 0, 1),  # A01 B11 -> C01
        (1, 0, 0, 0, 1, 0),  # A10 B00 -> C10
        (1, 1, 1, 0, 1, 0),  # A11 B10 -> C10
        (1, 0, 0, 1, 1, 1),  # A10 B01 -> C11
        (1, 1, 1, 1, 1, 1),  # A11 B11 -> C11
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        a_base_r, a_base_c = ar * 2, ac * 2
        b_base_r, b_base_c = br * 2, bc * 2
        c_base_r, c_base_c = cr * 2, cc * 2
        
        def a_idx(r, c):
            return (a_base_r + r) * 4 + (a_base_c + c)
        
        def b_idx(r, c):
            return (b_base_r + r) * 5 + (b_base_c + c)
        
        def c_idx(r, c):
            # W[k*n + i, rank] where k=col of C, i=row of C
            # Global row i = c_base_r + r, global col k = c_base_c + c
            return (c_base_c + c) * 4 + (c_base_r + r)
        
        # M1: (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> contributes to C10 (+1), C11 (-1)
        U[a_idx(1, 0)][col] += 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        W[c_idx(1, 0)][col] += 1
        W[c_idx(1, 1)][col] -= 1
        col += 1
        
        # M3: A00(B01-B11) -> contributes to C01 (+1), C11 (+1)
        U[a_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        V[b_idx(1, 1)][col] -= 1
        W[c_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> contributes to C00 (+1), C10 (+1)
        U[a_idx(1, 1)][col] += 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(0, 0)][col] -= 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(1, 0)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> contributes to C00 (-1), C01 (+1)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(0, 1)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] -= 1
        W[c_idx(0, 1)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> contributes to C11 (+1)
        U[a_idx(1, 0)][col] += 1
        U[a_idx(0, 0)][col] -= 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> contributes to C00 (+1)
        U[a_idx(0, 1)][col] += 1
        U[a_idx(1, 1)][col] -= 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    # A (4x4) * B_col4 (4x1) -> C_col4 (4x1)
    if time.time() < deadline - 2.0:
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 5 + 4][col] = 1
                W[4 * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
