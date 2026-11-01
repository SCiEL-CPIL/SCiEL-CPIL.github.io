import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        if 0 <= row < len(mat) and 0 <= col < len(mat[0]):
            mat[row][col] = val

    for t in range(4):
        c_base = 11 * t
        # k indices for this block
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # --- Strassen Terms ---
        
        # Term 1: M1
        c = c_base
        set_val(U, 0, c, 1) # a00
        set_val(U, 4, c, 1) # a11
        set_val(V, 0*8 + k0, c, 1) # b0, k0
        set_val(V, 1*8 + k1, c, 1) # b1, k1
        set_val(W, k0*2 + 0, c, 1) # C0, k0
        set_val(W, k1*2 + 1, c, 1) # C1, k1
        
        # Term 2: M2
        c = c_base + 1
        set_val(U, 3, c, 1) # a10
        set_val(U, 4, c, 1) # a11
        set_val(V, 0*8 + k0, c, 1) # b0, k0
        set_val(W, k0*2 + 1, c, 1) # C1, k0
        set_val(W, k1*2 + 1, c, -1) # C1, k1
        
        # Term 3: M3
        c = c_base + 2
        set_val(U, 0, c, 1) # a00
        set_val(V, 1*8 + k0, c, 1) # b1, k0
        set_val(V, 1*8 + k1, c, -1) # b1, k1
        set_val(W, k1*2 + 0, c, 1) # C0, k1
        set_val(W, k1*2 + 1, c, 1) # C1, k1
        
        # Term 4: M4
        c = c_base + 3
        set_val(U, 4, c, 1) # a11
        set_val(V, 0*8 + k0, c, 1) # b0, k0
        set_val(V, 0*8 + k1, c, -1) # b0, k1
        set_val(W, k0*2 + 0, c, 1) # C0, k0
        set_val(W, k0*2 + 1, c, 1) # C1, k0
        
        # Term 5: M5
        c = c_base + 4
        set_val(U, 0, c, 1) # a00
        set_val(U, 1, c, 1) # a01
        set_val(V, 1*8 + k1, c, 1) # b1, k1
        set_val(W, k0*2 + 0, c, -1) # C0, k0
        set_val(W, k1*2 + 0, c, 1) # C0, k1
        
        # Term 6: M6
        c = c_base + 5
        set_val(U, 3, c, 1) # a10
        set_val(U, 0, c, -1) # a00
        set_val(V, 0*8 + k0, c, 1) # b0, k0
        set_val(V, 0*8 + k1, c, 1) # b0, k1
        set_val(W, k1*2 + 1, c, 1) # C1, k1
        
        # Term 7: M7
        c = c_base + 6
        set_val(U, 1, c, 1) # a01
        set_val(U, 4, c, -1) # a11
        set_val(V, 1*8 + k0, c, 1) # b1, k0
        set_val(V, 1*8 + k1, c, 1) # b1, k1
        set_val(W, k0*2 + 0, c, 1) # C0, k0
        
        # --- Rank 1 Terms for A2 B2 ---
        
        # Term 8: a02 * b2, k0
        c = c_base + 7
        set_val(U, 2, c, 1) # a02
        set_val(V, 2*8 + k0, c, 1) # b2, k0
        set_val(W, k0*2 + 0, c, 1) # C0, k0
        
        # Term 9: a02 * b2, k1
        c = c_base + 8
        set_val(U, 2, c, 1) # a02
        set_val(V, 2*8 + k1, c, 1) # b2, k1
        set_val(W, k1*2 + 0, c, 1) # C0, k1
        
        # Term 10: a12 * b2, k0
        c = c_base + 9
        set_val(U, 5, c, 1) # a12
        set_val(V, 2*8 + k0, c, 1) # b2, k0
        set_val(W, k0*2 + 1, c, 1) # C1, k0
        
        # Term 11: a12 * b2, k1
        c = c_base + 10
        set_val(U, 5, c, 1) # a12
        set_val(V, 2*8 + k1, c, 1) # b2, k1
        set_val(W, k1*2 + 1, c, 1) # C1, k1

    if time.time() > deadline:
        # Should not happen for this static construction
        pass

    return {"U": U, "V": V, "W": W}
