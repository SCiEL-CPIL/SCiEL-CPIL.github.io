import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Build a rank-66 decomposition using block structure
    # A is 3x4, B is 4x6, C is 3x6
    # W index: k*n + i where k is row of C, i is col of C
    # U index: i*m + j where i is row of A, j is col of A
    # V index: j*p + k where j is row of B, k is col of B
    
    U = [[0] * 66 for _ in range(n * m)]  # 12 x 66
    V = [[0] * 66 for _ in range(m * p)]  # 24 x 66
    W = [[0] * 66 for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    def idx_U(i, j):
        return i * m + j
    
    def idx_V(j, k):
        return j * p + k
    
    def idx_W(k, i):
        return k * n + i
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
        col += 1
    
    # Strategy: 
    # Row 0 of A (1x4) times B (4x6) -> Row 0 of C (1x6): use 1x4 x 4x6 directly = 6 terms? No, use block structure
    # Actually let's use: split B into two 4x3 blocks
    # A[0,:] (1x4) * B[:,0:3] (4x3) = C[0,0:3] (1x3)
    # A[0,:] (1x4) * B[:,3:6] (4x3) = C[0,3:6] (1x3)
    # A[1:3,:] (2x4) * B[:,0:3] (4x3) = C[1:3,0:3] (2x3)
    # A[1:3,:] (2x4) * B[:,3:6] (4x3) = C[1:3,3:6] (2x3)
    
    # For 1x4 * 4x3 -> 1x3: rank 3 (direct)
    # For 2x4 * 4x3 -> 2x3: can use Strassen-like? 2x2*2x2 is rank 7, but 2x4*4x3...
    # Split 2x4 into two 2x2, 4x3 into 4x3
    # (2x2)*(4x3) + (2x2)*(4x3) -> each 2x2*4x3 has rank... 
    # Actually 2x2 * 4x3: treat as (2x2)*(4x3). 
    # Let's just use direct for the 1x4*4x3 blocks: 3 terms each
    # For 2x4*4x3: split into 2x2 blocks of A, multiply each by 4x3
    # 2x2 * 4x3: rank 3*2 = 6? Or use Strassen on 2x2*2x2 subblocks?
    
    # Let me try a cleaner approach:
    # C[0,0:3] = A[0,0:2]*B[0:2,0:3] + A[0,2:4]*B[2:4,0:3]
    # Each 1x2*2x3 = rank 3 (direct) or can we do better? 1x2*2x3: min rank is 3
    # So 2*3 = 6 for C[0,0:3], same for C[0,3:6] = 6, total 12 for row 0
    
    # C[1:3,0:3] = A[1:3,0:2]*B[0:2,0:3] + A[1:3,2:4]*B[2:4,0:3]
    # 2x2*2x3: rank? Direct is 6. Strassen doesn't directly apply.
    # Let's use direct: 2x2*2x3 = 6 terms each, so 12 for C[1:3,0:3]
    # Same for C[1:3,3:6] = 12
    # Total: 12+12+12+12 = 48? That's way too low, something is wrong.
    
    # Wait, I'm confusing myself. Let me just build the rank-66 and try to reduce.
    
    # Actually, let me just use the known good rank-66 construction and return it.
    # The current program already has a working rank-66. Let me just clean it up and return it.
    
    # Rebuild the rank-66 from the current program's logic
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = idx_U(a_r, a_c)
        A01 = idx_U(a_r, a_c + 1)
        A10 = idx_U(a_r + 1, a_c)
        A11 = idx_U(a_r + 1, a_c + 1)
        B00 = idx_V(b_r, b_c)
        B01 = idx_V(b_r, b_c + 1)
        B10 = idx_V(b_r + 1, b_c)
        B11 = idx_V(b_r + 1, b_c + 1)
        C00 = idx_W(c_c, c_r)
        C01 = idx_W(c_c + 1, c_r)
        C10 = idx_W(c_c, c_r + 1)
        C11 = idx_W(c_c + 1, c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        add_term([(A00,1),(A11,1)], [(B00,1),(B11,1)], [(C00,1),(C11,1)])
        # M2 = A10*(B00+B01)
        add_term([(A10,1)], [(B00,1),(B01,1)], [(C10,1),(C11,-1)])
        # M3 = A01*(B10-B11) -- wait, standard Strassen:
        # M3 = A01*(B11-B01)
        add_term([(A01,1)], [(B11,1),(B01,-1)], [(C01,1),(C11,1)])
        # M4 = (A11-A00)*(B11-B10)
        add_term([(A11,1),(A00,-1)], [(B11,1),(B10,-1)], [(C00,1),(C10,1)])
        # M5 = (A00+A01)*B11
        add_term([(A00,1),(A01,1)], [(B11,1)], [(C00,-1),(C01,1)])
        # M6 = (A10-A00)*(B00+B10)
        add_term([(A10,1),(A00,-1)], [(B00,1),(B10,1)], [(C11,1)])
        # M7 = (A01-A11)*(B10+B11)
        add_term([(A01,1),(A11,-1)], [(B10,1),(B11,1)], [(C00,1)])
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = idx_U(a_r, a_c)
        A1 = idx_U(a_r, a_c + 1)
        B00 = idx_V(b_r, b_c)
        B01 = idx_V(b_r, b_c + 1)
        B10 = idx_V(b_r + 1, b_c)
        B11 = idx_V(b_r + 1, b_c + 1)
        C0 = idx_W(c_c, c_r)
        C1 = idx_W(c_c + 1, c_r)
        
        add_term([(A0,1)], [(B00,1)], [(C0,1)])
        add_term([(A0,1)], [(B01,1)], [(C1,1)])
        add_term([(A1,1)], [(B10,1)], [(C0,1)])
        add_term([(A1,1)], [(B11,1)], [(C1,1)])
    
    # Row 0: A[0,0:2]*B[0:2,0:3] and A[0,2:4]*B[2:4,0:3] etc.
    # C[0,0:2] from A[0,0:2]*B[0:2,0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)  # 4 terms
    # C[0,2] from A[0,0:2]*B[0:2,2] + A[0,2:4]*B[2:4,2]
    # Hmm, this is getting complicated. Let me just use the exact same construction as before.
    
    # Reset
    col = 0
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]
    
    # Use the same block structure as the current program
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Verify col == 66
    assert col == 66, f"Expected 66 columns, got {col}"
    
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to reduce rank by checking for linear dependencies
    # Check if any column can be expressed as a linear combination of others
    # This is expensive, so let's just try a few random linear combinations
    
    # For now, return rank 66
    return {"U": U_arr, "V": V_arr, "W": W_arr}
