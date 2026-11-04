import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 48
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a 3x2x2 Strassen block
    # u_offset: offset in U for A (rows 0-2, cols 0-1 relative)
    # v_offset: offset in V for B (rows 0-1, cols 0-1 relative)
    # w_offset: offset in W for C (cols 0-1, rows 0-2 relative)
    # col_start: start column in U,V,W
    def add_strassen_322(u_off, v_off, w_off, col_start):
        # Strassen multipliers M1..M7
        # M1 = (A00+A11)(B00+B11)
        U[u_off + 0, col_start + 0] += 1.0
        U[u_off + 2, col_start + 0] += 1.0
        V[v_off + 0, col_start + 0] += 1.0
        V[v_off + 2, col_start + 0] += 1.0
        W[w_off + 0, col_start + 0] += 1.0 # C00
        W[w_off + 3, col_start + 0] += 1.0 # C11
        
        # M2 = (A10+A11)B00
        U[u_off + 2, col_start + 1] += 1.0
        U[u_off + 3, col_start + 1] += 1.0 # Wait, u_off+3 is row 1, col 1?
        # U index: i*m + j. i=1, j=1 -> 1*2+1=3. Correct.
        V[v_off + 0, col_start + 1] += 1.0
        W[w_off + 3, col_start + 1] += 1.0 # C10
        W[w_off + 4, col_start + 1] += -1.0 # C11
        
        # M3 = A00(B01-B11)
        U[u_off + 0, col_start + 2] += 1.0
        V[v_off + 1, col_start + 2] += 1.0
        V[v_off + 3, col_start + 2] += -1.0
        W[w_off + 1, col_start + 2] += 1.0 # C01
        W[w_off + 4, col_start + 2] += 1.0 # C11
        
        # M4 = A11(B10-B00)
        U[u_off + 3, col_start + 3] += 1.0
        V[v_off + 2, col_start + 3] += 1.0
        V[v_off + 0, col_start + 3] += -1.0
        W[w_off + 0, col_start + 3] += 1.0 # C00
        W[w_off + 3, col_start + 3] += 1.0 # C10
        
        # M5 = (A00+A01)B11
        U[u_off + 0, col_start + 4] += 1.0
        U[u_off + 1, col_start + 4] += 1.0
        V[v_off + 3, col_start + 4] += 1.0
        W[w_off + 0, col_start + 4] += -1.0 # C00
        W[w_off + 1, col_start + 4] += 1.0 # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[u_off + 2, col_start + 5] += 1.0
        U[u_off + 0, col_start + 5] += -1.0
        V[v_off + 0, col_start + 5] += 1.0
        V[v_off + 1, col_start + 5] += 1.0
        W[w_off + 4, col_start + 5] += 1.0 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[u_off + 1, col_start + 6] += 1.0
        U[u_off + 3, col_start + 6] += -1.0
        V[v_off + 2, col_start + 6] += 1.0
        V[v_off + 3, col_start + 6] += 1.0
        W[w_off + 0, col_start + 6] += 1.0 # C00
        W[w_off + 4, col_start + 6] += 1.0 # C11
        
        # Row 2 (i=2) schoolbook
        # C20 = A20 B00 + A21 B10
        # C21 = A20 B01 + A21 B11
        
        # Term A20 B00
        U[u_off + 4, col_start + 7] += 1.0 # i=2, j=0 -> 2*2+0=4
        V[v_off + 0, col_start + 7] += 1.0
        W[w_off + 2, col_start + 7] += 1.0 # C20 (k=0, i=2 -> 0*3+2=2)
        
        # Term A21 B10
        U[u_off + 5, col_start + 8] += 1.0 # i=2, j=1 -> 2*2+1=5
        V[v_off + 2, col_start + 8] += 1.0
        W[w_off + 2, col_start + 8] += 1.0
        
        # Term A20 B01
        U[u_off + 4, col_start + 9] += 1.0
        V[v_off + 1, col_start + 9] += 1.0
        W[w_off + 5, col_start + 9] += 1.0 # C21 (k=1, i=2 -> 1*3+2=5)
        
        # Term A21 B11
        U[u_off + 5, col_start + 10] += 1.0
        V[v_off + 3, col_start + 10] += 1.0
        W[w_off + 5, col_start + 10] += 1.0
        
        return col_start + 11

    # Helper to add a 3x2x1 block
    # u_offset: offset in U for A (rows 0-2, cols 0-1 relative)
    # v_offset: offset in V for B (rows 0-1, col 0 relative)
    # w_offset: offset in W for C (col 0, rows 0-2 relative)
    def add_321(u_off, v_off, w_off, col_start):
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        # C2 = A20 B00 + A21 B10
        
        # Term 1: B00 * [A00, A10, A20]
        U[u_off + 0, col_start + 0] += 1.0
        U[u_off + 2, col_start + 0] += 1.0
        U[u_off + 4, col_start + 0] += 1.0
        V[v_off + 0, col_start + 0] += 1.0
        W[w_off + 0, col_start + 0] += 1.0
        W[w_off + 1, col_start + 0] += 1.0
        W[w_off + 2, col_start + 0] += 1.0
        
        # Term 2: B10 * [A01, A11, A21]
        U[u_off + 1, col_start + 1] += 1.0
        U[u_off + 3, col_start + 1] += 1.0
        U[u_off + 5, col_start + 1] += 1.0
        V[v_off + 1, col_start + 1] += 1.0
        W[w_off + 0, col_start + 1] += 1.0
        W[w_off + 1, col_start + 1] += 1.0
        W[w_off + 2, col_start + 1] += 1.0
        
        return col_start + 2

    # Part 1: A_left (cols 0-1) * B_top (rows 0-1)
    # U offset: 0
    # V offset: 0
    # W offset: 0
    
    # Block 1: B cols 0-1
    col = add_strassen_322(0, 0, 0, 0)
    # Block 2: B cols 2-3
    # V offset for cols 2-3 is 2 (since p=5, but local p=2? No, global V index j*p+k.
    # j=0, k=2 -> 2. j=1, k=2 -> 7.
    # Wait, my helper assumes local V indices 0,1,2,3 for j=0,1 and k=0,1.
    # Global V index = j*5 + k.
    # For j=0, k=2 -> 2. j=0, k=3 -> 3.
    # For j=1, k=2 -> 7. j=1, k=3 -> 8.
    # My helper adds to v_off + 0, v_off + 1, v_off + 2, v_off + 3.
    # This assumes contiguous k.
    # For cols 2-3, k=2,3.
    # v_off should be 2?
    # v_off+0 -> 2 (j=0, k=2). Correct.
    # v_off+1 -> 3 (j=0, k=3). Correct.
    # v_off+2 -> 4? No, j=1, k=2 is 7.
    # So the helper assumes V is laid out as [j0k0, j0k1, j1k0, j1k1].
    # But global V is [j0k0, j0k1, j0k2, j0k3, j0k4, j1k0, ...].
    # So we cannot use a single offset for non-contiguous k in the global array if the helper assumes local contiguity.
    # We need to adjust the helper or the offsets.
    
    # Let's redefine the helper to take explicit V indices.
