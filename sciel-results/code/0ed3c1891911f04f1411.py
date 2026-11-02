"""Strassen-Winograd block decomposition for 3x4 * 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

# Strassen 2x2x2 decomposition:
# A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
# m1 = (a+d)(e+h) -> C[p]=+1, C[s]=+1
# m2 = d(f-h)     -> C[q]=-1, C[s]=+1
# m3 = (a-b)g     -> C[p]=+1, C[r]=+1
# m4 = (c+d)h     -> C[r]=+1, C[s]=-1
# m5 = (a+c)f     -> C[q]=+1, C[r]=-1
# m6 = (b-d)(e+g) -> C[p]=+1, C[q]=+1
# m7 = (a-c)(h-g) -> C[p]=-1, C[s]=-1

def strassen_222_contributions():
    """Return list of (U_coeff, V_coeff, W_coeff) for the 7 Strassen multipliers.
    U indices: [a, b, c, d] (row-major 2x2)
    V indices: [e, f, g, h] (row-major 2x2)
    W indices: [p, q, r, s] (row-major 2x2)
    """
    contributions = []
    # m1 = (a+d)(e+h)
    contributions.append(([1,0,0,1], [1,1,0,0], [1,0,0,1]))
    # m2 = d(f-h)
    contributions.append(([0,0,0,1], [0,1,0,-1], [0,-1,0,1]))
    # m3 = (a-b)g
    contributions.append(([1,-1,0,0], [0,0,1,0], [1,0,1,0]))
    # m4 = (c+d)h
    contributions.append(([0,0,1,1], [0,0,0,1], [0,0,1,-1]))
    # m5 = (a+c)f
    contributions.append(([1,0,1,0], [0,1,0,0], [0,1,0,-1]))
    # m6 = (b-d)(e+g)
    contributions.append(([0,1,0,-1], [1,0,1,0], [1,1,0,0]))
    # m7 = (a-c)(h-g)
    contributions.append(([1,0,-1,0], [0,0,-1,1], [-1,0,0,-1]))
    return contributions

def schoolbook_2x2x2():
    """Standard 8-term decomposition for 2x2x2."""
    contributions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = [0]*4; v = [0]*4; w = [0]*4
                u[i*2+j] = 1
                v[j*2+k] = 1
                w[k*2+i] = 1
                contributions.append((u, v, w))
    return contributions

def schoolbook_2x1x1():
    """2x1 * 1x1 -> 2x1. Rank 2."""
    contributions = []
    for i in range(2):
        u = [0]*2; v = [0]; w = [0]*2
        u[i] = 1
        v[0] = 1
        w[i] = 1
        contributions.append((u, v, w))
    return contributions

def schoolbook_1x1x1():
    """1x1 * 1x1 -> 1x1. Rank 1."""
    return ([(1,)], [(1,)], [(1,)])

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll partition the 3x4 matrix A into blocks and the 4x7 matrix B into blocks.
    # Strategy: Use Strassen where possible for 2x2x2 blocks, fall back to schoolbook otherwise.
    # 
    # Partition A (3x4) into row blocks and column blocks, B (4x7) into row blocks and column blocks.
    # Let's try: rows of A: [2, 1], cols of A / rows of B: [2, 2], cols of B: [2, 2, 2, 1]
    # Actually, let's be more systematic. 
    #
    # A is 3x4. B is 4x7.
    # Let's partition:
    #   A rows: [2, 1]
    #   A cols / B rows: [2, 2]  
    #   B cols: [2, 2, 2, 1]
    #
    # This gives sub-blocks:
    #   A[0:2, 0:2] * B[0:2, 0:2] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 0:2] * B[0:2, 2:4] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 0:2] * B[0:2, 4:6] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 0:2] * B[0:2, 6:7] -> 2x2x1 schoolbook (rank 4)
    #   A[0:2, 2:4] * B[2:4, 0:2] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 2:4] * B[2:4, 2:4] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 2:4] * B[2:4, 4:6] -> 2x2x2 Strassen (rank 7)
    #   A[0:2, 2:4] * B[2:4, 6:7] -> 2x2x1 schoolbook (rank 4)
    #   A[2:3, 0:2] * B[0:2, 0:2] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 0:2] * B[0:2, 2:4] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 0:2] * B[0:2, 4:6] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 0:2] * B[0:2, 6:7] -> 1x2x1 schoolbook (rank 2)
    #   A[2:3, 2:4] * B[2:4, 0:2] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 2:4] * B[2:4, 2:4] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 2:4] * B[2:4, 4:6] -> 1x2x2 schoolbook (rank 4)
    #   A[2:3, 2:4] * B[2:4, 6:7] -> 1x2x1 schoolbook (rank 2)
    #
    # Total rank: 7*6 + 4*2 + 4*6 + 2*2 = 42 + 8 + 24 + 4 = 78
    
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Build the global factor matrices
    U_rows = n * m  # 12
    V_rows = m * p  # 28
    W_rows = p * n  # 21
    
    U = []
    V = []
    W = []
    
    def add_column(u_local, v_local, w_local, a_row_start, a_col_start, b_row_start, b_col_start):
        """Add a column to U, V, W by mapping local indices to global.
        u_local: coefficients for A sub-block (ar x ac)
        v_local: coefficients for B sub-block (br x bc)
        w_local: coefficients for C sub-block (ar x bc)
        """
        ar = len(a_row_starts_list)  # placeholder, we'll handle differently
        # Actually, let's pass the dimensions explicitly
        pass
    
    # Let me restructure. We'll collect all columns first.
    columns = []
    
    def add_block(a_r, a_c, b_r, b_c, a_rs, a_cs, b_rs, b_cs):
        """Add contributions for a sub-block of A[a_rs:a_rs+a_r, a_cs:a_cs+a_c] * B[b_rs:b_rs+b_r, b_cs:b_cs+b_c].
        a_rs, a_cs: start row/col in A (global)
        b_rs, b_cs: start row/col in B (global)
        """
        if a_r == 2 and a_c == 2 and b_r == 2 and b_c == 2:
            contribs = strassen_222_contributions()
            local_ar, local_ac = 2, 2
            local_br, local_bc = 2, 2
            local_cr, local_cc = 2, 2
        elif a_r == 2 and a_c == 2 and b_r == 2 and b_c == 1:
            # 2x2 * 2x1 -> 2x1, rank 4 (schoolbook)
            contribs = []
            for i in range(2):
                for j in range(2):
                    u = [0]*4; v = [0]*2; w = [0]*2
                    u[i*2+j] = 1
                    v[j] = 1
                    w[i] = 1
                    contribs.append((u, v, w))
            local_ar, local_ac = 2, 2
            local_br, local_bc = 2, 1
            local_cr, local_cc = 2, 1
        elif a_r == 1 and a_c == 2 and b_r == 2 and b_c == 2:
            # 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
            contribs = []
            for j in range(2):
                for k in range(2):
                    u = [0]*2; v = [0]*4; w = [0]*2
                    u[j] = 1
                    v[j*2+k] = 1
                    w[k] = 1
                    contribs.append((u, v, w))
            local_ar, local_ac = 1, 2
            local_br, local_bc = 2, 2
            local_cr, local_cc = 1, 2
        elif a_r == 1 and a_c == 2 and b_r == 2 and b_c == 1:
            # 1x2 * 2x1 -> 1x1, rank 2 (schoolbook)
            contribs = []
            for j in range(2):
                u = [0]*2; v = [0]*2; w = [0]
                u[j] = 1
                v[j] = 1
                w[0] = 1
                contribs.append((u, v, w))
            local_ar, local_ac = 1, 2
            local_br, local_bc = 2, 1
            local_cr, local_cc = 1, 1
        else:
            # Fallback to schoolbook
            contribs = []
            for i in range(a_r):
                for j in range(a_c):
                    for k in range(b_c):
                        u = [0]*(a_r*a_c); v = [0]*(b_r*b_c); w = [0]*(a_r*b_c)
                        u[i*a_c+j] = 1
                        v[j*b_c+k] = 1
                        w[i*b_c+k] = 1
                        contribs.append((u, v, w))
            local_ar, local_ac = a_r, a_c
            local_br, local_bc = b_r, b_c
            local_cr, local_cc = a_r, b_c
        
        for (u_loc, v_loc, w_loc) in contribs:
            # Map local to global indices
            u_global = [0] * U_rows
            v_global = [0] * V_rows
            w_global = [0] * W_rows
            
            for li in range(local_ar):
                for lj in range(local_ac):
                    val = u_loc[li*local_ac + lj]
                    if val != 0:
                        gi = a_rs + li
                        gj = a_cs + lj
                        u_global[gi * m + gj] = val
            
            for lj in range(local_br):
                for lk in range(local_bc):
                    val = v_loc[lj*local_bc + lk]
                    if val != 0:
                        gj = b_rs + lj
                        gk = b_cs + lk
                        v_global[gj * p + gk] = val
            
            for li in range(local_cr):
                for lk in range(local_cc):
                    val = w_loc[li*local_cc + lk]
                    if val != 0:
                        gi = a_rs + li
                        gk = b_cs + lk
                        # W index: k*n + i
                        w_global[gk * n + gi] = val
            
            columns.append((u_global, v_global, w_global))
        
        if time.time() > deadline:
            return False
    
    # Partition: A rows [2,1], A cols/B rows [2,2], B cols [2,2,2,1]
    a_row_parts = [2, 1]
    a_col_parts = [2, 2]
    b_col_parts = [2, 2, 2, 1]
    
    a_row_starts = [0, 2]
    a_col_starts = [0, 2]
    b_col_starts = [0, 2, 4, 6]
    
    success = True
    for ar_idx, ar_size in enumerate(a_row_parts):
        a_rs = a_row_starts[ar_idx]
        for ac_idx, ac_size in enumerate(a_col_parts):
            a_cs = a_col_starts[ac_idx]
            b_rs = a_cs  # B row start = A col start
            for bc_idx, bc_size in enumerate(b_col_parts):
                b_cs = b_col_starts[bc_idx]
                success = add_block(ar_size, ac_size, ac_size, bc_size, a_rs, a_cs, b_rs, b_cs)
                if not success:
                    return None
    
    if not success:
        return None
    
    rank = len(columns)
    
    # Build final matrices
    U = [list(col[0]) for col in columns]
    V = [list(col[1]) for col in columns]
    W = [list(col[2]) for col in columns]
    
    return {"U": U, "V": V, "W": W}
