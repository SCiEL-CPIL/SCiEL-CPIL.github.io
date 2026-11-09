import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    rank = 41
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (index, value), etc."""
        nonlocal col
        for (idx, val) in u_entries:
            U[idx][col] = val
        for (idx, val) in v_entries:
            V[idx][col] = val
        for (idx, val) in w_entries:
            W[idx][col] = val
        col += 1
    
    def strassen_2x2x2(i_off, j_off, k_off):
        """Add Strassen decomposition for 2x2*2x2 at given offsets."""
        nonlocal col
        # U indices: (i_off+ii)*m + (j_off+jj)
        # V indices: (j_off+jj)*p + (k_off+kk)
        # W indices: (k_off+kk)*n + (i_off+ii)
        
        def u(idx_ii, idx_jj): return (i_off + idx_ii) * m + (j_off + idx_jj)
        def v(idx_jj, idx_kk): return (j_off + idx_jj) * p + (k_off + idx_kk)
        def w(idx_kk, idx_ii): return (k_off + idx_kk) * n + (i_off + idx_ii)
        
        # M1 = (A00+A11)(B00+B11)
        add_term([(u(0,0),1),(u(1,1),1)], [(v(0,0),1),(v(1,1),1)], [(w(0,0),1),(w(1,1),1)])
        # M2 = (A10+A11)B00
        add_term([(u(1,0),1),(u(1,1),1)], [(v(0,0),1)], [(w(0,1),1)])
        # M3 = A00(B10-B11)
        add_term([(u(0,0),1)], [(v(1,0),1),(v(1,1),-1)], [(w(1,0),1)])
        # M4 = A11(B01-B00)
        add_term([(u(1,1),1)], [(v(0,1),1),(v(0,0),-1)], [(w(0,0),1),(w(0,1),1)])
        # M5 = (A00+A01)B11
        add_term([(u(0,0),1),(u(0,1),1)], [(v(1,1),1)], [(w(0,0),-1),(w(1,0),1)])
        # M6 = (A10-A00)(B00+B10)
        add_term([(u(1,0),1),(u(0,0),-1)], [(v(0,0),1),(v(1,0),1)], [(w(1,1),1)])
        # M7 = (A01-A11)(B01+B11)
        add_term([(u(0,1),1),(u(1,1),-1)], [(v(0,1),1),(v(1,1),1)], [(w(0,0),1),(w(1,0),1),(w(1,1),1)])
    
    def dot_2x2x2x1(i_off, j_off, k_off):
        """Add rank-1 term for 2x2*2x1 at given offsets."""
        nonlocal col
        def u(idx_ii, idx_jj): return (i_off + idx_ii) * m + (j_off + idx_jj)
        def v(idx_jj, idx_kk): return (j_off + idx_jj) * p + (k_off + idx_kk)
        def w(idx_kk, idx_ii): return (k_off + idx_kk) * n + (i_off + idx_ii)
        
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        add_term([(u(0,0),1),(u(0,1),1),(u(1,0),1),(u(1,1),1)],
                 [(v(0,0),1),(v(1,0),1)],
                 [(w(0,0),1),(w(0,1),1)])
    
    def dot_4x1x5(i_off, j_off, k_off):
        """Add rank-1 term for 4x1*1x5 at given offsets."""
        nonlocal col
        def u(idx_ii, idx_jj): return (i_off + idx_ii) * m + (j_off + idx_jj)
        def v(idx_jj, idx_kk): return (j_off + idx_jj) * p + (k_off + idx_kk)
        def w(idx_kk, idx_ii): return (k_off + idx_kk) * n + (i_off + idx_ii)
        
        # C[i,k] = A[i,0]*B[0,k]
        u_entries = [(u(ii, 0), 1) for ii in range(4)]
        v_entries = [(v(0, kk), 1) for kk in range(5)]
        w_entries = [(w(kk, ii), 1) for kk in range(5) for ii in range(4)]
        add_term(u_entries, v_entries, w_entries)
    
    # Block 1: A[0:4, 0:2] * B[0:2, 0:5]
    # Top half: A[0:2, 0:2] * B[0:2, 0:5] = 2x2*2x5
    #   Strassen on B[0:2, 0:2]: 2x2*2x2
    strassen_2x2x2(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    #   Dot on B[0:2, 2:3]: 2x2*2x1
    dot_2x2x2x1(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    dot_2x2x2x1(0, 0, 3)  # A[0:2,0:2] * B[0:2,3:4] -> C[0:2,3:4]
    dot_2x2x2x1(0, 0, 4)  # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    
    # Bottom half: A[2:4, 0:2] * B[0:2, 0:5] = 2x2*2x5
    strassen_2x2x2(2, 0, 0)  # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    dot_2x2x2x1(2, 0, 2)  # A[2:4,0:2] * B[0:2,2:3] -> C[2:4,2:3]
    dot_2x2x2x1(2, 0, 3)  # A[2:4,0:2] * B[0:2,3:4] -> C[2:4,3:4]
    dot_2x2x2x1(2, 0, 4)  # A[2:4,0:2] * B[0:2,4:5] -> C[2:4,4:5]
    
    # Block 2: A[0:4, 2:4] * B[2:4, 0:5]
    # Top half: A[0:2, 2:4] * B[2:4, 0:5] = 2x2*2x5
    strassen_2x2x2(0, 2, 0)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    dot_2x2x2x1(0, 2, 2)  # A[0:2,2:4] * B[2:4,2:3] -> C[0:2,2:3]
    dot_2x2x2x1(0, 2, 3)  # A[0:2,2:4] * B[2:4,3:4] -> C[0:2,3:4]
    dot_2x2x2x1(0, 2, 4)  # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    
    # Bottom half: A[2:4, 2:4] * B[2:4, 0:5] = 2x2*2x5
    strassen_2x2x2(2, 2, 0)  # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    dot_2x2x2x1(2, 2, 2)  # A[2:4,2:4] * B[2:4,2:3] -> C[2:4,2:3]
    dot_2x2x2x1(2, 2, 3)  # A[2:4,2:4] * B[2:4,3:4] -> C[2:4,3:4]
    dot_2x2x2x1(2, 2, 4)  # A[2:4,2:4] * B[2:4,4:5] -> C[2:4,4:5]
    
    # Block 3: A[0:4, 4:5] * B[4:5, 0:5] = 4x1*1x5, rank 1
    dot_4x1x5(0, 4, 0)
    
    return {"U": U, "V": V, "W": W}
