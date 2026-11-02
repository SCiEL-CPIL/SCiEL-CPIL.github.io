"""Matrix multiplication 3,4,5 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    U = []
    V = []
    W = []
    
    def add_block(block_U, block_V, block_W, offset):
        """Add a block decomposition with given index mappings."""
        # block_U: list of (i, j, val)
        # block_V: list of (j, k, val)
        # block_W: list of (k, i, val)
        # We need to add to global U, V, W
        for idx, (ui, uval) in enumerate(block_U):
            i, j = ui
            U.append((i * m + j, uval, offset + idx))
        for idx, (vi, vval) in enumerate(block_V):
            j, k = vi
            V.append((j * p + k, vval, offset + idx))
        for idx, (wi, wval) in enumerate(block_W):
            k, i = wi
            W.append((k * n + i, wval, offset + idx))
    
    rank = 0
    U_rows = {}
    V_rows = {}
    W_rows = {}
    
    def _add(U_list, V_list, W_list):
        nonlocal rank
        r = len(U_list)
        for idx, (row_idx, val) in enumerate(U_list):
            if row_idx not in U_rows:
                U_rows[row_idx] = []
            U_rows[row_idx].append((rank + idx, val))
        for idx, (row_idx, val) in enumerate(V_list):
            if row_idx not in V_rows:
                V_rows[row_idx] = []
            V_rows[row_idx].append((rank + idx, val))
        for idx, (row_idx, val) in enumerate(W_list):
            if row_idx not in W_rows:
                W_rows[row_idx] = []
            W_rows[row_idx].append((rank + idx, val))
        rank += r
    
    # Strassen 2x2x2: rank 7
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_222(i0, j0, k0):
        """Return (U_list, V_list, W_list) for Strassen 2x2x2 at base indices."""
        # U: (i, j) -> value, for 2x2 block starting at (i0, j0)
        # V: (j, k) -> value, for 2x2 block starting at (j0, k0)
        # W: (k, i) -> value, for 2x2 block starting at (k0, i0)
        # Indices: i0, i0+1; j0, j0+1; k0, k0+1
        U_list = []
        V_list = []
        W_list = []
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[i0,j0], A22 = A[i0+1,j0+1]
        # B11 = B[j0,k0], B22 = B[j0+1,k0+1]
        U_list.append(((i0, j0), 1))
        U_list.append(((i0+1, j0+1), 1))
        V_list.append(((j0, k0), 1))
        V_list.append(((j0+1, k0+1), 1))
        W_list.append(((k0, i0), 1))
        W_list.append(((k0+1, i0+1), 1))
        # C11 += M1, C12 += M1? No.
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # W entries: (k, i) -> value
        # C11: k=k0, i=i0 -> +1 (M1)
        # C12: k=k0, i=i0 -> wait, C12 is at (i0, j0+1)? No.
        # C is 2x2: C[i,j] where i,j in 0,1
        # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
        # W: (k, i) -> but we also need j. 
        # Ah, W is (k, i) but the tensor is T[i,j,k]. 
        # The output C[i,j] = sum_k sum_r U[i,j,r] V[j,k,r] W[k,i,r]... 
        # Wait, the standard decomposition: C[i,j] = sum_r U[i,j,r] * V[j,k,r]... 
        # No, the standard is: C[i,j] = sum_{r} (sum_k A[i,k] U[k,r]) * (sum_k' B[k',j] V[r,k'])
        # Or: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]? 
        # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # And the tensor is T[i,j,k] for matrix mult C = A*B where A is n x m, B is m x p.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = A[i,j] B[j,k].
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for Strassen, we need to express the 8 products A[i,j]B[j,k] for i,j,k in {0,1}^2
        # as sum of 7 terms.
        
        # Let me redo. The 8 terms are:
        # A[0,0]B[0,0], A[0,0]B[0,1], A[0,0]B[1,0], A[0,0]B[1,1]
        # A[0,1]B[0,0], A[0,1]B[0,1], A[0,1]B[1,0], A[0,1]B[1,1]
        # A[1,0]B[0,0], A[1,0]B[0,1], A[1,0]B[1,0], A[1,0]B[1,1]
        # A[1,1]B[0,0], A[1,1]B[0,1], A[1,1]B[1,0], A[1,1]B[1,1]
        
        # Strassen computes C[0,0], C[0,1], C[1,0], C[1,1] where
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # This contributes to:
        # C[0,0] += A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # C[0,1] += ... no, M1 is a scalar. It's added to C[0,0] and C[1,1] etc.
        
        # Let me use the standard Strassen formulas.
        # The 7 products are:
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B22)
        # P4 = A22(B12 - B11)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Each P is a product of two sums. In the tensor decomposition,
        # each P corresponds to one rank-1 term? No, each P is a product
        # of (sum of A's) and (sum of B's), which expands to multiple A*B terms.
        # So each P is actually multiple rank-1 terms? No.
        
        # In the tensor decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r],
        # each r is a rank-1 term. For Strassen, we have 7 terms.
        # Each term r corresponds to one of the P's.
        # For term r = P1: U[i,j,1] = coeff of A[i,j] in (A11+A22)
        # V[j,k,1] = coeff of B[j,k] in (B11+B22)
        # W[k,i,1] = 1 (since P1 contributes to C11 and C22)
        # Wait, P1 contributes to C11 (i=0,k=0) and C22 (i=1,k=1).
        # So W[0,0,1] = 1, W[1,1,1] = 1, others 0? 
        # But W is (k, i) -> so W[k*n+i, r].
        # For C11: k=0, i=0 -> W[0*2+0, 1] = W[0,1] = 1
        # For C22: k=1, i=1 -> W[1*2+1, 1] = W[3,1] = 1
        
        # Let me implement this properly.
        
        # P1: (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]=1, C[1,1]=1 -> W[0,0]=1, W[3,1]=1 (for 2x2, k*n+i: 0*2+0=0, 1*2+1=3)
        U_list = [((0,0), 1), ((1,1), 1)]
        V_list = [((0,0), 1), ((1,1), 1)]
        W_list = [((0,0), 1), ((1,1), 1)]  # (k,i)
        # Add with offset 0
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+0, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+0, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+0, wval)
        rank += 1
        
        # P2: (A[1,0]+A[1,1])B[0,0]
        U_list = [((1,0), 1), ((1,1), 1)]
        V_list = [((0,0), 1)]
        W_list = [((0,1), 1)]  # C[1,0]: k=0, i=1 -> W[0*2+1, r] = W[1, r]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+1, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+1, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+1, wval)
        rank += 1
        
        # P3: A[0,0](B[1,0]-B[1,1])
        U_list = [((0,0), 1)]
        V_list = [((1,0), 1), ((1,1), -1)]
        W_list = [((0,0), 1), ((0,1), 1)]  # C[0,0]=1, C[0,1]=1 -> k=0,i=0 and k=0,i=0? 
        # C12 = P3+P5. C12 is C[0,1], so i=0, k=1. W[k=1, i=0] = W[1*2+0] = W[2]
        # C11 is C[0,0], i=0, k=0. W[0,0] = W[0]
        W_list = [((0,0), 1), ((1,0), 1)]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+2, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+2, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+2, wval)
        rank += 1
        
        # P4: A[1,1](B[0,1]-B[0,0])
        U_list = [((1,1), 1)]
        V_list = [((0,1), 1), ((0,0), -1)]
        W_list = [((0,0), 1), ((0,1), 1)]  # C11+=P4, C21+=P4
        # C11: i=0,k=0 -> W[0,0]
        # C21: i=1,k=0 -> W[0,1] = W[0*2+1] = W[1]
        W_list = [((0,0), 1), ((0,1), 1)]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+3, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+3, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+3, wval)
        rank += 1
        
        # P5: (A[0,0]+A[0,1])B[1,1]
        U_list = [((0,0), 1), ((0,1), 1)]
        V_list = [((1,1), 1)]
        W_list = [((0,0), -1), ((1,0), 1)]  # C11-=P5, C12+=P5, C22+=P5? 
        # C11 = P1+P4-P5+P7 -> C11: i=0,k=0 -> W[0,0] = -1
        # C12 = P3+P5 -> C12: i=0,k=1 -> W[1,0] = 1
        W_list = [((0,0), -1), ((1,0), 1)]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+4, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+4, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+4, wval)
        rank += 1
        
        # P6: (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        U_list = [((1,0), 1), ((0,0), -1)]
        V_list = [((0,0), 1), ((0,1), 1)]
        W_list = [((1,1), 1)]  # C22 += P6: i=1,k=1 -> W[1,1] = W[3]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+5, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+5, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+5, wval)
        rank += 1
        
        # P7: (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        U_list = [((0,1), 1), ((1,1), -1)]
        V_list = [((1,0), 1), ((1,1), 1)]
        W_list = [((0,0), 1)]  # C11 += P7: i=0,k=0 -> W[0,0]
        for idx, (ui, uval) in enumerate(U_list):
            _add_single(U_rows, (i0+ui[0])*m + (j0+ui[1]), rank+6, uval)
        for idx, (vi, vval) in enumerate(V_list):
            _add_single(V_rows, (j0+vi[0])*p + (k0+vi[1]), rank+6, vval)
        for idx, (wi, wval) in enumerate(W_list):
            _add_single(W_rows, (k0+wi[0])*n + (i0+wi[1]), rank+6, wval)
        rank += 1
    
    def _add_single(d, row, col, val):
        if d[row] is None:
            d[row] = {}
        if col not in d[row]:
            d[row][col] = 0
        d[row][col] += val
    
    # Initialize U_rows, V_rows, W_rows as dicts
    U_rows = {}
    V_rows = {}
    W_rows = {}
    rank = 0
    
    # We need a cleaner approach. Let me restart with a simpler method.
    pass

# Let me restart with a cleaner implementation.
def construct():
    n, m, p = 3, 4, 5
    
    # We'll build U, V, W as lists of (row, col, val) tuples, then convert to dense.
    U_entries = []
    V_entries = []
    W_entries = []
    rank = 0
    
    def add_term(U_dict, V_dict, W_dict):
        nonlocal rank
        # U_dict: {i*m+j: val}
        # V_dict: {j*p+k: val}
        # W_dict: {k*n+i: val}
        for row, val in U_dict.items():
            U_entries.append((row, rank, val))
        for row, val in V_dict.items():
            V_entries.append((row, rank, val))
        for row, val in W_dict.items():
            W_entries.append((row, rank, val))
        rank += 1
    
    # Strassen 2x2x2 at base (i0, j0, k0)
    def strassen(i0, j0, k0):
        # P1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
        add_term({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                 {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        
        # P2 = (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0]
        add_term({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                 {j0*p+k0: 1},
                 {k0*n+(i0+1): 1})
        
        # P3 = A[i0,j0](B[j0+1,k0]-B[j0+1,k0+1])
        add_term({i0*m+j0: 1},
                 {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): -1},
                 {k0*n+i0: 1, (k0+1)*n+i0: 1})
        
        # P4 = A[i0+1,j0+1](B[j0,k0+1]-B[j0,k0])
        add_term({(i0+1)*m+(j0+1): 1},
                 {j0*p+(k0+1): 1, j0*p+k0: -1},
                 {k0*n+i0: 1, k0*n+(i0+1): 1})
        
        # P5 = (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1]
        add_term({i0*m+j0: 1, i0*m+(j0+1): 1},
                 {(j0+1)*p+(k0+1): 1},
                 {k0*n+i0: -1, (k0+1)*n+i0: 1})
        
        # P6 = (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1])
        add_term({(i0+1)*m+j0: 1, i0*m+j0: -1},
                 {j0*p+k0: 1, j0*p+(k0+1): 1},
                 {(k0+1)*n+(i0+1): 1})
        
        # P7 = (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
        add_term({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                 {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1})
    
    # Schoolbook 2x2x2 at base (i0, j0, k0): rank 4
    def schoolbook_222(i0, j0, k0):
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    add_term({(i0+di)*m+(j0+dj): 1},
                             {(j0+dj)*p+(k0+dk): 1},
                             {(k0+dk)*n+(i0+di): 1})
    
    # Schoolbook for 1x2x2 at base (i0, j0, k0): rank 4
    def schoolbook_122(i0, j0, k0):
        for dj in range(2):
            for dk in range(2):
                add_term({i0*m+(j0+dj): 1},
                         {(j0+dj)*p+(k0+dk): 1},
                         {(k0+dk)*n+i0: 1})
    
    # Decompose 3x4x5:
    # Split i into 2+1: i=0,1 and i=2
    # For i=0,1 (2x4x5):
    #   Split j into 2+2: j=0,1 and j=2,3
    #   For j=0,1 (2x2x5):
    #     Split k into 2+3: k=0,1 and k=2,3,4
    #     For k=0,1 (2x2x2): Strassen, rank 7
    #     For k=2,3,4 (2x2x3): 
    #       Split k into 2+1: k=2,3 and k=4
    #       For k=2,3 (2x2x2): Strassen, rank 7
    #       For k=4 (2x2x1): Schoolbook, rank 4
    #       Total: 11
    #   For j=2,3 (2x2x5): same, rank 18
    #   Total for i=0,1: 36
    # For i=2 (1x4x5):
    #   Split j into 2+2: j=0,1 and j=2,3
    #   For j=0,1 (1x2x5): Schoolbook, rank 10
    #   For j=2,3 (1x2x5): Schoolbook, rank 10
    #   Total: 20
    # Grand total: 56
    
    # i=0,1; j=0,1; k=0,1: Strassen
    strassen(0, 0, 0)
    # i=0,1; j=0,1; k=2,3: Strassen
    strassen(0, 0, 2)
    # i=0,1; j=0,1; k=4: Schoolbook 2x2x1
    for dj in range(2):
        add_term({dj*m+0: 1, (dj+1)*m+0: 1},  # wrong, should be i=0,1
                 {dj*p+4: 1},
                 {4*n+0: 1, 4*n+1: 1})
    # Wait, let me fix the schoolbook for 2x2x1.
    
    # Let me restart the implementation more carefully.
    pass

def construct():
    n, m, p = 3, 4, 5
    U_entries = []
    V_entries = []
    W_entries = []
    rank = 0
    
    def add_term(U_dict, V_dict, W_dict):
        nonlocal rank
        for row, val in U_dict.items():
            U_entries.append((row, rank, val))
        for row, val in V_dict.items():
            V_entries.append((row, rank, val))
        for row, val in W_dict.items():
            W_entries.append((row, rank, val))
        rank += 1
    
    def strassen(i0, j0, k0):
        add_term({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                 {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        add_term({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                 {j0*p+k0: 1},
                 {k0*n+(i0+1): 1})
        add_term({i0*m+j0: 1},
                 {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): -1},
                 {k0*n+i0: 1, (k0+1)*n+i0: 1})
        add_term({(i0+1)*m+(j0+1): 1},
                 {j0*p+(k0+1): 1, j0*p+k0: -1},
                 {k0*n+i0: 1, k0*n+(i0+1): 1})
        add_term({i0*m+j0: 1, i0*m+(j0+1): 1},
                 {(j0+1)*p+(k0+1): 1},
                 {k0*n+i0: -1, (k0+1)*n+i0: 1})
        add_term({(i0+1)*m+j0: 1, i0*m+j0: -1},
                 {j0*p+k0: 1, j0*p+(k0+1): 1},
                 {(k0+1)*n+(i0+1): 1})
        add_term({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                 {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1})
    
    def schoolbook(i0, j0, k0, ni, nj, nk):
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    add_term({(i0+di)*m+(j0+dj): 1},
                             {(j0+dj)*p+(k0+dk): 1},
                             {(k0+dk)*n+(i0+di): 1})
    
    # 3x4x5 decomposition:
    # Block 1: i=0,1; j=0,1; k=0,1 (2x2x2): Strassen
    strassen(0, 0, 0)
    # Block 2: i=0,1; j=0,1; k=2,3 (2x2x2): Strassen
    strassen(0, 0, 2)
    # Block 3: i=0,1; j=0,1; k=4 (2x2x1): Schoolbook
    schoolbook(0, 0, 4, 2, 2, 1)
    # Block 4: i=0,1; j=2,3; k=0,1 (2x2x2): Strassen
    strassen(0, 2, 0)
    # Block 5: i=0,1; j=2,3; k=2,3 (2x2x2): Strassen
    strassen(0, 2, 2)
    # Block 6: i=0,1; j=2,3; k=4 (2x2x1): Schoolbook
    schoolbook(0, 2, 4, 2, 2, 1)
    # Block 7: i=2; j=0,1; k=0,1 (1x2x2): Schoolbook
    schoolbook(2, 0, 0, 1, 2, 2)
    # Block 8: i=2; j=0,1; k=2,3 (1x2x2): Schoolbook
    schoolbook(2, 0, 2, 1, 2, 2)
    # Block 9: i=2; j=0,1; k=4 (1x2x1): Schoolbook
    schoolbook(2, 0, 4, 1, 2, 1)
    # Block 10: i=2; j=2,3; k=0,1 (1x2x2): Schoolbook
    schoolbook(2, 2, 0, 1, 2, 2)
    # Block 11: i=2; j=2,3; k=2,3 (1x2x2): Schoolbook
    schoolbook(2, 2, 2, 1, 2, 2)
    # Block 12: i=2; j=2,3; k=4 (1x2x1): Schoolbook
    schoolbook(2, 2, 4, 1, 2, 1)
    
    # Convert to dense matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for row, col, val in U_entries:
        U[row][col] = val
    for row, col, val in V_entries:
        V[row][col] = val
    for row, col, val in W_entries:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

