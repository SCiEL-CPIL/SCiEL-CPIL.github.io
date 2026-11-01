import time

def construct():
    """
    Returns a rank-7 exact decomposition of the 2x2x2 matrix multiplication tensor.
    
    This corresponds to Strassen's algorithm for 2x2 matrix multiplication,
    which is known to be rank-minimal for this problem over the reals.
    
    The decomposition satisfies:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all i, j, k in {0, 1} (the identity tensor for matmul).
    
    Returns:
        dict with keys 'U', 'V', 'W'
        U: shape (4, 7) list of lists
        V: shape (4, 7) list of lists
        W: shape (4, 7) list of lists
    """
    # Deadline guard (not strictly necessary for O(1) return, but good practice per directive)
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing conventions:
    # U index: i*m + j  (0: A00, 1: A01, 2: A10, 3: A11)
    # V index: j*p + k  (0: B00, 1: B01, 2: B10, 3: B11)
    # W index: k*n + i  (0: C00, 1: C10, 2: C01, 3: C11)
    
    # U: (4, 7)
    # Rows: A00, A01, A10, A11
    U = [
        [1.0,  0.0,  1.0,  0.0,  1.0, -1.0,  0.0], # A00
        [0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0], # A01
        [0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0], # A10
        [1.0,  1.0,  0.0,  1.0,  0.0,  0.0, -1.0]  # A11
    ]
    
    # V: (4, 7)
    # Rows: B00, B01, B10, B11
    V = [
        [1.0,  1.0,  0.0, -1.0,  0.0,  1.0,  0.0], # B00
        [0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0], # B01
        [0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0], # B10
        [1.0,  0.0, -1.0,  0.0,  1.0,  0.0,  1.0]  # B11
    ]
    
    # W: (4, 7)
    # Rows: C00, C10, C01, C11
    # k*n + i:
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    
    # W rows correspond to coefficients of M1..M7 for each C element
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # C10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # C11 = 1*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    
    W = [
        [1.0,  0.0,  0.0,  1.0, -1.0,  0.0,  1.0], # C00 (idx 0)
        [0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0], # C10 (idx 1)
        [0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0], # C01 (idx 2)
        [1.0, -1.0,  1.0,  0.0,  0.0,  1.0,  0.0]  # C11 (idx 3)
    ]
    
    if time.time() > deadline:
        # Fallback, though unlikely to be reached
        pass

    return {
        "U": U,
        "V": V,
        "W": W
    }
