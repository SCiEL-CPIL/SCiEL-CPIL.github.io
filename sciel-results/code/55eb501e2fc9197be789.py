"""Matrix multiplication 2x2 x 2x13 - rank-minimal tensor decomposition"""

def construct():
    """Construct a rank-46 exact tensor decomposition for (2x2)*(2x13) matrix multiplication.
    
    Uses Strassen's algorithm on 6 blocks of 2 columns (rank 7 each) plus
    schoolbook multiplication on the final column (rank 4).
    Total rank: 6*7 + 4 = 46.
    
    Returns dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = 2, 2, 13
    total_rank = 46
    
    # Initialize factor matrices
    U = [[0.0] * total_rank for _ in range(n * m)]   # 4 x 46
    V = [[0.0] * total_rank for _ in range(m * p)]   # 26 x 46
    W = [[0.0] * total_rank for _ in range(p * n)]   # 26 x 46
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for i, val in u_dict.items():
            U[i][col] += val
        for i, val in v_dict.items():
            V[i][col] += val
        for i, val in w_dict.items():
            W[i][col] += val
        col += 1
    
    # Process 6 blocks of 2 columns using Strassen (rank 7 per block)
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        # B row 0 col k: index k in V; B row 1 col k: index 13+k in V
        # C row i col k: index k*2+i in W (trace convention W[k*n+i, r])
        # A[i,j]: index i*2+j in U
        
        # M1 = (a00 + a11)(b0k0 + b1k1)
        add_term({0: 1, 3: 1}, {k0: 1, 13 + k1: 1}, {2*k0: 1, 2*k1 + 1: 1})
        # M2 = (a10 + a11) * b0k0
        add_term({2: 1, 3: 1}, {k0: 1}, {2*k0 + 1: 1, 2*k1 + 1: -1})
        # M3 = a00 * (b0k1 - b1k1)
        add_term({0: 1}, {k1: 1, 13 + k1: -1}, {2*k1: 1, 2*k1 + 1: 1})
        # M4 = a11 * (b1k0 - b0k0)
        add_term({3: 1}, {13 + k0: 1, k0: -1}, {2*k0: 1, 2*k0 + 1: 1})
        # M5 = (a00 + a01) * b1k1
        add_term({0: 1, 1: 1}, {13 + k1: 1}, {2*k0: -1, 2*k1: 1})
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        add_term({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1 + 1: 1})
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        add_term({1: 1, 3: -1}, {13 + k0: 1, 13 + k1: 1}, {2*k0: 1})
    
    # Final single column k=12: schoolbook (rank 4)
    k = 12
    # C0k = a00*b0k + a01*b1k
    add_term({0: 1}, {k: 1}, {2*k: 1})
    add_term({1: 1}, {13 + k: 1}, {2*k: 1})
    # C1k = a10*b0k + a11*b1k
    add_term({2: 1}, {k: 1}, {2*k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2*k + 1: 1})
    
    return {"U": U, "V": V, "W": W}
