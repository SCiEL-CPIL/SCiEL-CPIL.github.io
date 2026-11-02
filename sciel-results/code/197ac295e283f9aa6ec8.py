"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Uses Block-Strassen decomposition:
    Split A (2x4) into A1 (2x2) and A2 (2x2).
    Split B (4x5) into B1 (2x5) and B2 (2x5).
    C = A1*B1 + A2*B2.
    Each 2x2 * 2x5 product is decomposed into:
    - 2x2 * 2x2 (Strassen, rank 7)
    - 2x2 * 2x2 (Strassen, rank 7)
    - 2x2 * 2x1 (Schoolbook, rank 4)
    Total rank per block: 18.
    Total rank: 36.
    """
    import time
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        """
        Add Strassen terms for X(2x2) * Y(2x2) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0:k0+2]
        contributes to C[i0:i0+2, k0:k0+2]
        """
        terms = [
            # m1 = (x11+x22)*(y11+y22)
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22)
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11)
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12)
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22)
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
            
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        """
        Add schoolbook terms for X(2x2) * Y(2x1) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0]
        contributes to C[i0:i0+2, k0]
        """
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))

    # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)   # cols 0-1
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)   # cols 2-3
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1) # col 4
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:5]
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)   # cols 0-1
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)   # cols 2-3
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1) # col 4
    
    rank = len(rank_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
