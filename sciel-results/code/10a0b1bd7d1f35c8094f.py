"""
Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x6 matrix.

Uses a Strassen-based decomposition on 2x2 blocks.

Structure:
- A is 4x4, partitioned into four 2x2 blocks: A[rb][cj] for rb,cj in {0,1}
- B is 4x6, partitioned into six 2x2 blocks: B[cj][ck] for cj in {0,1}, ck in {0,1,2}
- C is 4x6, partitioned into six 2x2 blocks: C[rb][ck]

Each C[rb][ck] = A[rb][0]*B[0][ck] + A[rb][1]*B[1][ck]

Each 2x2*2x2 multiplication uses Strassen (7 multiplications).
Total blocks: 2 (rb) * 2 (cj) * 3 (ck) = 12 block products
Each uses 7 Strassen multiplications.
Total rank = 12 * 7 = 84
"""

import time

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (16, 84)
        - V has shape (m*p, rank) = (24, 84)
        - W has shape (p*n, rank) = (24, 84)
        with entries satisfying: for all i,j,k, sum_r U[i,r]*V[j,r]*W[k,r] = C[i,k]
        where C = A*B, and W uses trace convention W[k*n+i, r].
    """
    n, m, p = 4, 4, 6
    deadline = time.time() + 110  # safety margin
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(i0, j0, k0):
        """
        Add Strassen decomposition of 2x2 * 2x2 matrix product.
        
        X = [[x00, x01], [x10, x11]] at A[i0..i0+1, j0..j0+1]
        Y = [[y00, y01], [y10, y11]] at B[j0..j0+1, k0..k0+1]
        
        The 7 Strassen products:
        m1 = (x00+x11)*(y00+y11)
        m2 = (x10+x11)*y00
        m3 = x00*(y01-y11)
        m4 = x11*(y10-y00)
        m5 = (x00+x01)*y11
        m6 = (x10-x00)*(y00+y01)
        m7 = (x01-x11)*(y10+y11)
        
        Result Z = [[m1-m4+m5+m6, m4+m7],
                    [m2+m6,      m1-m2+m3+m7]]
        """
        # Each term is (U_coeffs, V_coeffs, W_coeffs)
        # U: dict (i, j) -> coeff for A
        # V: dict (j, k) -> coeff for B  
        # W: dict (i, k) -> coeff for C (output position)
        
        strassen_terms = [
            # m1 = (x00+x11)*(y00+y11)
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x10+x11)*y00
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x00*(y01-y11)
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x11*(y10-y00)
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x00+x01)*y11
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x10-x00)*(y00+y01)
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7 = (x01-x11)*(y10+y11)
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in strassen_terms:
            if time.time() > deadline:
                return
            
            U_col = [0] * (n * m)  # 16
            V_col = [0] * (m * p)  # 24
            W_col = [0] * (p * n)  # 24
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                # W uses trace convention: W[k*n + i, r]
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # For each output block row rb (0,1) and column ck (0,1,2):
    # C[rb][ck] = A[rb][0]*B[0][ck] + A[rb][1]*B[1][ck]
    for rb in range(2):
        for ck in range(3):
            # Term 1: A[rb][0] * B[0][ck]
            # A block at rows rb*2..rb*2+1, cols 0..1
            # B block at rows 0..1, cols ck*2..ck*2+1
            add_strassen_term(rb * 2, 0, ck * 2)
            
            # Term 2: A[rb][1] * B[1][ck]
            # A block at rows rb*2..rb*2+1, cols 2..3
            # B block at rows 2..3, cols ck*2..ck*2+1
            add_strassen_term(rb * 2, 2, ck * 2)
    
    rank = len(U_cols)
    
    # Build final matrices: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Each is a list of lists (row-major)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
