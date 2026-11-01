"""Matrix multiplication 2x2 by 2x10: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x10 matrix using 5 Strassen blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35  # 5 blocks * 7 Strassen products
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen terms for 2x2 matrices
    # U: A coeffs, V: B coeffs, W: C coeffs
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},   # U: a+d
            {(0,0): 1, (1,1): 1},   # V: e+h
            {(0,0): 1, (1,1): 1}    # W: C[0,0]+=1, C[1,1]+=1
        ),
        # M2 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},   # U: c+d
            {(0,0): 1},             # V: e
            {(0,1): 1, (1,1): -1}   # W: C[1,0]+=1, C[1,1]-=1
        ),
        # M3 = a(f-h)
        (
            {(0,0): 1},             # U: a
            {(0,1): 1, (1,1): -1},  # V: f-h
            {(1,0): 1, (1,1): 1}    # W: C[0,1]+=1, C[1,1]+=1
        ),
        # M4 = d(g-e)
        (
            {(1,1): 1},             # U: d
            {(1,0): 1, (0,0): -1},  # V: g-e
            {(0,0): 1, (0,1): 1}    # W: C[0,0]+=1, C[1,0]+=1
        ),
        # M5 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},   # U: a+b
            {(1,1): 1},             # V: h
            {(0,0): -1, (1,0): 1}   # W: C[0,0]-=1, C[0,1]+=1
        ),
        # M6 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},  # U: c-a
            {(0,0): 1, (0,1): 1},   # V: e+f
            {(1,1): 1}              # W: C[1,1]+=1
        ),
        # M7 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},  # U: b-d
            {(1,0): 1, (1,1): 1},   # V: g+h
            {(0,0): 1}              # W: C[0,0]+=1
        ),
    ]
    
    for r in range(5):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_entries.items():
                # B's column for this block is 2*r + k
                V[j * p + (2 * r + k)][col] = val
            
            for (k, i), val in w_entries.items():
                # C's column for this block is 2*r + k
                W[(2 * r + k) * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
