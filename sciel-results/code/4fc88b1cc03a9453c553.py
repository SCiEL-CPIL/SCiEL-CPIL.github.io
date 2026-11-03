"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build all terms
    all_terms = []
    
    # Part 1: A1*B1 where A1 is 2x2 (cols 0,1), B1 is 2x10 (rows 0,1)
    # Split B1 into 5 blocks of 2x2, use Strassen for each
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        # m1: (a+d)(e+h), a=A[0,0], d=A[1,1], e=B[0,k0], h=B[1,k1]
        all_terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,k0): 1, (1,k1): 1},
            'W': {(0,0): 1, (1,1): 1}
        })
        # m2: (c+d)e, c=A[1,0], d=A[1,1], e=B[0,k0]
        all_terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,k0): 1},
            'W': {(1,0): 1, (1,1): -1}
        })
        # m3: a(f-h), a=A[0,0], f=B[0,k1], h=B[1,k1]
        all_terms.append({
            'U': {(0,0): 1},
            'V': {(0,k1): 1, (1,k1): -1},
            'W': {(0,1): 1, (1,1): 1}
        })
        # m4: d(g-e), d=A[1,1], g=B[1,k0], e=B[0,k0]
        all_terms.append({
            'U': {(1,1): 1},
            'V': {(1,k0): 1, (0,k0): -1},
            'W': {(0,0): 1, (1,0): 1}
        })
        # m5: (a+b)h, a=A[0,0], b=A[0,1], h=B[1,k1]
        all_terms.append({
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,k1): 1},
            'W': {(0,0): -1, (0,1): 1}
        })
        # m6: (c-a)(e+f), c=A[1,0], a=A[0,0], e=B[0,k0], f=B[0,k1]
        all_terms.append({
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,k0): 1, (0,k1): 1},
            'W': {(1,1): 1}
        })
        # m7: (b-d)(g+h), b=A[0,1], d=A[1,1], g=B[1,k0], h=B[1,k1]
        all_terms.append({
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,k0): 1, (1,k1): 1},
            'W': {(0,0): 1}
        })
    
    # Part 2: A2*B2 where A2 is 2x1 (col 2), B2 is 1x10 (row 2)
    # Schoolbook: 2*1*10 = 20 terms
    for i in range(n):
        for k in range(p):
            all_terms.append({
                'U': {(i, 2): 1},
                'V': {(2, k): 1},
                'W': {(i, k): 1}
            })
    
    rank = len(all_terms)
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(all_terms):
        for (i, j), val in term['U'].items():
            U[i * m + j][c] = val
        for (j, k), val in term['V'].items():
            V[j * p + k][c] = val
        for (i, k), val in term['W'].items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
