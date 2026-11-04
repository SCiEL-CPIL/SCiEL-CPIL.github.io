"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 28.
    # Split m=3 into m1=1 (j=0) and m2=2 (j=1,2).
    # Part 1: M_{2,1,5} rank 10.
    # Part 2: M_{2,2,5} rank 18 (two M_{2,2,2} Strassen blocks + one M_{2,2,1} schoolbook block).
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0. i in {0,1}, k in {0..4}.
    # U[i*3+0], V[0*5+k], W[k*2+i]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1,2}. Map j=1->0, j=2->1 for Strassen.
    # Split k into {0,1}, {2,3}, {4}.
    
    # Helper to add a Strassen block for k_start in {0, 2}
    def add_strassen_block(k_start):
        nonlocal col
        # k_start is 0 or 2.
        # k1 = k_start, k2 = k_start + 1
        # j_prime 0 corresponds to j=1, j_prime 1 corresponds to j=2.
        
        # Strassen terms for 2x2 * 2x2
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        # x_ij = U[i*3 + (j_prime+1)]
        # y_jk = V[(j_prime+1)*5 + (k_start+k)]
        
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y01-y11)
        # M4 = x11(y10-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # Z00 = M1 - M4 + M5 - M7
        # Z01 = M3 + M7
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M2 - M6
        
        # Coefficients for U, V, W for each M
        # U indices: i*3 + j
        # V indices: j*5 + k
        # W indices: k*2 + i
        
        terms = [
            # M1
            {
                'U': {0*3+1: 1, 1*3+2: 1},
                'V': {1*5+k_start: 1, 2*5+(k_start+1): 1},
                'W': {k_start*n+0: 1, (k_start+1)*n+1: 1}
            },
            # M2
            {
                'U': {1*3+1: 1, 1*3+2: 1},
                'V': {1*5+k_start: 1},
                'W': {k_start*n+1: 1, (k_start+1)*n+1: -1} # Z10=M2+M4, Z11=M1-M3+M2-M6 -> coeff for M2 in Z11 is +1? No.
                # Z10 = M2 + M4. W[k_start, 1] gets M2.
                # Z11 = M1 - M3 + M2 - M6. W[k_start+1, 1] gets M2.
                # So W[k_start*n+1] += 1, W[(k_start+1)*n+1] += 1.
                # Wait, let's re-verify Z11 formula.
                # Z11 = M1 - M3 + M2 - M6. Yes, +M2.
            },
            # M3
            {
                'U': {0*3+1: 1},
                'V': {1*5+(k_start+1): 1, 2*5+(k_start+1): -1},
                'W': {(k_start+1)*n+0: 1, (k_start+1)*n+1: -1} # Z01=M3+M7 (+1), Z11=M1-M3... (-1)
            },
            # M4
            {
                'U': {1*3+2: 1},
                'V': {2*5+k_start: 1, 1*5+k_start: -1},
                'W': {k_start*n+0: -1, k_start*n+1: 1} # Z00=M1-M4... (-1), Z10=M2+M4 (+1)
            },
            # M5
            {
                'U': {0*3+1: 1, 0*3+2: 1},
                'V': {2*5+(k_start+1): 1},
                'W': {k_start*n+0: 1, (k_start+1)*n+1: -1} # Z00=M1...+M5 (+1), Z11=...-M6? No, Z11 doesn't have M5.
                # Z00 = M1 - M4 + M5 - M7.
                # Z11 = M1 - M3 + M2 - M6.
                # So M5 only affects Z00.
                # W[k_start*n+0] += 1.
                # W[(k_start+1)*n+1] is 0 for M5.
            },
            # M6
            {
                'U': {1*3+1: 1, 0*3+1: -1},
                'V': {1*5+k_start: 1, 1*5+(k_start+1): 1},
                'W': {(k_start+1)*n+1: -1} # Z11 = ... - M6.
            },
            # M7
            {
                'U': {0*3+2: 1, 1*3+2: -1},
                'V': {2*5+k_start: 1, 2*5+(k_start+1): 1},
                'W': {k_start*n+0: -1, (k_start+1)*n+0: 1} # Z00=...-M7, Z01=M3+M7 (+1)
            }
        ]
        
        # Correcting W coefficients based on formulas:
        # Z00 (i=0, k=k_start): M1 - M4 + M5 - M7
        # Z01 (i=0, k=k_start+1): M3 + M7
        # Z10 (i=1, k=k_start): M2 + M4
        # Z11 (i=1, k=k_start+1): M1 - M3 + M2 - M6
        
        # Let's rewrite the terms explicitly to avoid confusion.
        terms = [
            # M1: (x00+x11)(y00+y11)
            {
                'U': {1: 1, 5: 1}, # x00 (i=0,j=1->idx1), x11 (i=1,j=2->idx5)
                'V': {1*5+k_start: 1, 2*5+(k_start+1): 1}, # y00, y11
                'W': {k_start*n+0: 1, (k_start+1)*n+1: 1} # Z00, Z11
            },
            # M2: (x10+x11)y00
            {
                'U': {4: 1, 5: 1}, # x10 (i=1,j=1->idx4), x11 (i=1,j=2->idx5)
                'V': {1*5+k_start: 1}, # y00
                'W': {k_start*n+1: 1, (k_start+1)*n+1: 1} # Z10, Z11
            },
            # M3: x00(y01-y11)
            {
                'U': {1: 1}, # x00
                'V': {1*5+(k_start+1): 1, 2*5+(k_start+1): -1}, # y01, y11
                'W': {(k_start+1)*n+0: 1, (k_start+1)*n+1: -1} # Z01, Z11
            },
            # M4: x11(y10-y00)
            {
                'U': {5: 1}, # x11
                'V': {2*5+k_start: 1, 1*5+k_start: -1}, # y10, y00
                'W': {k_start*n+0: -1, k_start*n+1: 1} # Z00, Z10
            },
            # M5: (x00+x01)y11
            {
                'U': {1: 1, 2: 1}, # x00, x01 (i=0,j=2->idx2)
                'V': {2*5+(k_start+1): 1}, # y11
                'W': {k_start*n+0: 1} # Z00
            },
            # M6: (x10-x00)(y00+y01)
            {
                'U': {4: 1, 1: -1}, # x10, x00
                'V': {1*5+k_start: 1, 1*5+(k_start+1): 1}, # y00, y01
                'W': {(k_start+1)*n+1: -1} # Z11
            },
            # M7: (x01-x11)(y10+y11)
            {
                'U': {2: 1, 5: -1}, # x01, x11
                'V': {2*5+k_start: 1, 2*5+(k_start+1): 1}, # y10, y11
                'W': {k_start*n+0: -1, (k_start+1)*n+0: 1} # Z00, Z01
            }
        ]
        
        for t in terms:
            for idx, val in t['U'].items():
                U[idx][col] += val
            for idx, val in t['V'].items():
                V[idx][col] += val
            for idx, val in t['W'].items():
                W[idx][col] += val
            col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    
    # Group C: k=4. M_{2,2,1}. Rank 4.
    # X 2x2, Y 2x1.
    # Z00 = x00*y00 + x01*y10
    # Z10 = x10*y00 + x11*y10
    # 4 terms:
    # 1. x00*y00 -> W[4*2+0]=W[8]
    # 2. x01*y10 -> W[4*2+0]=W[8]
    # 3. x10*y00 -> W[4*2+1]=W[9]
    # 4. x11*y10 -> W[4*2+1]=W[9]
    
    # Term 1: x00 (U[1]) * y00 (V[1*5+4]=V[9])
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 2: x01 (U[2]) * y10 (V[2*5+4]=V[14])
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 3: x10 (U[4]) * y00 (V[9])
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Term 4: x11 (U[5]) * y10 (V[14])
    U[5][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
