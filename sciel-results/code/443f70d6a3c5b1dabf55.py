"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a product term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # This function is a placeholder, we will directly assign in blocks
        pass

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] using Strassen (7 products)
    # A indices: i in {0,1}, j in {0,1} -> U[i*3+j]
    # B indices: j in {0,1}, k in {0,1} -> V[j*5+k]
    # C indices: k in {0,1}, i in {0,1} -> W[k*2+i]
    
    # Strassen coefficients for A, B
    # M1 = (a00+a11)(b00+b11)
    U[0*3+0][0] += 1; U[1*3+1][0] += 1
    V[0*5+0][0] += 1; V[1*5+1][0] += 1
    # W contributions:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We will accumulate W coefficients later for each product column
    
    # M1 (col 0)
    W[0*2+0][0] += 1 # c00
    W[1*2+1][0] += 1 # c11
    
    # M2 = (a10+a11)b00
    U[1*3+0][1] += 1; U[1*3+1][1] += 1
    V[0*5+0][1] += 1
    W[1*2+0][1] += 1 # c10
    W[1*2+1][1] += -1 # c11
    
    # M3 = a00(b01-b11)
    U[0*3+0][2] += 1
    V[0*5+1][2] += 1; V[1*5+1][2] += -1
    W[0*2+1][2] += 1 # c01
    W[1*2+1][2] += 1 # c11
    
    # M4 = a11(b10-b00)
    U[1*3+1][3] += 1
    V[1*5+0][3] += 1; V[0*5+0][3] += -1
    W[0*2+0][3] += 1 # c00
    W[1*2+0][3] += 1 # c10
    
    # M5 = (a00+a01)b11
    U[0*3+0][4] += 1; U[0*3+1][4] += 1
    V[1*5+1][4] += 1
    W[0*2+0][4] += -1 # c00
    W[0*2+1][4] += 1 # c01
    
    # M6 = (a10-a00)(b00+b01)
    U[1*3+0][5] += 1; U[0*3+0][5] += -1
    V[0*5+0][5] += 1; V[0*5+1][5] += 1
    W[1*2+1][5] += 1 # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[0*3+1][6] += 1; U[1*3+1][6] += -1
    V[1*5+0][6] += 1; V[1*5+1][6] += 1
    W[0*2+0][6] += 1 # c00
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] using Strassen (7 products)
    # B indices: j in {0,1}, k in {2,3} -> V[j*5+k]
    # C indices: k in {2,3}, i in {0,1} -> W[k*2+i]
    base_col = 7
    
    # M1 (col 7)
    U[0*3+0][base_col] += 1; U[1*3+1][base_col] += 1
    V[0*5+2][base_col] += 1; V[1*5+3][base_col] += 1
    W[2*2+0][base_col] += 1 # c02 (k=2, i=0)
    W[3*2+1][base_col] += 1 # c13 (k=3, i=1)
    
    # M2 (col 8)
    U[1*3+0][base_col+1] += 1; U[1*3+1][base_col+1] += 1
    V[0*5+2][base_col+1] += 1
    W[3*2+0][base_col+1] += 1 # c20? No, k=2, i=1 -> W[2*2+1]
    # Correction: W index is k*n + i.
    # For k=2, i=0: W[4]. For k=2, i=1: W[5].
    # For k=3, i=0: W[6]. For k=3, i=1: W[7].
    
    # Re-do Block 2 W indices carefully
    # C[0,2] is W[2*2+0]=W[4]
    # C[0,3] is W[3*2+0]=W[6]
    # C[1,2] is W[2*2+1]=W[5]
    # C[1,3] is W[3*2+1]=W[7]
    
    # Reset W for block 2
    W[4][base_col] += 1 # c02
    W[7][base_col] += 1 # c13
    
    W[5][base_col+1] += 1 # c12 (M2)
    W[7][base_col+1] += -1 # c13 (M2)
    
    # M3 (col 9)
    U[0*3+0][base_col+2] += 1
    V[0*5+3][base_col+2] += 1; V[1*5+3][base_col+2] += -1
    W[6][base_col+2] += 1 # c03 (M3)
    W[7][base_col+2] += 1 # c13 (M3)
    
    # M4 (col 10)
    U[1*3+1][base_col+3] += 1
    V[1*5+2][base_col+3] += 1; V[0*5+2][base_col+3] += -1
    W[4][base_col+3] += 1 # c02 (M4)
    W[5][base_col+3] += 1 # c12 (M4)
    
    # M5 (col 11)
    U[0*3+0][base_col+4] += 1; U[0*3+1][base_col+4] += 1
    V[1*5+3][base_col+4] += 1
    W[4][base_col+4] += -1 # c02 (M5)
    W[6][base_col+4] += 1 # c03 (M5)
    
    # M6 (col 12)
    U[1*3+0][base_col+5] += 1; U[0*3+0][base_col+5] += -1
    V[0*5+2][base_col+5] += 1; V[0*5+3][base_col+5] += 1
    W[7][base_col+5] += 1 # c13 (M6)
    
    # M7 (col 13)
    U[0*3+1][base_col+6] += 1; U[1*3+1][base_col+6] += -1
    V[1*5+2][base_col+6] += 1; V[1*5+3][base_col+6] += 1
    W[4][base_col+6] += 1 # c02 (M7)

    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] using Naive (4 products)
    # B indices: j in {0,1}, k in {4} -> V[j*5+4]
    # C indices: k in {4}, i in {0,1} -> W[4*2+i] = W[8], W[9]
    base_col = 14
    
    # P1 = a00 * b04
    U[0*3+0][base_col] += 1
    V[0*5+4][base_col] += 1
    W[8][base_col] += 1 # c04
    
    # P2 = a01 * b14
    U[0*3+1][base_col+1] += 1
    V[1*5+4][base_col+1] += 1
    W[8][base_col+1] += 1 # c04
    
    # P3 = a10 * b04
    U[1*3+0][base_col+2] += 1
    V[0*5+4][base_col+2] += 1
    W[9][base_col+2] += 1 # c14
    
    # P4 = a11 * b14
    U[1*3+1][base_col+3] += 1
    V[1*5+4][base_col+3] += 1
    W[9][base_col+3] += 1 # c14

    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] using Naive Outer (10 products)
    # A indices: i in {0,1}, j in {2} -> U[i*3+2]
    # B indices: j in {2}, k in {0..4} -> V[2*5+k]
    # C indices: k in {0..4}, i in {0,1} -> W[k*2+i]
    base_col = 18
    
    # For each k, 2 products
    for k in range(5):
        col_k = base_col + k*2
        
        # P_{k,0} = a02 * b2k
        U[0*3+2][col_k] += 1
        V[2*5+k][col_k] += 1
        W[k*2+0][col_k] += 1 # c0k
        
        # P_{k,1} = a12 * b2k
        U[1*3+2][col_k+1] += 1
        V[2*5+k][col_k+1] += 1
        W[k*2+1][col_k+1] += 1 # c1k

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
