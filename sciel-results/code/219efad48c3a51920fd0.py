"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-55 exact decomposition for 2x3x10 matrix multiplication.
    # Strategy: split the 3x10 product into 5 blocks of (3x2)*(2x2) along the k dimension.
    # For each block: use Strassen (rank 7) for the 2x2 core (using A columns 0,1 and B rows 0,1)
    #                 plus 4 outer-product terms for A column 2 with B rows 0,1 (rank 4).
    # Total rank = 5 * (7 + 4) = 55.
    
    deadline = time.time() + 115  # safety margin before 120s limit
    
    n, m, p = 2, 3, 10
    rank = 55
    
    # Pre-allocate U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    col = 0
    # 5 blocks, each covering 2 k-values
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part: 2x2 x 2x2 on A cols {0,1}, B rows {0,1}, C rows {0,1} ---
        # A_sub is 2x2: A[i][j] where i in {0,1} (rows of A), j in {0,1} (cols of A)
        # B_sub is 2x2: B[j][k] where j in {0,1} (rows of B), k in {k0,k1}
        # C_sub is 2x2: C[i][k] where i in {0,1} (rows of C), k in {k0,k1}
        
        # Strassen 7 products:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each Strassen term, we have coefficients on A entries and B entries.
        # U[row = i*m+j] gets the A coefficient
        # V[row = j*p+k] gets the B coefficient
        # W[row = k*n+i] gets the C coefficient
        
        # Term M1: A: A00(+1), A11(+1); B: B00(+1), B11(+1)
        # A00 -> U[0*3+0] = U[0]; A11 -> U[1*3+1] = U[4]
        # B00 -> V[0*10+k0] = V[k0]; B11 -> V[1*10+k1] = V[10+k1]
        # C contributions: C00(+1), C11(+1)
        # C00: W[k0*2+0] = W[2k0]; C11: W[k1*2+1] = W[2k1+1]
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0 + 0][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # Term M2: A: A10(+1), A11(+1); B: B00(+1)
        # A10 -> U[1*3+0] = U[3]; A11 -> U[4]
        # B00 -> V[k0]
        # C contributions: C10(+1)
        # C10: W[k0*2+1] = W[2k0+1]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # Term M3: A: A00(+1); B: B01(+1), B11(-1)
        # A00 -> U[0]; B01 -> V[0*10+k1] = V[k1]; B11 -> V[10+k1]
        # C contributions: C01(+1)
        # C01: W[k1*2+0] = W[2k1]
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] += -1
        W[2 * k1 + 0][col] += 1
        col += 1
        
        # Term M4: A: A11(+1); B: B10(+1), B00(-1)
        # A11 -> U[4]; B10 -> V[1*10+k0] = V[10+k0]; B00 -> V[k0]
        # C contributions: C00(+1), C10(+1)
        # C00: W[2k0]; C10: W[2k0+1]
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] += -1
        W[2 * k0 + 0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # Term M5: A: A00(+1), A01(+1); B: B11(+1)
        # A00 -> U[0]; A01 -> U[0*3+1] = U[1]
        # B11 -> V[10+k1]
        # C contributions: C00(-1), C01(+1)
        # C00: W[2k0]; C01: W[2k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0 + 0][col] += -1
        W[2 * k1 + 0][col] += 1
        col += 1
        
        # Term M6: A: A10(+1), A00(-1); B: B00(+1), B01(+1)
        # A10 -> U[3]; A00 -> U[0]
        # B00 -> V[k0]; B01 -> V[k1]
        # C contributions: C11(+1)
        # C11: W[2k1+1]
        U[3][col] += 1
        U[0][col] += -1
        V[k0][col] += 1
        V[k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # Term M7: A: A01(+1), A11(-1); B: B10(+1), B11(+1)
        # A01 -> U[1]; A11 -> U[4]
        # B10 -> V[10+k0]; B11 -> V[10+k1]
        # C contributions: C00(+1)
        # C00: W[2k0]
        U[1][col] += 1
        U[4][col] += -1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[2 * k0 + 0][col] += 1
        col += 1
        
        # --- Outer product part: A col 2 (A02, A12) with B rows 0,1 ---
        # 4 terms:
        # T1: A02 * B0_{k0} -> C_{0,k0}  i=0, j=2, k=k0
        # T2: A02 * B0_{k1} -> C_{0,k1}  i=0, j=2, k=k1
        # T3: A12 * B0_{k0} -> C_{1,k0}  i=1, j=2, k=k0
        # T4: A12 * B0_{k1} -> C_{1,k1}  i=1, j=2, k=k1
        
        # T1: U[0*3+2]=U[2], V[0*10+k0]=V[k0], W[k0*2+0]=W[2k0]
        U[2][col] += 1
        V[k0][col] += 1
        W[2 * k0 + 0][col] += 1
        col += 1
        
        # T2: U[2], V[k1], W[2k1]
        U[2][col] += 1
        V[k1][col] += 1
        W[2 * k1 + 0][col] += 1
        col += 1
        
        # T3: U[1*3+2]=U[5], V[0*10+k0]=V[k0], W[k0*2+1]=W[2k0+1]
        U[5][col] += 1
        V[k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # T4: U[5], V[k1], W[2k1+1]
        U[5][col] += 1
        V[k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
