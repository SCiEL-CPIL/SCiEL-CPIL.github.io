"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen, rank 7.
    Indexing: U[i*2+j, c] (8 x 7), V[j*2+k, c] (4 x 7), W[k*2+i, c] (4 x 7).
    Strassen formulas:
      C00 = M1 - M4 + M5 + M7
      C01 = M3 + M4
      C10 = M1 + M2
      C11 = M3 - M2 + M5 + M6
    where:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    """
    # Strassen factor matrices for 2x2 multiplication
    # U rows indexed by (i,j) for A[i,j], V rows by (j,k) for B[j,k], W rows by (k,i) for C[k,i]
    # A indices: A00=0, A01=1, A10=2, A11=3  (i*2+j)
    # B indices: B00=0, B01=1, B10=2, B11=3  (j*2+k)
    # C indices: C00=0, C01=1, C10=2, C11=3  (k*2+i)
    
    # For each Strassen factor M_t, we need:
    # M_t = sum over (i,j) of a_{i,j} * sum over (j,k) of b_{j,k}
    # Then C_{i,k} = sum_t w_{i,k,t} * M_t
    
    # Let's define each factor as a list of (U_entries, V_entries, W_entries)
    # U_entries: list of (row, val) for A
    # V_entries: list of (row, val) for B
    # W_entries: list of (row, val) for C
    
    factors = []
    
    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # C00 += M1, C10 += M1
    factors.append({
        'U': [(0, 1), (3, 1)],       # A00 + A11
        'V': [(0, 1), (3, 1)],       # B00 + B11
        'W': [(0, 1), (2, 1)],       # C00 + C10  (k*2+i: C00=0*2+0=0, C10=1*2+0=2)
    })
    
    # M2 = (A10+A11)B00
    # = A10*B00 + A11*B00
    # C10 -= M2, C11 -= M2
    factors.append({
        'U': [(2, 1), (3, 1)],       # A10 + A11
        'V': [(0, 1)],               # B00
        'W': [(2, -1), (3, -1)],     # -C10 - C11  (C10=2, C11=3)
    })
    
    # M3 = A00(B01-B11)
    # = A00*B01 - A00*B11
    # C01 += M3, C11 += M3
    factors.append({
        'U': [(0, 1)],               # A00
        'V': [(1, 1), (3, -1)],      # B01 - B11
        'W': [(1, 1), (3, 1)],       # C01 + C11  (C01=0*2+1=1, C11=1*2+1=3)
    })
    
    # M4 = A11(B10-B00)
    # = A11*B10 - A11*B00
    # C00 -= M4, C01 += M4
    factors.append({
        'U': [(3, 1)],               # A11
        'V': [(2, 1), (0, -1)],      # B10 - B00
        'W': [(0, -1), (1, 1)],      # -C00 + C01
    })
    
    # M5 = (A00+A01)B11
    # = A00*B11 + A01*B11
    # C00 += M5, C11 += M5
    factors.append({
        'U': [(0, 1), (1, 1)],       # A00 + A01
        'V': [(3, 1)],               # B11
        'W': [(0, 1), (3, 1)],       # C00 + C11
    })
    
    # M6 = (A10-A00)(B00+B01)
    # = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # C11 += M6
    factors.append({
        'U': [(2, 1), (0, -1)],      # A10 - A00
        'V': [(0, 1), (1, 1)],       # B00 + B01
        'W': [(3, 1)],               # C11
    })
    
    # M7 = (A01-A11)(B10+B11)
    # = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # C00 += M7
    factors.append({
        'U': [(1, 1), (3, -1)],      # A01 - A11
        'V': [(2, 1), (3, 1)],       # B10 + B11
        'W': [(0, 1)],               # C00
    })
    
    # Build the matrices
    U = [[0] * 7 for _ in range(8)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    for t, f in enumerate(factors):
        for row, val in f['U']:
            U[row][t] = val
        for row, val in f['V']:
            V[row][t] = val
        for row, val in f['W']:
            W[row][t] = val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Split into 8 independent 2x2 * 2x2 blocks:
    # A is 2x4, B is 4x8, C is 2x8
    # For block (i, k_block): A[i, 2*k_block:2*k_block+2] * B[2*k_block:2*k_block+2, :] -> C[i, :]
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x8.
    # We can split B's rows into 4 groups of 2: B[0:2,:], B[2:4,:]
    # And A's columns into 2 groups of 2: A[:,0:2], A[:,2:4]
    # C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:]
    # Each term is 2x2 * 2x8 -> 2x8
    # But 2x2 * 2x8 can be split into 4 independent 2x2 * 2x2 blocks
    # So total: 2 (from A/B split) * 4 (from B columns) = 8 blocks of 2x2*2x2
    
    # Block (a, b) where a in {0,1} indexes the pair of rows of B (and cols of A),
    # b in {0,1,2,3} indexes the pair of columns of B (and rows of C)
    # A_sub = A[:, 2*a : 2*a+2]  (2x2)
    # B_sub = B[2*a : 2*a+2, 2*b : 2*b+2]  (2x2)
    # C_sub = C[:, 2*b : 2*b+2]  (2x2)
    
    # Total rank = 8 * 7 = 56
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Get Strassen factors for 2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    for a in range(2):  # which pair of A cols / B rows
        for b in range(4):  # which pair of B cols / C cols
            base_col = (a * 4 + b) * 7
            
            # A indices: A[i, 2*a + j] for i in {0,1}, j in {0,1}
            # U row index: i*m + j = i*4 + j, but for the sub-block we use i*2+j
            # In the full U matrix, row = i*m + j = i*4 + j
            # For sub-block (a), the A column is 2*a + j, so full row = i*4 + (2*a + j)
            for i in range(2):
                for j in range(2):
                    full_row = i * m + (2 * a + j)
                    sub_row = i * 2 + j
                    for c in range(7):
                        if U_s[sub_row][c] != 0:
                            U[full_row][base_col + c] = U_s[sub_row][c]
            
            # B indices: B[2*a + j, 2*b + k] for j in {0,1}, k in {0,1}
            # V row index: j*p + k = j*8 + k
            # For sub-block, B row = 2*a + j, B col = 2*b + k
            # Full V row = (2*a + j)*p + (2*b + k)
            for j in range(2):
                for k in range(2):
                    full_row = (2 * a + j) * p + (2 * b + k)
                    sub_row = j * 2 + k
                    for c in range(7):
                        if V_s[sub_row][c] != 0:
                            V[full_row][base_col + c] = V_s[sub_row][c]
            
            # C indices: C[i, 2*b + k] for i in {0,1}, k in {0,1}
            # W row index: k*n + i = k*2 + i
            # For sub-block, C row = i, C col = 2*b + k
            # Full W row = (2*b + k)*n + i
            for k in range(2):
                for i in range(2):
                    full_row = (2 * b + k) * n + i
                    sub_row = k * 2 + i
                    for c in range(7):
                        if W_s[sub_row][c] != 0:
                            W[full_row][base_col + c] = W_s[sub_row][c]
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
