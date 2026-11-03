"""
Rank-minimal tensor decomposition for (2x4) * (4x8) matrix multiplication.

Approach:
  We use a block structure and a "sum-product" trick over the inner dimension m=4
  to reduce the rank below the schoolbook 64.

  Let A be 2x4, B be 4x8. We write:
    A = [a00 a01 a02 a03]
        [a10 a11 a12 a13]
    B = [b00 b01 ... b07]
        [b10 b11 ... b17]
        [b20 b21 ... b27]
        [b30 b31 ... b37]

  We group the inner dimension into two pairs: (j=0,1) and (j=2,3).
  For each pair, we form 2x2 sub-multiplications.

  For pair 1 (j=0,1):
    Let A1 = [a00 a01; a10 a11] (2x2)
    Let B1 = [b00..b07; b10..b17] (2x8)

  For pair 2 (j=2,3):
    Let A2 = [a02 a03; a10 a13] (2x2)  -- wait, a12, a13
    Let A2 = [a02 a03; a12 a13] (2x2)
    Let B2 = [b20..b27; b30..b37] (2x8)

  Then C = A1 * B1 + A2 * B2.

  Now, for a 2x2 * 2x8 multiplication, we can use a rank-7 decomposition for the 2x2*2x2 part
  (Strassen's algorithm generalized) but here the second matrix is 2x8, so we can apply
  Strassen's 7-term scheme to the 2x2 block of A times each 2x2 block of B.

  Actually, let's think differently. The 2x2 * 2x8 product can be seen as 4 independent 2x2*2x2
  products (one for each pair of columns in B), each of which can be done with Strassen rank 7.
  That would give 4 * 7 = 28 terms for each half, so 56 total. That's better than 64.

  But wait - Strassen's algorithm for 2x2*2x2 uses 7 multiplications. The 2x2*2x8 can be decomposed
  as 4 separate 2x2*2x2 products, each using 7 terms = 28 terms. Two such halves = 56 terms.

  Let me construct this explicitly.

  Strassen for 2x2 * 2x2:
    P1 = (a00 + a11)(b00 + b11)
    P2 = (a10 + a11) b00
    P3 = a00 (b10 - b11)
    P4 = a11 (b01 - b00)
    P5 = (a00 + a01) b11
    P6 = (a10 - a00) (b00 + b10)
    P7 = (a01 - a11) (b01 + b11)

    c00 = P1 - P4 + P5 - P7
    c01 = P3 + P7
    c10 = P2 + P6
    c11 = P1 - P2 + P3 + P4

  For 2x2 * 2x8, we split B into four 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
  Each gives 7 terms, total 28 terms for the 2x2*2x8.

  Two such (for j-pairs 0,1 and 2,3) = 56 terms.

  Rank = 56.
"""

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct rank-56 tensor decomposition for (2x4)*(4x8) matrix multiplication.
    
    Uses Strassen's algorithm on 2x2*2x2 blocks, applied to 2x2*2x8 sub-problems.
    
    U shape: (n*m, rank) = (8, 56)
    V shape: (m*p, rank) = (32, 56)
    W shape: (p*n, rank) = (16, 56)
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    # Strassen's 7-term scheme for 2x2 * 2x2
    # For matrices A=[[a00,a01],[a10,a11]] and B=[[b00,b01],[b10,b11]]:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    #
    # c00 = P1 - P4 + P5 - P7
    # c01 = P3 + P7
    # c10 = P2 + P6
    # c11 = P1 - P2 + P3 + P4
    
    # Strassen coefficients for each output element
    # strassen[t][out_idx] where out_idx: 0=c00, 1=c01, 2=c10, 3=c11
    # and t: 0=P1, 1=P2, 2=P3, 3=P4, 4=P5, 5=P6, 6=P7
    strassen_coeff = [
        [1, 0, 0, 0],   # P1: c00+=1, c11+=1
        [0, 0, 0, -1],  # P2: c10+=1, c11-=1  -> wait, c10 = P2+P6, c11 = P1-P2+P3+P4
        [0, 1, 0, 1],   # P3: c01+=1, c11+=1
        [-1, 0, 0, 1],  # P4: c00-=1, c11+=1
        [1, 0, 0, 0],   # P5: c00+=1
        [0, 0, 1, 0],   # P6: c10+=1
        [-1, 1, 0, 0],  # P7: c00-=1, c01+=1
    ]
    
    # Fix: let me recompute
    # c00 = P1 - P4 + P5 - P7
    # c01 = P3 + P7
    # c10 = P2 + P6
    # c11 = P1 - P2 + P3 + P4
    
    # So coefficient matrix [coeff of Pt in c0i]:
    #        P1  P2  P3  P4  P5  P6  P7
    # c00:   1   0   0  -1   1   0  -1
    # c01:   0   0   1   0   0   0   1
    # c10:   0   1   0   0   0   1   0
    # c11:   1  -1   1   1   0   0   0
    
    strassen_coeff = [
        [1, 0, 0, -1, 1, 0, -1],  # c00
        [0, 0, 1, 0, 0, 0, 1],    # c01
        [0, 1, 0, 0, 0, 1, 0],    # c10
        [1, -1, 1, 1, 0, 0, 0],   # c11
    ]
    
    # For 2x2 * 2x2, the A coefficients for each Pt:
    # P1 = (a00+a11)(b00+b11): A-coeff: a00:1, a01:0, a10:0, a11:1
    # P2 = (a10+a11)b00:        A-coeff: a00:0, a01:0, a10:1, a11:1
    # P3 = a00(b10-b11):        A-coeff: a00:1, a01:0, a10:0, a11:0
    # P4 = a11(b01-b00):        A-coeff: a00:0, a01:0, a10:0, a11:1
    # P5 = (a00+a01)b11:        A-coeff: a00:1, a01:1, a10:0, a11:0
    # P6 = (a10-a00)(b00+b10):  A-coeff: a00:-1, a01:0, a10:1, a11:0
    # P7 = (a01-a11)(b01+b11):  A-coeff: a00:0, a01:1, a10:0, a11:-1
    
    a_coeffs = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1],  # P7
    ]
    
    # B coefficients for each Pt:
    # P1 = (a00+a11)(b00+b11): B-coeff: b00:1, b01:0, b10:0, b11:1
    # P2 = (a10+a11)b00:        B-coeff: b00:1, b01:0, b10:0, b11:0
    # P3 = a00(b10-b11):        B-coeff: b00:0, b01:0, b10:1, b11:-1
    # P4 = a11(b01-b00):        B-coeff: b00:-1, b01:1, b10:0, b11:0
    # P5 = (a00+a01)b11:        B-coeff: b00:0, b01:0, b10:0, b11:1
    # P6 = (a10-a00)(b00+b10):  B-coeff: b00:1, b01:0, b10:1, b11:0
    # P7 = (a01-a11)(b01+b11):  B-coeff: b00:0, b01:1, b10:0, b11:1
    
    b_coeffs = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 0, 1, -1],  # P3
        [-1, 1, 0, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 0, 1, 0],   # P6
        [0, 1, 0, 1],   # P7
    ]
    
    # Now build the full decomposition.
    # 
    # Indexing:
    # A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    #   U row index: i*m + j = i*4 + j
    # B is 4x8: B[j,k] where j in {0,1,2,3}, k in {0,...,7}
    #   V row index: j*p + k = j*8 + k
    # C is 2x8: C[i,k] 
    #   W row index: k*n + i = k*2 + i (trace convention W[k*n+i, r])
    #
    # We process two halves:
    # Half 1: j in {0, 1}
    # Half 2: j in {2, 3}
    #
    # For each half, we have a 2x2 A-block and a 2x8 B-block.
    # The 2x8 B-block is split into 4 2x2 blocks by column pairs: (0,1), (2,3), (4,5), (6,7)
    #
    # For each 2x2*2x2 sub-product, we use 7 terms.
    # Total: 2 halves * 4 column-pairs * 7 terms = 56 terms.
    
    col_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    half_pairs = [(0, 1), (2, 3)]  # (j_start, j_end) for the two halves
    
    col = 0
    for j0, j1 in half_pairs:
        for k0, k1 in col_pairs:
            for t in range(7):
                # t-th Strassen term for A-block rows (j0,j1) times B-block cols (k0,k1)
                
                # U contributions: A[i, j] for i in {0,1}, j in {j0, j1}
                # A-index within the 2x2: [i, jj] where jj = 0 if j==j0, 1 if j==j1
                for i in range(2):
                    for jj_local in range(2):
                        jj = j0 + jj_local
                        coeff = a_coeffs[t][i * 2 + jj_local]
                        if coeff != 0:
                            U[i * m + jj][col] += coeff
                
                # V contributions: B[j, k] for j in {j0, j1}, k in {k0, k1}
                # B-index within the 2x2: [jj, kk] where jj=0 if j==j0, 1 if j==j1
                #                             kk=0 if k==k0, 1 if k==k1
                for jj_local in range(2):
                    jj = j0 + jj_local
                    for kk_local in range(2):
                        kk = k0 + kk_local
                        coeff = b_coeffs[t][jj_local * 2 + kk_local]
                        if coeff != 0:
                            V[jj * p + kk][col] += coeff
                
                # W contributions: C[i, k] for i in {0,1}, k in {k0, k1}
                # C-index within the 2x2 output: [i, kk] where kk=0 if k==k0, 1 if k==k1
                for i in range(2):
                    for kk_local in range(2):
                        kk = k0 + kk_local
                        coeff = strassen_coeff[i * 2 + kk_local][t]
                        if coeff != 0:
                            W[kk * n + i][col] += coeff
                
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
