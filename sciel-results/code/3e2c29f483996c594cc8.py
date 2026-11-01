"""Matrix multiplication 2,2,11 - rank-minimal decomposition via Strassen on 2x2 blocks.
The 2x2 x 2x11 multiplication can be decomposed by splitting the 11 columns into
five 2-column blocks plus one single column. For each 2x2 block we use Strassen's
7-term decomposition, and for the single column we use 4 terms (schoolbook).
Total rank = 5*7 + 4 = 39.
"""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 11
    # We build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using 0-indexed: i in {0,1}, j in {0,1}, k in {0,...,10}
    # W index: k*n + i (0-indexed: k*2 + i)
    
    rank = 39
    U = [[0.0] * rank for _ in range(4)]  # n*m = 4
    V = [[0.0] * rank for _ in range(22)] # m*p = 22
    W = [[0.0] * rank for _ in range(22)] # p*n = 22
    
    col = 0
    
    # Strassen decomposition for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]] (n x m)
    # B = [[b00, b01], [b10, b11]] (m x p)
    # C = A*B
    #
    # M1 = a00*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a01*(b10+b11)
    # M4 = a10*(b00-b10)
    # M5 = a11*(b01-b11)
    # M6 = (a00+a01)*b11
    # M7 = (a10-a01)*(b00+b10)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    strassen_coeffs = [
        # (U_idx, V_idx, W_idx, sign_U, sign_V, sign_W)
        # M1: a00 * (b00 + b11)
        ((0, 0, 0), 1, 1, 1),   # U[0]=a00, V[0]=b00, W[0]=C00
        ((0, 1, 0), 1, 1, 1),   # U[0]=a00, V[1]=b11, W[0]=C00  -- wait, need to be careful
    ]
    
    # Let me re-derive more carefully with explicit index mappings.
    # A is 2x2: A[i,j] where i in {0,1}, j in {0,1}
    # B is 2x2: B[j,k] where j in {0,1}, k in {0,1}
    # C is 2x2: C[i,k] where i in {0,1}, k in {0,1}
    #
    # U index for A[i,j] is i*m + j = i*2 + j
    # V index for B[j,k] is j*p_block + k where p_block=2, so j*2 + k
    # W index for C[i,k] is k*n + i = k*2 + i
    
    # Strassen:
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,1] * (B[1,0] + B[1,1])
    # M4 = A[1,0] * (B[0,0] - B[1,0])
    # M5 = A[1,1] * (B[0,1] - B[1,1])
    # M6 = (A[0,0] + A[0,1]) * B[1,1]
    # M7 = (A[1,0] - A[0,1]) * (B[0,0] + B[1,0])
    
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    
    # For each 2-column block of B (columns k_base, k_base+1), we apply Strassen.
    # For the last single column, we use 4 schoolbook terms.
    
    # Build the full decomposition:
    # We'll create a list of (U_entries, V_entries, W_entries) for each term.
    # Each entry is (index, coefficient)
    
    terms = []
    
    # 5 blocks of 2 columns each (columns 0-1, 2-3, 4-5, 6-7, 8-9)
    for block in range(5):
        kb = block * 2  # base column index for this block
        
        # M1 = A[0,0] * (B[0,kb] + B[1,kb+1])
        terms.append(((0, 1), [(0, 1), (2, 1)], [(2*kb+0, 1)]))  # U:a00=1, V:b00=1,b11=1, W:C00=1
        # M2 = (A[1,0]+A[1,1]) * B[0,kb]
        terms.append(((2, 1), (3, 1), [(0, 1)], [(2*kb+2, 1)]))  # U:a10=1,a11=1, V:b00=1, W:C10=1
        # M3 = A[0,1] * (B[1,kb] + B[1,kb+1])
        terms.append(((1, 1), [(4, 1), (6, 1)], [(2*kb+1, 1)]))  # U:a01=1, V:b10=1,b11=1, W:C01=1
        # M4 = A[1,0] * (B[0,kb] - B[1,kb])
        terms.append(((2, 1), [(0, 1), (4, -1)], [(2*kb+2, 1)]))  # U:a10=1, V:b00=1,b10=-1, W:C10=1
        # M5 = A[1,1] * (B[0,kb+1] - B[1,kb+1])
        terms.append(((3, 1), [(1, 1), (5, -1)], [(2*kb+0, 1), (2*kb+1, 1)]))  # U:a11=1, V:b01=1,b11=-1, W:C00=1,C01=1
        # M6 = (A[0,0]+A[0,1]) * B[1,kb+1]
        terms.append(((0, 1), (1, 1), [(5, 1)], [(2*kb+3, 1)]))  # U:a00=1,a01=1, V:b11=1, W:C11=1
        # M7 = (A[1,0]-A[0,1]) * (B[0,kb] + B[1,kb])
        terms.append(((2, 1), (1, -1), [(0, 1), (4, 1)], [(2*kb+0, -1), (2*kb+3, 1)]))  # U:a10=1,a01=-1, V:b00=1,b10=1, W:C00=-1,C11=1
    
    # 1 single column (column 10): schoolbook 4 terms
    kb = 10
    # C[0,0] += A[0,0]*B[0,10] + A[0,1]*B[1,10]
    terms.append(((0, 1), (0, 1), [(2*kb+0, 1)]))  # a00*b0
    terms.append(((1, 1), (1, 1), [(2*kb+0, 1)]))  # a01*b1
    # C[1,0] += A[1,0]*B[0,10] + A[1,1]*B[1,10]
    terms.append(((2, 1), (0, 1), [(2*kb+2, 1)]))  # a10*b0
    terms.append(((3, 1), (1, 1), [(2*kb+2, 1)]))  # a11*b1
    
    # Wait, I need to be more careful. Let me redo this properly.
    # The issue is that each "term" in the tensor decomposition is a rank-1 term:
    # T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product)
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-1 decomposition is: for each term c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For schoolbook: for each (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # For Strassen on a 2x2 block:
    # We need to express sum_j A[i,j]*B[j,k] using 7 multiplications.
    # The trick is to pre-compute linear combinations.
    
    # Let me restart with a cleaner approach.
    
    # Actually, let me just build it directly.
    # For a 2x2 * 2x2 block at column offset kb:
    # The relevant B entries are B[j, kb] and B[j, kb+1] for j in {0,1}
    # V indices: j*p + kb = j*11 + kb and j*11 + kb + 1
    
    # Let me redo:
    terms2 = []
    
    for block in range(5):
        kb = block * 2
        # B[j, kb] is at V index j*11 + kb
        # B[j, kb+1] is at V index j*11 + kb + 1
        # C[i, kb] is at W index kb*2 + i
        # C[i, kb+1] is at W index (kb+1)*2 + i
        
        # M1 = A[0,0] * (B[0,kb] + B[1,kb+1])
        # contributes to C[0,kb] with +1, C[1,kb+1] with +1
        terms2.append(([(0, 1)], [(0*11+kb, 1), (1*11+kb+1, 1)], [(kb*2+0, 1), ((kb+1)*2+1, 1)]))
        # M2 = (A[1,0]+A[1,1]) * B[0,kb]
        # contributes to C[1,kb] with +1
        terms2.append(([(2, 1), (3, 1)], [(0*11+kb, 1)], [(kb*2+1, 1)]))
        # M3 = A[0,1] * (B[1,kb] + B[1,kb+1])
        # contributes to C[0,kb+1] with +1
        terms2.append(([(1, 1)], [(1*11+kb, 1), (1*11+kb+1, 1)], [((kb+1)*2+0, 1)]))
        # M4 = A[1,0] * (B[0,kb] - B[1,kb])
        # contributes to C[1,kb] with +1, C[0,kb] with -1
        terms2.append(([(2, 1)], [(0*11+kb, 1), (1*11+kb, -1)], [(kb*2+1, 1), (kb*2+0, -1)]))
        # M5 = A[1,1] * (B[0,kb+1] - B[1,kb+1])
        # contributes to C[0,kb] with +1, C[0,kb+1] with +1
        terms2.append(([(3, 1)], [(0*11+kb+1, 1), (1*11+kb+1, -1)], [(kb*2+0, 1), ((kb+1)*2+0, 1)]))
        # M6 = (A[0,0]+A[0,1]) * B[1,kb+1]
        # contributes to C[1,kb+1] with +1
        terms2.append(([(0, 1), (1, 1)], [(1*11+kb+1, 1)], [((kb+1)*2+1, 1)]))
        # M7 = (A[1,0]-A[0,1]) * (B[0,kb] + B[1,kb])
        # contributes to C[0,kb] with -1, C[1,kb+1] with +1
        terms2.append(([(2, 1), (1, -1)], [(0*11+kb, 1), (1*11+kb, 1)], [(kb*2+0, -1), ((kb+1)*2+1, 1)]))
    
    # Single column kb=10: schoolbook
    kb = 10
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    terms2.append(([(0, 1)], [(0*11+kb, 1)], [(kb*2+0, 1)]))
    terms2.append(([(1, 1)], [(1*11+kb, 1)], [(kb*2+0, 1)]))
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    terms2.append(([(2, 1)], [(0*11+kb, 1)], [(kb*2+1, 1)]))
    terms2.append(([(3, 1)], [(1*11+kb, 1)], [(kb*2+1, 1)]))
    
    # Now fill U, V, W
    rank = len(terms2)
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(22)]
    W = [[0.0] * rank for _ in range(22)]
    
    for c, (u_entries, v_entries, w_entries) in enumerate(terms2):
        for (idx, coeff) in u_entries:
            U[idx][c] += coeff
        for (idx, coeff) in v_entries:
            V[idx][c] += coeff
        for (idx, coeff) in w_entries:
            W[idx][c] += coeff
    
    # Convert to integers where possible
    def to_int_or_half(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        return x
    
    U = [[to_int_or_half(x) for x in row] for row in U]
    V = [[to_int_or_half(x) for x in row] for row in V]
    W = [[to_int_or_half(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
