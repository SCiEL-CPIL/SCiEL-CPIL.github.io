"""Matrix multiplication 2x2 x 2x15: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 15]

def construct():
    """
    Returns a dict with 'U', 'V', 'W' factor matrices.
    U: shape (n*m, rank) = (4, rank)
    V: shape (m*p, rank) = (30, rank)
    W: shape (p*n, rank) = (30, rank)
    
    The decomposition satisfies:
    sum_c U[i,j,c] * V[j,k,c] = W_component for the (i,k) entry of A*B
    
    Using Strassen's 7-term algorithm for 2x2x2 multiplication applied to each
    of the 15 columns of B.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Strassen's 7 terms for 2x2 matrix multiplication:
    # For matrices A (2x2) and B (2x2):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our case: A is 2x2, B is 2x15
    # We apply Strassen for each column k of B.
    
    # U matrix: rows indexed by (i, j) where i in {0,1} (row of A), j in {0,1} (col of A)
    # Actually, U[i*m+j, c] corresponds to A[i,j] coefficient
    # V[j*p+k, c] corresponds to B[j,k] coefficient  
    # W[k*n+i, c] corresponds to C[i,k] coefficient
    
    # For each column k (0..14), we use 7 Strassen terms.
    # Total rank = 15 * 7 = 105
    
    rank = p * 7  # 105
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 105
    V = [[0] * rank for _ in range(m * p)]  # 30 x 105
    W = [[0] * rank for _ in range(p * n)]  # 30 x 105
    
    # Strassen terms definition:
    # term t has:
    #   U contributions: dict from (i,j) to coefficient
    #   V contributions: dict from (j,k_offset) to coefficient (k_offset relative to current col)
    #   W contributions: dict from (i,k_offset) to coefficient
    
    # For column k, the 7 Strassen multiplications are:
    # M1: (A00+A11)*(B0k+B1k)
    # M2: (A10+A11)*B0k
    # M3: A00*(B1k-B0k)
    # M4: A11*(B1k-B0k)  -- wait, let me re-derive
    
    # Standard Strassen for C = A*B where A, B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our indexing:
    # A[i,j]: i=row of A (0,1), j=col of A (0,1)
    #   A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[j,k]: j=row of B (0,1), k=col of B (0..14)
    #   B[0,k]=e_k, B[1,k]=g_k (using e=f=... for column k)
    #   Actually B[0,k] is the "e" position, B[1,k] is the "g" position
    #   For 2x15, B has rows 0,1 and cols 0..14
    #   So B[0,k] plays role of both e_k and f_k? No.
    #   
    # Wait, B is 2x15. So B[j,k] where j in {0,1}, k in {0..14}.
    # The 2x2 Strassen applies to 2x2 * 2x2. For 2x2 * 2x15, we can think of
    # B as having 15 columns, each column is a 2x1 vector.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For a single column k: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #                         C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is just a 2x2 matrix times a 2x1 vector. We can still use a 7-term
    # decomposition if we embed B's column into a 2x2 by setting one column to 0.
    # But that would waste terms. Actually for 2x2 * 2x1, the rank is 4 (it's
    # just 4 scalar multiplications). So Strassen doesn't help here directly.
    
    # Hmm, let me reconsider. The task says "2x2 matrix by 2x15 matrix".
    # The tensor is M_{n,m,p} = M_{2,2,15}.
    # 
    # The standard schoolbook rank is 2*2*15 = 60.
    # 
    # Can we do better? The known lower bound for M_{2,2,2} is 7 (Strassen).
    # For M_{2,2,15}, we can potentially use Strassen-like decompositions.
    #
    # One approach: view the 2x15 multiplication as 15 separate 2x2 * 2x1 multiplications.
    # Each 2x2 * 2x1 has rank 4 (you can't do better than 4 scalar mults for a
    # 2x2 times 2x1 since it's 4 independent products).
    # So total rank would be 15*4 = 60. Same as schoolbook.
    #
    # But wait - maybe there's structure. Let me think about M_{2,2,p} more carefully.
    # 
    # Actually, I recall that for M_{n,m,p}, if we fix n and m and vary p, the rank
    # is at most n*m*p, and can sometimes be less due to algebraic structure.
    #
    # For 2x2 x 2xp: The rank is known to be at most 2p + something? No...
    # 
    # Let me think differently. We can use the following decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # If we group by rows of A: for row i of A, we compute A[i,:] * B which gives
    # a 1x15 vector. This is a dot product of a 2-vector with each column of B.
    #
    # Alternatively, think of it as: for each pair (i,j), we have A[i,j]*B[j,k] for all k.
    # That's 4 * 15 = 60 terms in schoolbook.
    #
    # Can we share computation across k? The A coefficients are fixed, B varies with k.
    # The products A[i,j]*B[j,k] don't naturally share across k unless we use
    # Strassen-like tricks on the 2x2 part.
    
    # Let me try: for each k, use Strassen's 7 terms but adapted for 2x2 * 2x1.
    # Actually, for 2x2 * 2x1, the minimum rank is 4 (it's an outer product structure).
    # So per column, we need 4 multiplications, giving 15*4 = 60 total.
    
    # Unless... we can use a decomposition that works across all 15 columns simultaneously.
    # 
    # Think of B as a 2x15 matrix. We're computing A*B.
    # 
    # Let me try to construct a rank-53 decomposition as mentioned in the observations.
    # The observation says: "Using Strassen's algorithm for 7 pairs of columns and naive
    # multiplication for the last column reduces the rank from 60 to 53"
    
    # So the idea is: pair up columns of B, and for each pair, use Strassen's 7-term
    # decomposition for 2x2 * 2x2. We have 15 columns, so we can form 7 pairs (14 columns)
    # and have 1 leftover column handled naively (4 multiplications).
    # Total: 7*7 + 4 = 49 + 4 = 53.
    
    # For a pair of columns (k1, k2), we form a 2x2 matrix B_pair = [[B[0,k1], B[0,k2]],
    # [B[1,k1], B[1,k2]]] and compute A * B_pair using Strassen's 7 terms.
    # This gives us C[:, k1] and C[:, k2] from 7 multiplications instead of 8.
    
    # Let me implement this.
    
    rank = 7 * 7 + 4  # 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # Column pairing: columns 0,1 -> pair 0; columns 2,3 -> pair 1; ... columns 12,13 -> pair 6
    # Column 14 -> handled naively
    
    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For column pair (k1, k2):
    # B[0,k1] = e, B[0,k2] = f, B[1,k1] = g, B[1,k2] = h
    # C[0,k1] = C00 = M1 + M4 - M5 + M7
    # C[0,k2] = C01 = M3 + M5
    # C[1,k1] = C10 = M2 + M4
    # C[1,k2] = C11 = M1 - M2 + M3 + M6
    
    # U coefficients for each term (from A = [[a,b],[c,d]] = [[A[0,0],A[0,1]],[A[1,0],A[1,1]]]):
    # M1: a+d -> U[0,0]=1, U[1,1]=1
    # M2: c+d -> U[1,0]=1, U[1,1]=1
    # M3: a   -> U[0,0]=1
    # M4: d   -> U[1,1]=1
    # M5: a+b -> U[0,0]=1, U[0,1]=1
    # M6: c-a -> U[1,0]=1, U[0,0]=-1
    # M7: b-d -> U[0,1]=1, U[1,1]=-1
    
    # V coefficients for each term (from B = [[e,f],[g,h]] = [[B[0,k1],B[0,k2]],[B[1,k1],B[1,k2]]]):
    # M1: e+h -> V[0,k1]=1, V[1,k2]=1
    # M2: e   -> V[0,k1]=1
    # M3: f-h -> V[0,k2]=1, V[1,k1]=-1
    # M4: e-g -> V[0,k1]=1, V[1,k1]=-1
    # M5: h   -> V[1,k2]=1
    # M6: e+f -> V[0,k1]=1, V[0,k2]=1
    # M7: g+h -> V[1,k1]=1, V[1,k2]=1
    
    # W coefficients for each term (contributions to C[0,k1], C[0,k2], C[1,k1], C[1,k2]):
    # M1: C00+=1, C11+=1 -> W[k1*2+0]+=1, W[k2*2+1]+=1  (using W[k*n+i] = W[k*2+i])
    # M2: C10+=1, C11-=1 -> W[k1*2+1]+=1, W[k2*2+1]-=1
    # M3: C01+=1, C11+=1 -> W[k2*2+0]+=1, W[k2*2+1]+=1
    # M4: C00+=1, C10+=1 -> W[k1*2+0]+=1, W[k1*2+1]+=1
    # M5: C00-=1, C01+=1 -> W[k1*2+0]-=1, W[k2*2+0]+=1
    # M6: C11+=1 -> W[k2*2+1]+=1
    # M7: C00+=1 -> W[k1*2+0]+=1
    
    # Let me define the 7 terms more carefully:
    # term: (U_contribs, V_contribs, W_contribs)
    # U_contribs: dict {(i,j): coeff}
    # V_contribs: dict {(j, k_offset): coeff} where k_offset is 0 for k1, 1 for k2
    # W_contribs: dict {(i, k_offset): coeff} where k_offset is 0 for k1, 1 for k2
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2: (c+d)*e
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # M3: a*(f-h)
        (
            {(0,0): 1},
            {(0,1): 1, (1,0): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4: d*(e-g)
        (
            {(1,1): 1},
            {(0,0): 1, (1,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5: (a+b)*h
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # M6: (c-a)*(e+f)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7: (b-d)*(g+h)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    col = 0
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        for t in range(7):
            u_contribs, v_contribs, w_contribs = strassen_terms[t]
            
            # U: rows indexed by i*m+j = i*2+j
            for (i, j), coeff in u_contribs.items():
                U[i * m + j][col] += coeff
            
            # V: rows indexed by j*p+k
            for (j, ko), coeff in v_contribs.items():
                k_actual = k1 if ko == 0 else k2
                V[j * p + k_actual][col] += coeff
            
            # W: rows indexed by k*n+i
            for (i, ko), coeff in w_contribs.items():
                k_actual = k1 if ko == 0 else k2
                W[k_actual * n + i][col] += coeff
            
            col += 1
    
    # Handle column 14 naively (4 multiplications)
    k_last = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + k_last][col] = 1
            W[k_last * n + i][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
