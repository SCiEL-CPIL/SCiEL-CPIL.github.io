"""Rewritten program: apply Strassen to 2x4x8 blocks to reduce rank from 96 to 88."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # We'll decompose the 4x8 multiplication into two 2x4x8 blocks (rows 0-1 and rows 2-3 of M)
    # Each 2x4x8 block uses 8 Strassen subproblems of 2x2x2, each rank 7 -> 56 per block
    # Total rank = 2 * 56 = 112? No wait.
    
    # Actually, let me think more carefully.
    # Matrix M is 3x4, matrix N is 4x8. Product P is 3x8.
    # P[i,k] = sum_j M[i,j] * N[j,k] for j=0..3
    
    # We can split the 4 columns of M (and 4 rows of N) into two groups of 2:
    # Group A: j=0,1; Group B: j=2,3
    # P[i,k] = sum_{j=0,1} M[i,j]*N[j,k] + sum_{j=2,3} M[i,j]*N[j,k]
    
    # For each group, we have a 3x2 times 2x8 multiplication.
    # Split the 3 rows into... hmm, Strassen works on 2x2.
    
    # Alternative: split the 8 columns of N into four groups of 2.
    # P[i, 2q+r] = sum_j M[i,j] * N[j, 2q+r]
    
    # Let's use a different approach. Split M's 4 columns into two 2-column blocks.
    # For block b (b=0,1), M_b is 3x2, N_b is 2x8.
    # P_b = M_b * N_b is 3x8.
    
    # For each 3x2 * 2x8 product, we can split the 3 rows: row 0, row 1, row 2.
    # P_b[0,k] = M_b[0,0]*N_b[0,k] + M_b[0,1]*N_b[1,k]
    # P_b[1,k] = M_b[1,0]*N_b[0,k] + M_b[1,1]*N_b[1,k]
    # P_b[2,k] = M_b[2,0]*N_b[0,k] + M_b[2,1]*N_b[1,k]
    
    # The first two rows form a 2x2 * 2x8 problem. Split N_b's 8 columns into four 2-col groups.
    # For each 2x2 * 2x2 subproblem, use Strassen (rank 7).
    # So for rows 0,1 of P_b: 4 subproblems * 7 = 28 rank.
    # For row 2 of P_b: it's a 1x2 * 2x8 problem = 8 scalar products, but we can be smarter.
    # Actually row 2: P_b[2,k] = M_b[2,0]*N_b[0,k] + M_b[2,1]*N_b[1,k]
    # Split into four 2-col groups: 4 * (1x2*2x2) = 4 * 4 = 16 rank if schoolbook, 
    # but 1x2*2x2 is just 4 terms, no Strassen benefit. So 16.
    
    # Total per block b: 28 + 16 = 44. Two blocks: 88.
    
    # Let me verify: schoolbook for 3x4*4x8 = 3*4*8 = 96. We save 8. Good.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Strassen formulas for 2x2*2x2:
    # C00 = M00*N00 + M01*N10  ... wait, standard Strassen:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # In our tensor decomposition, each "term" corresponds to one product of a U entry, V entry, W entry.
    # For a Strassen subproblem with inputs A=[[a,b],[c,d]] (2x2) and B=[[e,f],[g,h]] (2x2):
    # We need to express C = A*B using 7 terms.
    # Term 1: p1 = (a+d)(e+h) -> U contributes to a and d, V contributes to e and h, W contributes to C00 and C11
    # This is getting complex. Let me think in terms of the bilinear form.
    
    # The tensor T[i,j,k] = delta_{i,j} for (i,j) in M, (j,k) in N.
    # T[M_idx, N_idx, P_idx] = 1 if P_idx = (k*n+i) and N_idx = (j*p+k) and M_idx = (i*m+j)
    
    # For a Strassen subproblem on rows i1,i2 of M, cols j1,j2 of M (which are rows j1,j2 of N), 
    # and cols k1,k2 of N (which are cols k1,k2 of P):
    # The 8 schoolbook terms are:
    # For each (i in {i1,i2}, j in {j1,j2}, k in {k1,k2}):
    #   M[i*m+j] * N[j*p+k] contributes to P[k*n+i]
    
    # Strassen replaces these 8 with 7. The 7 terms are linear combinations.
    # Let me define the Strassen terms for a specific subproblem.
    
    def add_strassen_subproblem(i1, i2, j1, j2, k1, k2, start_col):
        """Add 7 Strassen terms for the 2x2*2x2 subproblem.
        M rows i1,i2; M cols j1,j2 (= N rows j1,j2); N cols k1,k2; P cols k1,k2.
        Returns number of columns used (7)."""
        nonlocal col
        # The 8 schoolbook terms:
        # (i1,j1,k1): M[i1*m+j1]*N[j1*p+k1] -> P[k1*n+i1]
        # (i1,j1,k2): M[i1*m+j1]*N[j1*p+k2] -> P[k2*n+i1]
        # (i1,j2,k1): M[i1*m+j2]*N[j2*p+k1] -> P[k1*n+i1]
        # (i1,j2,k2): M[i1*m+j2]*N[j2*p+k2] -> P[k2*n+i1]
        # (i2,j1,k1): M[i2*m+j1]*N[j1*p+k1] -> P[k1*n+i2]
        # (i2,j1,k2): M[i2*m+j1]*N[j1*p+k2] -> P[k2*n+i2]
        # (i2,j2,k1): M[i2*m+j2]*N[j2*p+k1] -> P[k1*n+i2]
        # (i2,j2,k2): M[i2*m+j2]*N[j2*p+k2] -> P[k2*n+i2]
        
        # Strassen terms:
        # p1 = (M[i1*m+j1] + M[i2*m+j2]) * (N[j1*p+k1] + N[j2*p+k2])
        # p2 = (M[i2*m+j1] + M[i2*m+j2]) * N[j1*p+k1]
        # p3 = M[i1*m+j1] * (N[j2*p+k2] - N[j2*p+k1])
        # p4 = M[i2*m+j2] * (N[j2*p+k1] - N[j1*p+k1])
        # p5 = (M[i1*m+j1] + M[i1*m+j2]) * N[j2*p+k2]
        # p6 = (M[i2*m+j1] - M[i1*m+j1]) * (N[j1*p+k1] + N[j2*p+k2])
        # p7 = (M[i1*m+j2] - M[i2*m+j2]) * (N[j2*p+k1] + N[j2*p+k2])
        
        # Outputs:
        # P[k1*n+i1] = p1 + p4 - p5 + p7
        # P[k2*n+i1] = p3 + p5
        # P[k1*n+i2] = p2 + p4
        # P[k2*n+i2] = p1 - p2 + p3 + p6
        
        # Term p1:
        U[i1*m+j1][start_col] += 1
        U[i2*m+j2][start_col] += 1
        V[j1*p+k1][start_col] += 1
        V[j2*p+k2][start_col] += 1
        W[k1*n+i1][start_col] += 1
        W[k2*n+i2][start_col] += 1
        col += 1
        
        # Term p2:
        U[i2*m+j1][start_col] += 1
        U[i2*m+j2][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k1*n+i2][start_col] += 1
        col += 1
        
        # Term p3:
        U[i1*m+j1][start_col] += 1
        V[j2*p+k2][start_col] += 1
        V[j2*p+k1][start_col] += -1
        W[k2*n+i1][start_col] += 1
        col += 1
        
        # Term p4:
        U[i2*m+j2][start_col] += 1
        V[j2*p+k1][start_col] += 1
        V[j1*p+k1][start_col] += -1
        W[k1*n+i1][start_col] += 1
        W[k1*n+i2][start_col] += 1
        col += 1
        
        # Term p5:
        U[i1*m+j1][start_col] += 1
        U[i1*m+j2][start_col] += 1
        V[j2*p+k2][start_col] += 1
        W[k1*n+i1][start_col] += -1
        W[k2*n+i1][start_col] += 1
        col += 1
        
        # Term p6:
        U[i2*m+j1][start_col] += 1
        U[i1*m+j1][start_col] += -1
        V[j1*p+k1][start_col] += 1
        V[j2*p+k2][start_col] += 1
        W[k2*n+i2][start_col] += 1
        col += 1
        
        # Term p7:
        U[i1*m+j2][start_col] += 1
        U[i2*m+j2][start_col] += -1
        V[j2*p+k1][start_col] += 1
        V[j2*p+k2][start_col] += 1
        W[k1*n+i1][start_col] += 1
        col += 1
        
        return 7
    
    # Now build the full decomposition:
    # Block b=0: M cols j=0,1; N rows j=0,1
    # Block b=1: M cols j=2,3; N rows j=2,3
    
    for b in range(2):
        j1 = 2 * b
        j2 = 2 * b + 1
        
        # Rows 0,1 of P: use Strassen on 2x2*2x2 subproblems for each pair of N cols
        for k_group in range(4):
            k1 = 2 * k_group
            k2 = 2 * k_group + 1
            add_strassen_subproblem(0, 1, j1, j2, k1, k2, col)
        
        # Row 2 of P: schoolbook (4 terms per 2-col group, 4 groups = 16 terms)
        for k_group in range(4):
            k1 = 2 * k_group
            k2 = 2 * k_group + 1
            for j in [j1, j2]:
                for k in [k1, k2]:
                    U[2*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+2][col] = 1
                    col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
