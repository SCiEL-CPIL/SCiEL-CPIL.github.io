DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We aim for a rank-50 decomposition.
    # Structure:
    # Split the inner dimension m=4 into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, we have a 2x2 matrix (rows of A) multiplied by a 2x7 matrix (rows of B).
    # This is effectively two independent (2,2,7) matrix multiplication problems.
    # The rank of (2,2,7) MM is at most 7*2 = 14 (schoolbook) but can be reduced.
    # However, we can use Strassen-like tricks on 2x2 sub-blocks of the output.
    # Specifically, we partition the 7 columns of the output (and corresponding rows of B)
    # into three 2x2 blocks (covering 6 columns) and one 2x1 block (covering the last column).
    # For each 2x2 block of the output (i.e., 2 columns of C), we can use a rank-7 Strassen
    # decomposition for the 2x2x2 MM tensor? No, Strassen is for 2x2 @ 2x2 -> 2x2.
    # Here we are doing 2x2 @ 2x2 -> 2x2 for each block of 2 output columns.
    # Wait, the tensor is (2, 2, 2) for the sub-problem if we group 2 output columns.
    # The rank of (2,2,2) MM is 7 (Strassen).
    # So for each of the three 2x2 blocks of output columns, we use 7 multiplications.
    # For the last 2x1 block of output columns (1 column), we need to compute 2 entries.
    # This is a 2x2 @ 2x1 -> 2x1 problem. The rank is 4 (schoolbook: a*e + b*g, c*e + d*g).
    # Can we do better? The rank of (2,2,1) MM is 4. (It's equivalent to computing 2 dot products of length 2).
    # So total rank per block = 3 * 7 + 4 = 21 + 4 = 25.
    # Since we have two blocks (j=0,1 and j=2,3), total rank = 2 * 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 @ 2x2 -> 2x2 sub-problem.
        i0: start row of A (0 or 1)
        j0: start col of A (0, 2)
        k0: start col of B (0, 2, 4)
        """
        nonlocal col
        
        # A indices (2x2 block starting at row i0, col j0)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices (2x2 block starting at row j0, col k0)
        # Note: B is m x p. The sub-block is rows j0, j0+1 and cols k0, k0+1.
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # C indices (2x2 block starting at row i0, col k0)
        # Note: C is n x p. The sub-block is rows i0, i0+1 and cols k0, k0+1.
        # W index convention: W[k*n + i, r]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # Strassen 7 multiplications:
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        """
        Adds a rank-4 schoolbook decomposition for a 2x2 @ 2x1 -> 2x1 sub-problem.
        i0: start row of A (0 or 1)
        j0: start col of A (0, 2)
        k0: start col of B (6)
        """
        nonlocal col
        
        # A indices (2x2 block starting at row i0, col j0)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices (2x1 block starting at row j0, col k0)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # C indices (2x1 block starting at row i0, col k0)
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C[i0, k0]
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C[i0, k0]
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C[i0+1, k0]
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C[i0+1, k0]
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    # We have two blocks of inner dimension: j_start = 0 and j_start = 2.
    # For each block, we process the output columns k in groups of 2 (Strassen) and 1 (Schoolbook).
    # k ranges from 0 to 6.
    # Groups: (0,1), (2,3), (4,5) use Strassen. (6) uses Schoolbook.
    
    for j_start in [0, 2]:
        # Strassen blocks for k = 0, 2, 4
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        
        # Schoolbook block for k = 6
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
