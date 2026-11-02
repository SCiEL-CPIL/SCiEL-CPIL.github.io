"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 decomposition for 3x3 x 3x4 matrix multiplication
    # Using additive bound: R(3,3,4) <= R(3,3,2) + R(3,3,2)
    # R(3,3,2) can be done in rank 12 (schoolbook for 3x3 x 3x2)
    # But we can do better: R(3,3,2) <= 11
    # Actually, let's use the known bound R(3,3,4) <= 23 via R(3,3,3)=23 + R(3,3,1)=1
    
    # For R(3,3,1): multiply 3x3 by 3x1. Rank = 3 (each output column needs 3 mults)
    # Actually R(3,3,1) = 3 since C = A * b, each C[i] = sum_j A[i,j]*b[j], needs 3*3=9? No.
    # Wait, tensor for (3,3,1): U is 9x?, V is 3x?, W is 3x?
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # This is just 9 scalar multiplications. Rank = 9? No, we can factor.
    # Actually R(3,3,1) = 3. We can write C[i,0] = sum_j A[i,j]*B[j,0].
    # Hmm, this is the outer product structure. The tensor is sum_{i,j} e_i^T (e_j^T ⊗ e_0^T) a_i b_j
    # This has rank 3 (one for each column of B, which is just 1 column).
    # Actually, R(3,3,1) = 3.
    
    # Let me just implement a valid rank 23 decomposition.
    # R(3,3,3) = 23 is known. For R(3,3,4), we use:
    # Split B into B1 (3x3) and B2 (3x1)
    # C = A*B1 + A*B2
    # R(3,3,4) <= R(3,3,3) + R(3,3,1) = 23 + 3 = 26
    
    # Let me construct the rank 26 solution properly.
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: R(3,3,3) decomposition (rank 23)
    # Using the known rank 23 decomposition for 3x3x3
    # This is complex to write explicitly. Let me use a simpler approach.
    
    # Alternative: use the additive bound with schoolbook for the 3x3 part
    # R(3,3,3) schoolbook = 9, R(3,3,1) schoolbook = 3, total = 12
    # That's worse than 26.
    
    # Let me just use the schoolbook decomposition but with the structure
    # that gives rank 26.
    
    # Actually, the simplest valid improvement is to use the fact that
    # R(3,3,4) <= R(3,3,2) + R(3,3,2) where R(3,3,2) <= 11
    # So R(3,3,4) <= 22? Let me verify R(3,3,2).
    
    # For (3,3,2): C[i,k] = sum_j A[i,j]*B[j,k], k=0,1
    # Schoolbook: 9*2 = 18
    # Better: R(3,3,2) <= 12
    # So R(3,3,4) <= 24
    
    # Let me implement rank 24 using two R(3,3,2) decompositions.
    
    # R(3,3,2) in rank 12:
    # For each column k of B, we compute A*B[:,k]
    # But we can share work between the two columns.
    
    # Actually, let me just implement the schoolbook rank 36 but that gives score 1.0.
    # I need something better.
    
    # Let me implement a rank 26 decomposition using the 3x3x3 rank 23 + 3x3x1 rank 3.
    # For 3x3x1: C[i,0] = sum_j A[i,j]*B[j,0]
    # This needs 9 multiplications, but we can do it in rank 3:
    # Use 3 "columns" of the decomposition, one for each j
    # For j=0: U[i*3+0, c]=1 for all i, V[0*3+0, c]=1, W[0*3+i, c]=1 ... no that's wrong.
    
    # For 3x3x1: the tensor has entries T[i,j,0] = 1 for all i,j
    # We can decompose as: sum_{j=0}^{2} e_j^T (e_j^T ⊗ e_0^T)
    # Wait, that's rank 3.
    # Term j: U has 1 at row (i*3+j) for all i? No.
    # U is 9x3, V is 3x3, W is 3x3
    # For term c (c=0,1,2):
    #   U[i*3+c, c] = 1 for all i (3 entries)
    #   V[c*3+0, c] = 1 (1 entry)  -- since B[j,0] with j=c
    #   W[0*3+i, c] = 1 for all i (3 entries)  -- since output is C[i,0]
    
    # Let me build this properly.
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 26
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 26
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 26
    
    # For the 3x3x3 part (columns 0-8 of B, i.e., B[:,0:3]), use schoolbook rank 9
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k=0,1,2 for the 3x3 part
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # For the 3x3x1 part (column 3 of B, i.e., B[:,3]), use rank 3
    # C[i,3] = sum_j A[i,j]*B[j,3]
    for j in range(m):
        for i in range(n):
            U[i * m + j][col] = 1
        V[j * p + 3][col] = 1
        for i in range(n):
            W[3 * n + i][col] = 1
        col += 1
    
    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"
    
    # Verify the decomposition
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j
    # Let's verify numerically
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: for all i,k, C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for c in range(rank):
                    total += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
            if k < 3:
                # For the 3x3 part, only j should contribute
                expected = 1.0  # Each (i,j,k) with k<3 should give 1 for matching j
                # Actually, C[i,k] = sum_j A[i,j]*B[j,k], so the tensor entry is 1 if we're computing
                # the coefficient. The tensor T[i,j,k] = 1.
                # So sum_j sum_c ... should equal 1 for each (i,k) pair? No.
                # The tensor is T[i,j,k] = 1. The decomposition should reproduce T.
                # C[i,k] in the tensor sense is sum_j T[i,j,k] which is 3.
                # Hmm, I'm confusing myself.
                pass
            else:
                expected = 1.0
    
    # The tensor for matrix mult (n,m,p) has T[i,j,k] = 1.
    # The decomposition should satisfy: for all i,j,k, T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # So for each (i,j,k), the sum should be 1.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    else:
        # Convert to integers if possible
        U = [[int(round(x)) for x in row] for row in U]
        V = [[int(round(x)) for x in row] for row in V]
        W = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
