"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # We exploit the structure: C = A @ B where A is (n x m) = (2 x 4), B is (m x p) = (4 x 5), C is (n x p) = (2 x 5).
    # We decompose the multiplication tensor into a sum of rank-1 terms.
    # The tensor T_{i,j,k} = delta_{i, row of C} delta_{j, col of B} ... wait.
    # T is a 3-tensor with indices (i, j, k) corresponding to C[i, k] = sum_{j} A[i, j] B[j, k].
    # In the decomposition C = sum_t U[:, t] * V[:, t] * W[:, t] (outer product sum),
    # we have C[i, k] = sum_t U[i*m + j? no] ...
    # The interface specifies:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The reconstruction is typically:
    # C[i, k] = sum_{j, l, t} U[i*m + j, t] * V[j*p + l, t] * W[l*n + i, t]
    # Wait, the indices in U, V, W are flattened.
    # U index: r in [0, n*m). Let r = i*m + j. This represents the (i, j) entry of A.
    # V index: s in [0, m*p). Let s = j*p + k. This represents the (j, k) entry of B.
    # W index: t in [0, p*n). Let t = k*n + i. This represents the (k, i) entry of C (transposed indexing? "W[k*n+i,r]")
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # So W index t = k*n + i.
    #
    # The multiplication identity is:
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # In the tensor decomposition framework:
    # C[i, k] = sum_{t} (sum_{j} U[i*m + j, t] * V[j*p + k, t]) * W[k*n + i, t]
    # This implies that for each term t, the contribution to C[i, k] is W[k*n + i, t] * (sum_{j} U[i*m + j, t] * V[j*p + k, t]).
    #
    # We can view this as:
    # For each t, we have vectors u_t (length n*m), v_t (length m*p), w_t (length p*n).
    # u_t is non-zero only at indices corresponding to some A elements.
    # v_t is non-zero only at indices corresponding to some B elements.
    # w_t is non-zero only at indices corresponding to some C elements.
    #
    # Let's try to find a low rank decomposition.
    # The schoolbook rank is n*m*p = 2*4*5 = 40.
    #
    # Strategy: Use the structure of small dimensions.
    # C has 10 elements. A has 8, B has 20.
    #
    # Let's try to decompose C = A @ B.
    # C[0, 0] = A[0, 0]B[0, 0] + A[0, 1]B[1, 0] + A[0, 2]B[2, 0] + A[0, 3]B[3, 0]
    # C[0, 1] = A[0, 0]B[0, 1] + A[0, 1]B[1, 1] + A[0, 2]B[2, 1] + A[0, 3]B[3, 1]
    # ...
    #
    # We can try to factor this.
    # Consider the case n=1. C (1x5) = A (1x4) @ B (4x5). Rank is at most 5 (by splitting columns) or 4 (by splitting rows of B? no).
    # For n=1, m=4, p=5:
    # C_k = sum_j A_j B_jk.
    # This is a bilinear map.
    # We can use Strassen-like tricks?
    #
    # Let's look at the current best score 1.1111 which corresponds to rank 36.
    # The seed program uses rank 36.
    # Can we do better?
    #
    # Let's try to construct a decomposition with rank 35 or less.
    #
    # Alternative approach:
    # We can view the multiplication as a map from A, B to C.
    # C = A B.
    #
    # Let's try to use a known low-rank decomposition for small matrices.
    # For 2x2 @ 2x2, Strassen is 7.
    # For 2x2 @ 2x3?
    # For 2x4 @ 4x5?
    #
    # Let's try to split the problem.
    # C (2x5) = A (2x4) B (4x5).
    # Split B into B1 (4x2) and B2 (4x3).
    # C1 (2x2) = A B1.
    # C2 (2x3) = A B2.
    #
    # Rank(C1):
    # A (2x4) @ B1 (4x2).
    # We can split A into A1 (2x2) and A2 (2x2).
    # C1 = A1 B11 + A2 B12, where B11 is top 2 rows of B1, B12 is bottom 2 rows.
    # B11 is 2x2, B12 is 2x2.
    # A1, A2 are 2x2.
    # C11_part = A1 B11 (2x2). Rank 7 (Strassen).
    # C12_part = A2 B12 (2x2). Rank 7 (Strassen).
    # Total for C1: 14.
    #
    # Rank(C2):
    # A (2x4) @ B2 (4x3).
    # Split B2 into B21 (4x1) and B22 (4x2).
    # C21 (2x1) = A B21.
    # C22 (2x2) = A B22.
    #
    # For C21 (2x1):
    # A (2x4) @ B21 (4x1).
    # Split A into A1 (2x2) and A2 (2x2).
    # B21 into B211 (2x1) and B212 (2x1).
    # C21 = A1 B211 + A2 B212.
    # A1 (2x2) @ B211 (2x1).
    # This is a 2x2 @ 2x1 multiplication.
    # C = [A00 B0 + A01 B1; A10 B0 + A11 B1].
    # Can we do better than 4?
    # M1 = A00 B0 -> C0
    # M2 = A01 B1 -> C0
    # M3 = A10 B0 -> C1
    # M4 = A11 B1 -> C1
    # This is rank 4.
    # Is rank 3 possible?
    # The tensor is [1,0;0,1] for C0 and [0,1;1,0] for C1? No.
    # The support of A is (0,0),(0,1),(1,0),(1,1). B is (0,0),(1,0). C is (0,0),(1,0).
    # It seems rank 4 is minimal for 2x2 @ 2x1.
    # So C211 part: 4.
    # C212 part: 4.
    # Total for C21: 8.
    #
    # For C22 (2x2):
    # A (2x4) @ B22 (4x2).
    # Split A into A1, A2. B22 into B221 (top 2), B222 (bottom 2).
    # C22 = A1 B221 + A2 B222.
    # A1 (2x2) @ B221 (2x2). Rank 7.
    # A2 (2x2) @ B222 (2x2). Rank 7.
    # Total for C22: 14.
    #
    # Total rank: 14 (C1) + 8 (C21) + 14 (C22) = 36.
    # This matches the current score.
    #
    # Can we improve?
    # The 2x2 @ 2x1 case used 4.
    # The 2x2 @ 2x2 case used 7.
    #
    # Is there a better way to handle 2x4 @ 4x5?
    # Maybe combine some terms?
    #
    # Let's check if we can reduce the rank of 2x2 @ 2x2.
    # Strassen is 7. It is known to be optimal for 2x2 over reals? No, 7 is optimal.
    #
    # What about 2x2 @ 2x3?
    # A (2x2) @ B (2x3).
    # C (2x3).
    # Split B into B1 (2x1) and B2 (2x2).
    # C1 = A B1 (rank 4).
    # C2 = A B2 (rank 7).
    # Total 11.
    # Is 10 possible?
    #
    # Let's try to find a decomposition for 2x2 @ 2x3 with rank 10 or less.
    #
    # Actually, let's look at the whole 2x4 @ 4x5.
    #
    # Another idea: Use the fact that n=2.
    #
    # Let's try to code the rank 36 decomposition carefully, ensuring correctness.
    # The previous code had a score of 1.1111, which is 36/40? No, score is seed/value.
    # If seed is 40, and value is 36, score = 40/36 = 1.1111.
    # So the current rank is 36.
    #
    # I will rewrite the rank 36 decomposition with cleaner code to avoid bugs.
    #
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A is 2x2 at a_r0..a_r0+1, a_c0..a_c0+1
        # B is 2x2 at b_r0..b_r0+1, b_c0..b_c0+1
        # C is 2x2 at c_r0..c_r0+1, c_c0..c_c0+1
        
        # U indices: i*m + j
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        # V indices: j*p + k
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        # W indices: k*n + i
        # C(i, k) corresponds to W[k*n + i]
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A is 2x2 at a_r0..a_r0+1, a_c0..a_c0+1
        # B is 2x1 at b_r0..b_r0+1, b_c0
        # C is 2x1 at c_r0..c_r0+1, c_c0
        
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v00 = b_r0 * p + b_c0
        v10 = (b_r0 + 1) * p + b_c0
        
        w00 = c_c0 * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        
        # A00 B00 -> C0
        U[u00][col] = 1; V[v00][col] = 1; W[w00][col] = 1
        col += 1
        # A01 B10 -> C0
        U[u01][col] = 1; V[v10][col] = 1; W[w00][col] = 1
        col += 1
        # A10 B00 -> C1
        U[u10][col] = 1; V[v00][col] = 1; W[w10][col] = 1
        col += 1
        # A11 B10 -> C1
        U[u11][col] = 1; V[v10][col] = 1; W[w10][col] = 1
        col += 1

    # C (2x5) = A (2x4) B (4x5)
    # Split B into B1 (4x2) and B2 (4x3)
    # C1 (2x2) = A B1
    # C2 (2x3) = A B2
    
    # C1:
    # Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3)
    # B1 into B11 (2x2, rows 0,1) and B12 (2x2, rows 2,3)
    # C1 = A1 B11 + A2 B12
    # A1 B11: 2x2 @ 2x2 -> Rank 7
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    # A2 B12: 2x2 @ 2x2 -> Rank 7
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # C2:
    # Split B2 into B21 (4x1, col 2) and B22 (4x2, cols 3,4)
    # C21 (2x1) = A B21
    # C22 (2x2) = A B22
    
    # C21:
    # Split A into A1, A2.
    # B21 into B211 (2x1, rows 0,1) and B212 (2x1, rows 2,3)
    # C21 = A1 B211 + A2 B212
    # A1 B211: 2x2 @ 2x1 -> Rank 4
    add_schoolbook_2x2x1(0, 0, 0, 2, 0, 2)
    # A2 B212: 2x2 @ 2x1 -> Rank 4
    add_schoolbook_2x2x1(0, 2, 2, 2, 0, 2)
    
    # C22:
    # Split A into A1, A2.
    # B22 into B221 (2x2, rows 0,1) and B222 (2x2, rows 2,3)
    # C22 = A1 B221 + A2 B222
    # A1 B221: 2x2 @ 2x2 -> Rank 7
    add_strassen_2x2(0, 0, 0, 3, 0, 3)
    # A2 B222: 2x2 @ 2x2 -> Rank 7
    add_strassen_2x2(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
