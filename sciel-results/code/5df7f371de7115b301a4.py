"""Rewritten program exploiting commutative property and block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 3, 10]

def construct():
    # We are multiplying a (2,3) matrix A by a (3,10) matrix B.
    # Output C is (2,10).
    # The task specifies W follows trace convention W[k*n+i, r] where n=2, p=10.
    # So W has shape (p*n, rank) = (10*2, rank) = (20, rank).
    # U has shape (n*m, rank) = (2*3, rank) = (6, rank).
    # V has shape (m*p, rank) = (3*10, rank) = (30, rank).
    
    # Schoolbook rank is n*m*p = 2*3*10 = 60.
    
    # Directive suggests: "transpose the problem to M_{10,3,2}"
    # Matrix multiplication (n,m) x (m,p) is isomorphic to (p,m) x (m,n) via transposition.
    # If C = A B, then C^T = B^T A^T.
    # B^T is (10,3), A^T is (3,2). So we multiply (10,3) by (3,2).
    # This is M_{10,3,2}.
    
    # The directive says: "apply a block-diagonal decomposition where the middle dimension m=3 is split into a 2x1 partition"
    # So we split the middle dimension of size 3 into parts of size 2 and 1.
    # In M_{10,3,2}, the middle dimension is 3.
    # Split into 2 and 1.
    # This gives two blocks:
    # Block 1: (10,2) x (2,2) -> (10,2). Rank of (10,2)x(2,2) is 10*2*2 = 40 schoolbook.
    # Block 2: (10,1) x (1,2) -> (10,2). Rank of (10,1)x(1,2) is 10*1*2 = 20 schoolbook.
    # Total schoolbook for this split: 40 + 20 = 60. Same as before.
    
    # But the directive says: "allowing the 2x2 sub-block to be solved with rank-7 Strassen"
    # Wait, the 2x2 sub-block refers to the (2,2) part?
    # In the block (10,2) x (2,2), we can view this as 10 independent multiplications of (1,2) x (2,2)? No.
    # Actually, (10,2) x (2,2) = (10,2). This is 10 rows, each row is (1,2) x (2,2) = (1,2).
    # So it's 10 independent (1,2)x(2,2) multiplications? No, (1,2)x(2,2) is just a vector-matrix product, which is trivial.
    
    # Let's re-read: "the 2x2 sub-block to be solved with rank-7 Strassen".
    # Strassen is for (2,2)x(2,2). Rank 7.
    # So maybe we should group the dimensions differently.
    
    # Let's look at the original problem: (2,3) x (3,10).
    # Split the middle dimension 3 into 2+1.
    # A = [A1, A2] where A1 is (2,2), A2 is (2,1).
    # B = [B1; B2] where B1 is (2,10), B2 is (1,10).
    # C = A1 B1 + A2 B2.
    # A1 B1 is (2,2)x(2,10) = (2,10). This is 10 independent (2,2)x(2,1) multiplications? No.
    # (2,2)x(2,10) can be viewed as 10 independent (2,2)x(2,1) multiplications.
    # Each (2,2)x(2,1) is a matrix-vector product. The tensor for (2,2)x(2,1) has rank 4 (schoolbook).
    # Can we do better? The rank of (2,2)x(2,1) is 4. It's equivalent to (2,1)x(2,2) transposed, which is also rank 4.
    # Strassen doesn't help here directly because one dimension is 1.
    
    # Alternative interpretation:
    # Maybe we split the output dimensions?
    # Or maybe the "2x2 sub-block" refers to a (2,2)x(2,2) multiplication hidden in the structure.
    
    # Let's reconsider the transposed problem: (10,3) x (3,2).
    # Split middle dimension 3 into 2+1.
    # (10,3) = [(10,2), (10,1)].
    # (3,2) = [(2,2); (1,2)].
    # Product = (10,2)(2,2) + (10,1)(1,2).
    # First term: (10,2)(2,2) = (10,2). This is 10 independent (1,2)(2,2) = (1,2) multiplications. Rank 4 each. Total 40.
    # Second term: (10,1)(1,2) = (10,2). This is 10 independent (1,1)(1,2) = (1,2) multiplications. Rank 2 each. Total 20.
    # Total 60.
    
    # The directive claims "yielding a total rank of 55".
    # 55 = 7 + 48? Or 7*? 
    # If we use Strassen for a (2,2)x(2,2) block, rank is 7.
    # Where is a (2,2)x(2,2) block?
    # In (10,3)x(3,2), if we split 10 into 5+5? No.
    # If we split 10 into groups of 2? 10 = 5*2.
    # So we have 5 independent (2,3)x(3,2) multiplications?
    # (2,3)x(3,2) = (2,2).
    # For each (2,3)x(3,2):
    # Split middle 3 into 2+1.
    # (2,2)(2,2) + (2,1)(1,2).
    # (2,2)(2,2) can be done with Strassen, rank 7.
    # (2,1)(1,2) is rank 4 (schoolbook: 2*1*2=4).
    # So each (2,3)x(3,2) block has rank 7 + 4 = 11.
    # Total rank = 5 * 11 = 55.
    
    # This matches the directive!
    # So the strategy is:
    # 1. Transpose the problem to (10,3)x(3,2).
    # 2. Split the 10 dimension into 5 groups of 2.
    # 3. For each group, solve (2,3)x(3,2) by splitting the middle dimension 3 into 2+1.
    # 4. The (2,2)x(2,2) part uses Strassen (rank 7).
    # 5. The (2,1)x(1,2) part uses schoolbook (rank 4).
    # 6. Total rank = 5 * (7 + 4) = 55.
    
    # Now we need to construct U, V, W for the original problem (2,3)x(3,10).
    # Original: A (2,3), B (3,10), C (2,10).
    # Transposed: A^T (3,2), B^T (10,3), C^T (10,2).
    # Wait, the tensor decomposition is for the multiplication map.
    # If we find a decomposition for (10,3)x(3,2), we can map it back to (2,3)x(3,10).
    # The multiplication tensor T_{n,m,p} is isomorphic to T_{p,m,n} via permutation.
    # Specifically, T_{2,3,10} is isomorphic to T_{10,3,2}.
    # The isomorphism maps indices (i,j,k) in T_{2,3,10} to (k,j,i) in T_{10,3,2}.
    # Here i in 0..1 (n=2), j in 0..2 (m=3), k in 0..9 (p=10).
    # In T_{10,3,2}, indices are (i',j',k') with i' in 0..9 (n'=10), j' in 0..2 (m'=3), k' in 0..1 (p'=2).
    # Mapping: i' = k, j' = j, k' = i.
    
    # So if we have U', V', W' for T_{10,3,2} with rank R:
    # U' shape (n'*m', R) = (10*3, R) = (30, R). U'[i'*m'+j', c].
    # V' shape (m'*p', R) = (3*2, R) = (6, R). V'[j'*p'+k', c].
    # W' shape (p'*n', R) = (2*10, R) = (20, R). W'[k'*n'+i', c].
    
    # We need U, V, W for T_{2,3,10} with rank R:
    # U shape (n*m, R) = (2*3, R) = (6, R). U[i*m+j, c].
    # V shape (m*p, R) = (3*10, R) = (30, R). V[j*p+k, c].
    # W shape (p*n, R) = (10*2, R) = (20, R). W[k*n+i, c].
    
    # The relation is:
    # U[i*m+j, c] = V'[j*m'+k', c] ? No.
    # Let's derive from the tensor definition.
    # T_{n,m,p}[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # T_{n',m',p'}[i',j',k'] = sum_c U'[i'*m'+j', c] V'[j'*p'+k', c] W'[k'*n'+i', c].
    # With i'=k, j'=j, k'=i.
    # T_{10,3,2}[k,j,i] = sum_c U'[k*3+j, c] V'[j*2+i, c] W'[i*10+k, c].
    # We want T_{2,3,10}[i,j,k] = T_{10,3,2}[k,j,i].
    # So sum_c U[i*3+j, c] V[j*10+k, c] W[k*2+i, c] = sum_c U'[k*3+j, c] V'[j*2+i, c] W'[i*10+k, c].
    # This suggests:
    # U[i*3+j, c] = V'[j*2+i, c]
    # V[j*10+k, c] = U'[k*3+j, c]
    # W[k*2+i, c] = W'[i*10+k, c]
    
    # Let's check dimensions:
    # U[i*3+j, c]: i in 0..1, j in 0..2. Index range 0..5.
    # V'[j*2+i, c]: j in 0..2, i in 0..1. Index range 0..5. OK.
    # V[j*10+k, c]: j in 0..2, k in 0..9. Index range 0..29.
    # U'[k*3+j, c]: k in 0..9, j in 0..2. Index range 0..29. OK.
    # W[k*2+i, c]: k in 0..9, i in 0..1. Index range 0..19.
    # W'[i*10+k, c]: i in 0..1, k in 0..9. Index range 0..19. OK.
    
    # So the mapping is:
    # U[i*m+j, c] = V'[j*p'+k', c] where p'=2, k'=i. So U[i*3+j, c] = V'[j*2+i, c].
    # V[j*p+k, c] = U'[i'*m'+j', c] where i'=k, m'=3, j'=j. So V[j*10+k, c] = U'[k*3+j, c].
    # W[k*n+i, c] = W'[k'*n'+i', c] where k'=i, n'=10, i'=k. So W[k*2+i, c] = W'[i*10+k, c].
    
    # Now we construct U', V', W' for T_{10,3,2} with rank 55.
    # n'=10, m'=3, p'=2.
    # Split n'=10 into 5 groups of 2. Let g be group index 0..4, and r be row within group 0..1.
    # i' = 2*g + r.
    # For each group g, we solve (2,3)x(3,2).
    # Split m'=3 into 2+1. Let s be sub-index 0..2.
    # s=0,1 are the first 2, s=2 is the last 1.
    # The multiplication is sum_s A[g,r,s] B[g,s,k'].
    # Where A[g,r,s] is the element of A' at (i',s) = (2g+r, s).
    # B[g,s,k'] is the element of B' at (s,k').
    
    # For each g:
    # Part 1: s=0,1. A1[g] is (2,2), B1[g] is (2,2). Product is (2,2).
    # Use Strassen for (2,2)x(2,2). Rank 7.
    # Part 2: s=2. A2[g] is (2,1), B2[g] is (1,2). Product is (2,2).
    # Use schoolbook for (2,1)x(1,2). Rank 4.
    # Total per group: 7 + 4 = 11.
    # Total rank: 5 * 11 = 55.
    
    # Strassen decomposition for (2,2)x(2,2):
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]].
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In terms of tensor factors:
    # For each multiplication m (1..7), we have a linear combination of A entries and B entries.
    # U' contributes to A', V' contributes to B', W' contributes to C'.
    # A' has shape (10,3). B' has shape (3,2). C' has shape (10,2).
    # For group g, A' rows are 2g, 2g+1. B' rows are 0,1,2. C' rows are 2g, 2g+1.
    
    # Let's define the factors for one group (g fixed).
    # Strassen part:
    # For each of the 7 products, we have:
    # U'_row, U'_col, V'_row, V'_col, W'_row, W'_col.
    # Actually, U' is indexed by (i', s). V' is indexed by (s, k'). W' is indexed by (k', i').
    # Wait, earlier I said U'[i'*m'+j', c] = U'[i'*3+s, c].
    # V'[j'*p'+k', c] = V'[s*2+k', c].
    # W'[k'*n'+i', c] = W'[k'*10+i', c].
    
    # For Strassen:
    # M1 = (a11+a22)(b11+b22).
    # a11 is A'[2g, 0], a22 is A'[2g+1, 1].
    # b11 is B'[0, 0], b22 is B'[1, 1].
    # So U'[2g*3+0, c] += 1, U'[(2g+1)*3+1, c] += 1.
    # V'[0*2+0, c] += 1, V'[1*2+1, c] += 1.
    # W'[0*10+2g, c] += 1 (for C11), W'[0*10+2g+1, c] += 1 (for C12), etc.
    # But W' depends on the output position.
    # C11 is at C'[2g, 0]. So W'[0*10+2g, c] += 1.
    # C12 is at C'[2g, 1]. So W'[1*10+2g, c] += 1.
    # C21 is at C'[2g+1, 0]. So W'[0*10+2g+1, c] += 1.
    # C22 is at C'[2g+1, 1]. So W'[1*10+2g+1, c] += 1.
    
    # Let's write a helper to add a Strassen block for group g.
    # And a helper to add a schoolbook block for (2,1)x(1,2) for group g.
    
    # Schoolbook for (2,1)x(1,2):
    # A2 = [a1; a2] (2x1), B2 = [b1, b2] (1x2).
    # C = [[a1*b1, a1*b2], [a2*b1, a2*b2]].
    # 4 products:
    # P1 = a1*b1. U'[2g*3+2, c] += 1, V'[2*2+0, c] += 1, W'[0*10+2g, c] += 1.
    # P2 = a1*b2. U'[2g*3+2, c] += 1, V'[2*2+1, c] += 1, W'[1*10+2g, c] += 1.
    # P3 = a2*b1. U'[(2g+1)*3+2, c] += 1, V'[2*2+0, c] += 1, W'[0*10+2g+1, c] += 1.
    # P4 = a2*b2. U'[(2g+1)*3+2, c] += 1, V'[2*2+1, c] += 1, W'[1*10+2g+1, c] += 1.
    
    # Note: s=2 corresponds to the third row of A' and third column of B'.
    # A'[i', 2] and B'[2, k'].
    # So U' index is i'*3+2. V' index is 2*2+k' = 4+k'.
    
    # Let's implement this.
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    n_prime, m_prime, p_prime = p, m, n  # 10, 3, 2
    
    rank = 55
    U_prime = [[0] * rank for _ in range(n_prime * m_prime)]  # 30 x 55
    V_prime = [[0] * rank for _ in range(m_prime * p_prime)]  # 6 x 55
    W_prime = [[0] * rank for _ in range(p_prime * n_prime)]  # 20 x 55
    
    col = 0
    num_groups = n_prime // 2  # 5
    
    for g in range(num_groups):
        # Strassen part: 7 columns
        # M1
        U_prime[2*g*3 + 0][col] += 1
        U_prime[(2*g+1)*3 + 1][col] += 1
        V_prime[0*2 + 0][col] += 1
        V_prime[1*2 + 1][col] += 1
        W_prime[0*10 + 2*g][col] += 1
        W_prime[1*10 + 2*g][col] += 1
        W_prime[0*10 + 2*g+1][col] += 1
        W_prime[1*10 + 2*g+1][col] += 1
        col += 1
        
        # M2
        U_prime[(2*g+1)*3 + 0][col] += 1
        U_prime[(2*g+1)*3 + 1][col] += 1
        V_prime[0*2 + 0][col] += 1
        W_prime[0*10 + 2*g+1][col] += 1
        W_prime[1*10 + 2*g+1][col] += 1
        col += 1
        
        # M3
        U_prime[2*g*3 + 0][col] += 1
        V_prime[1*2 + 0][col] += 1
        V_prime[1*2 + 1][col] -= 1
        W_prime[1*10 + 2*g][col] += 1
        W_prime[1*10 + 2*g+1][col] += 1
        col += 1
        
        # M4
        U_prime[(2*g+1)*3 + 1][col] += 1
        V_prime[1*2 + 1][col] += 1
        V_prime[0*2 + 0][col] -= 1
        W_prime[0*10 + 2*g][col] += 1
        W_prime[0*10 + 2*g+1][col] += 1
        col += 1
        
        # M5
        U_prime[2*g*3 + 0][col] += 1
        U_prime[2*g*3 + 1][col] += 1
        V_prime[1*2 + 1][col] += 1
        W_prime[0*10 + 2*g][col] -= 1
        W_prime[1*10 + 2*g][col] += 1
        col += 1
        
        # M6
        U_prime[(2*g+1)*3 + 0][col] += 1
        U_prime[2*g*3 + 0][col] -= 1
        V_prime[0*2 + 0][col] += 1
        V_prime[0*2 + 1][col] += 1
        W_prime[1*10 + 2*g+1][col] += 1
        col += 1
        
        # M7
        U_prime[2*g*3 + 1][col] += 1
        U_prime[(2*g+1)*3 + 1][col] -= 1
        V_prime[1*2 + 1][col] += 1
        V_prime[1*2 + 0][col] += 1
        W_prime[0*10 + 2*g][col] += 1
        col += 1
        
        # Schoolbook part for (2,1)x(1,2): 4 columns
        # s=2
        # P1: a1*b1
        U_prime[2*g*3 + 2][col] += 1
        V_prime[2*2 + 0][col] += 1
        W_prime[0*10 + 2*g][col] += 1
        col += 1
        
        # P2: a1*b2
        U_prime[2*g*3 + 2][col] += 1
        V_prime[2*2 + 1][col] += 1
        W_prime[1*10 + 2*g][col] += 1
        col += 1
        
        # P3: a2*b1
        U_prime[(2*g+1)*3 + 2][col] += 1
        V_prime[2*2 + 0][col] += 1
        W_prime[0*10 + 2*g+1][col] += 1
        col += 1
        
        # P4: a2*b2
        U_prime[(2*g+1)*3 + 2][col] += 1
        V_prime[2*2 + 1][col] += 1
        W_prime[1*10 + 2*g+1][col] += 1
        col += 1
    
    # Now map back to U, V, W for original problem.
    # U[i*3+j, c] = V_prime[j*2+i, c]
    # V[j*10+k, c] = U_prime[k*3+j, c]
    # W[k*2+i, c] = W_prime[i*10+k, c]
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                U[i*m + j][c] = V_prime[j*p_prime + i][c]
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V[j*p + k][c] = U_prime[k*m_prime + j][c]
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k*n + i][c] = W_prime[i*n_prime + k][c]
    
    return {"U": U, "V": V, "W": W}
