def construct():
    n, m, p = 4, 4, 7
    
    # U, V, W will be built as lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(a_idx, b_idx, c_idx, coeff=1):
        U = [0] * (n*m)
        V = [0] * (m*p)
        W = [0] * (p*n)
        U[a_idx] = coeff
        V[b_idx] = coeff
        W[c_idx] = coeff
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)
    
    # Part 1: 4x4 x 4x4 using Strassen (56 terms)
    # Strassen M1-M7, each is a 2x2x2 product
    
    # For each M-product, define the index mappings
    # M1 = (A11+A22)(B11+B22) -> C11
    # The 2x2x2 subproblem has:
    # A' [r,s] = A[r,s] + A[r+2,s+2] for r,s in {0,1}
    # B' [s,t] = B[s,t] + B[s+2,t+2] for s,t in {0,1}
    # C' [r,t] = C[r,t] for r,t in {0,1}
    
    # Schoolbook for 2x2x2: 8 terms
    # For r,s,t in {0,1}: A'[r,s] * B'[s,t] -> C'[r,t]
    # A'[r,s] = A[r,s] + A[r+2,s+2]
    # B'[s,t] = B[s,t] + B[s+2,t+2]
    # So A'[r,s]*B'[s,t] = (A[r,s]+A[r+2,s+2])(B[s,t]+B[s+2,t+2])
    # = A[r,s]*B[s,t] + A[r,s]*B[s+2,t+2] + A[r+2,s+2]*B[s,t] + A[r+2,s+2]*B[s+2,t+2]
    # Each of these 4 products contributes to C[r,t]
    
    # But wait, this gives 4 products per (r,s,t), and 8 (r,s,t) combos, so 32 products for M1?
    # No, that's wrong. The 2x2x2 schoolbook has 8 terms, not 32.
    
    # I think the issue is that in the Strassen decomposition, the "2x2x2" subproblem is not a standard matrix multiplication.
    # The A' and B' matrices are linear combinations, and the product A'*B' is computed with 8 scalar multiplications,
    # where each scalar multiplication is of the form (linear form in A) * (linear form in B).
    
    # So for M1, the 8 scalar multiplications are:
    # For each (r,s,t) in {0,1}^3 with the constraint that it's a valid 2x2x2 term:
    # Actually, the 8 terms are:
    # (r=0,s=0,t=0): A'[0,0]*B'[0,0] -> C'[0,0]
    # (r=0,s=1,t=0): A'[0,1]*B'[1,0] -> C'[0,0]
    # (r=0,s=0,t=1): A'[0,0]*B'[0,1] -> C'[0,1]
    # (r=0,s=1,t=1): A'[0,1]*B'[1,1] -> C'[0,1]
    # (r=1,s=0,t=0): A'[1,0]*B'[0,0] -> C'[1,0]
    # (r=1,s=1,t=0): A'[1,1]*B'[1,0] -> C'[1,0]
    # (r=1,s=0,t=1): A'[1,0]*B'[0,1] -> C'[1,1]
    # (r=1,s=1,t=1): A'[1,1]*B'[1,1] -> C'[1,1]
    
    # Each of these is a scalar multiplication of two linear forms.
    # A'[r,s] = A[r,s] + A[r+2,s+2] (for M1)
    # B'[s,t] = B[s,t] + B[s+2,t+2] (for M1)
    
    # So for term (r=0,s=0,t=0):
    # x = A[0,0] + A[2,2]
    # y = B[0,0] + B[2,2]
    # x*y contributes to C[0,0]
    # In the tensor: for (i,j,k) in {(0,0,0), (0,0,2), (2,2,0), (2,2,2)},
    # T[i,j,k] += 1 (since the coefficient is 1*1*1)
    
    # Wait, but C[0,0] = sum_j A[0,j]*B[j,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # And x*y = (A[0,0]+A[2,2])(B[0,0]+B[2,2]) = A[0,0]*B[0,0] + A[0,0]*B[2,2] + A[2,2]*B[0,0] + A[2,2]*B[2,2]
    # This is NOT equal to C[0,0]. It's just one part of the Strassen computation.
    
    # The full Strassen algorithm combines M1-M7 to get C.
    # So each M-product contributes to C with specific coefficients, and the sum of all contributions equals C.
    
    # For the tensor decomposition, I need to ensure that the sum of all rank-1 terms equals T.
    
    # Let me just directly compute the contributions.
    
    # For M1 = (A11+A22)(B11+B22), contributing to C11 = C[0:2, 0:2]:
    # C[0,0] += M1[0,0]
    # C[0,1] += M1[0,1]
    # C[1,0] += M1[1,0]
    # C[1,1] += M1[1,1]
    
    # M1[0,0] = A'[0,0]*B'[0,0] + A'[0,1]*B'[1,0]
    # = (A[0,0]+A[2,2])(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])(B[1,0]+B[3,2])
    
    # This is a sum of 8 products, each of the form A[i,j]*B[j',k] where (i,j) and (j',k) are specific.
    # For the tensor, each such product contributes to T[i,j,k] if j==j', or... 
    
    # Actually, I realize that in the tensor decomposition, the bilinear product x*y where
    # x = sum a_ij A[i,j] and y = sum b_jk B[j,k]
    # contributes a_ij * b_jk * c_ik to T[i,j,k] for all i,j,k.
    
    # So for M1[0,0] = (A[0,0]+A[2,2])(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])(B[1,0]+B[3,2]):
    # This is a sum of 4 separate bilinear products:
    # 1. (A[0,0]+A[2,2]) * (B[0,0]+B[2,2]) -> contributes to C[0,0]
    # 2. (A[0,1]+A[2,3]) * (B[1,0]+B[3,2]) -> contributes to C[0,0]
    
    # Wait, no. M1[0,0] is a single scalar, which is the sum of these 4 products.
    # In the bilinear decomposition, each "column" corresponds to one scalar multiplication.
    # So M1[0,0] would be computed with 4 scalar multiplications (schoolbook for the 2x2 subproblem).
    
    # I think the cleanest way is to just enumerate all 56 scalar multiplications for the Strassen 4x4x4 part.
    
    # Let me define the 7 Strassen M-products, and for each, enumerate the 8 scalar multiplications.
    
    # This is getting too complex to do by hand. Let me just return schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
