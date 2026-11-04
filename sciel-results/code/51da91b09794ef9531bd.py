"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition of 2x2 matrix multiplication (rank 7)
    # T_{2,2,2} = sum_{c=0}^{6} u_c ⊗ v_c ⊗ w_c
    # Each u_c, v_c, w_c is a vector in R^4
    # Indexing: u[i*2+j] for A[i,j], v[j*2+k] for B[j,k], w[k*2+i] for C[i,k]
    
    # Strassen terms for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, express as a product of two linear forms.
    # The rank-1 decomposition of T_{2,2,2} via Strassen:
    # Each term is (linear form in A) * (linear form in B) -> C
    
    # u_c vectors (in R^4, indexed by [i*2+j]):
    # v_c vectors (in R^4, indexed by [j*2+k]):
    # w_c vectors (in R^4, indexed by [k*2+i]):
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # u: A11+A22 -> [1,0,0,1] (A[0,0]=1, A[1,1]=1)
    # v: B11+B22 -> [1,0,0,1] (B[0,0]=1, B[1,1]=1)
    # w: C11+C22 -> [1,0,0,1] (C[0,0]=1, C[1,1]=1)
    # But wait, M1 contributes to C11 and C22 with different coefficients.
    
    # Actually, I need to be more careful. The Strassen decomposition gives 7 bilinear terms,
    # but each term contributes to multiple C entries with specific coefficients.
    
    # Let me define the 7 terms as (u, v, w) where u, v, w are vectors in R^4.
    # The tensor is sum_c u_c ⊗ v_c ⊗ w_c.
    
    # Term 0: (A11+A22)(B11+B22)
    # u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
    # This gives T[0,0,0] += 1, T[0,0,1] += 1, T[0,1,0] += 1, T[0,1,1] += 1, etc.
    # But this is NOT correct because (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # which is 4 separate products, not 1.
    
    # I think the issue is that the Strassen decomposition at the 2x2 level is already a rank-7
    # decomposition, meaning it's expressed as 7 rank-1 terms. Let me look up the correct form.
    
    # The correct rank-7 decomposition of T_{2,2,2}:
    # The 7 terms are:
    # 1. (A11+A22) * (B11+B22) -> contributes to C11 and C22
    # 2. (A21+A22) * B11 -> contributes to C21 and C22
    # 3. A11 * (B12-B22) -> contributes to C12 and C22
    # 4. A22 * (B21-B11) -> contributes to C21 and C11
    # 5. (A11+A12) * B22 -> contributes to C12 and C11
    # 6. (A21-A11) * (B11+B12) -> contributes to C21 and C22
    # 7. (A12-A22) * (B21+B22) -> contributes to C11 and C12
    
    # Each of these is a product of two linear forms, which is a rank-1 term in the tensor
    # IF we consider the linear forms as vectors. But (A11+A22)*(B11+B22) is not a single
    # rank-1 term - it's the product of two vectors, which gives a rank-1 term in the
    # outer product sense.
    
    # Actually, (sum_i a_i * A_i) * (sum_j b_j * B_j) = sum_{i,j} a_i * b_j * A_i * B_j
    # This is a sum of rank-1 terms, not a single rank-1 term.
    
    # So the Strassen decomposition is NOT a rank-7 decomposition in the tensor sense!
    # It's a decomposition into 7 bilinear terms, each of which has rank 1 in the
    # "block" sense but higher rank in the "element" sense.
    
    # I'm confused. Let me just implement the schoolbook decomposition and return that.
    
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
# EVOLVE-BLOCK-END
