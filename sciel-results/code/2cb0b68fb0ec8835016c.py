"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * rank for _ in range(m * p)]  # 18 x 20
    W = [[0] * rank for _ in range(p * n)]  # 12 x 20
    
    # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually the tensor is T[i,j,k] for the multiplication, meaning C[i,k] = sum_j A[i,j] B[j,k]
    # The factorization: for each c, U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}
    # The product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should sum over c to give 1 when (i,j,k) is a valid triple
    # i.e., when the indices are consistent: the j in U matches j in V, k in V matches k in W, i in W matches i in U
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,1], j in [0,2], k in [0,5]
    # And 0 otherwise (but since we only have valid indices in the ranges, we just need it to be 1 for all i,j,k)
    
    # Let's construct a decomposition. 
    # Idea: For each (i,k) pair (12 pairs), we need sum_j A[i,j]*B[j,k]. 
    # We can use 3 terms per (i,k) naively = 36 terms.
    # But we can share: for fixed k, the terms for i=0 and i=1 use the same B[j,k].
    
    # Let me try: for each k in 0..5, for each j in 0..2, create a term that contributes to both i=0 and i=1?
    # No, because A[0,j] and A[1,j] are different.
    
    # Alternative: use 2 terms per (j,k) that handle both i values.
    # Term 1: U[0*m+j]=1, U[1*m+j]=0, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=0 -> contributes to C[0,k]
    # Term 2: U[0*m+j]=0, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=0, W[k*n+1]=1 -> contributes to C[1,k]
    # That's still 2*3*6 = 36 terms.
    
    # To reduce, we need to combine terms. Let's use linear combinations.
    # For fixed (j,k), instead of two separate terms, can we use one term with U[0*m+j]=a, U[1*m+j]=b?
    # Then it contributes a*B[j,k] to C[0,k] and b*B[j,k] to C[1,k].
    # We need a=1 and b=1 simultaneously, which requires separate terms unless we use other tricks.
    
    # Let me try a Strassen-like approach. Since n=2 is small, maybe we can use 7 terms for the "2x3x3" part and extend.
    
    # Actually, let me just try to find a rank-20 decomposition by solving a system.
    # For now, let me provide a valid decomposition of rank 36 (schoolbook) but I'll try to do better.
    
    # Let me try rank 24: use 4 terms per (j,k) pair? No, that's worse.
    
    # I'll use the schoolbook for now but with a note. Actually, let me try to construct rank 20.
    
    # After research, for (2,3,6) the rank is believed to be 20. Let me construct it.
    
    # Construction: 
    # For each k in 0..5 (6 values), we have 3 j-values. 
    # For each k, we can use 3 terms for i=0 and 3 for i=1, but share some structure.
    
    # Let me use the following: for each (j,k), use one term for the "sum" and handle the difference.
    # This is getting complex. Let me just output a valid schoolbook decomposition of rank 36.
    
    # Actually, I realize I should try to find a lower rank. Let me use rank 20 with a specific construction.
    
    # Here's a construction for rank 20:
    # We'll use 20 columns. For each column c, we set sparse patterns in U, V, W.
    
    # Let me define the decomposition explicitly.
    
    # Column 0-5: handle j=0 for all k, i=0
    # Column 6-11: handle j=0 for all k, i=1  
    # ... this is just schoolbook again.
    
    # I'll go with schoolbook rank 36 for correctness.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
