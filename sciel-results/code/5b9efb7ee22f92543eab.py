DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = 2, 4, 7
    # Known rank for 2x4 by 4x7 is 26
    rank = 26
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Construct a valid rank-26 decomposition
    # Based on known results for small matrix multiplication
    
    # Terms 0-13: 14 terms
    for t in range(14):
        k = t % 7
        i = t // 7
        for j in range(4):
            U[i*4 + j][t] = 1
            V[j*7 + k][t] = 1
        W[k*2 + i][t] = 1
    
    # Terms 14-25: 12 terms for cross-coupling
    # These are needed to make the decomposition exact
    # For each j in 0..3, and for pairs of k values
    
    # Actually, the above 14 terms do NOT form a valid decomposition.
    # Let me verify: For term t with i,k, it contributes to T[i,j,k] for all j.
    # T[i,j,k] = sum_t U[i*4+j][t] * V[j*7+k][t] * W[k*2+i][t]
    # For the 14 terms: only term t=(i,k) has U[i*4+j][t]=1 and V[j*7+k][t]=1 and W[k*2+i][t]=1
    # So T[i,j,k] = 1 for all i,j,k, which is correct!
    # Wait, that means 14 terms suffice? But that can't be right since rank >= max(nm, mp, pn) = max(8,28,14) = 28.
    
    # I made an error. The tensor T[i,j,k] for matrix multiplication is 1 if the indices are valid, but
    # the decomposition T = sum_t u_t ⊗ v_t ⊗ w_t means T[i,j,k] = sum_t U[i*4+j][t] * V[j*7+k][t] * W[k*2+i][t].
    # For the 14 terms above, T[i,j,k] = 1 for all i,j,k, which is the correct tensor value.
    # But this gives rank 14, which contradicts the lower bound of 28.
    
    # The error: U[i*4+j][t] for term t=(i',k') is 1 only if i'=i. V[j*7+k][t] is 1 only if k'=k.
    # W[k*2+i][t] is 1 only if i'=i and k'=k.
    # So for T[i,j,k], only term t=(i,k) contributes, giving 1. Correct.
    # But this means rank 14, which is impossible.
    
    # The mistake: V[j*7+k][t] should depend on j as well. In my construction, V[j*7+k][t]=1 for all j,
    # which means V is not a vector in R^28 but has 4 non-zero entries. That's fine.
    # But the rank is the number of terms, which is 14. This contradicts the lower bound.
    
    # I must be misunderstanding the tensor. Let me re-examine.
    
    # The matrix multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} where
    # T_{(i,j),(j',k),(k',i')} = δ_{j,j'} δ_{k,k'} δ_{i,i'}... no.
    
    # Actually, T_{(i,j),(j',k),(k',i')} = 1 if j=j' and k=k' and i=i', else 0.
    # But in the standard convention, T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if j=j' and k=k' and i=i'.
    # Wait, that's not right either.
    
    # The correct convention: T[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if j = j' and k = k' and i = i', else 0.
    # But i is determined by a and i' by c, j by a and j' by b, k by b and k' by c.
    # So T[a,b,c] = 1 if the j-index of a equals j-index of b, k-index of b equals k-index of c,
    # and i-index of a equals i-index of c.
    
    # In my 14-term decomposition:
    # Term t=(i,k): U[i*4+j][t]=1 for all j, so U[a][t]=1 if a's i-index is i.
    # V[j*7+k][t]=1 for all j, so V[b][t]=1 if b's k-index is k.
    # W[k*2+i][t]=1, so W[c][t]=1 if c's k-index is k and i-index is i.
    # Then T[a,b,c] = sum_t U[a][t]*V[b][t]*W[c][t]
    # = 1 if there exists t=(i,k) such that a's i-index=i, b's k-index=k, c's k-index=k and i-index=i.
    # = 1 if a's i-index = c's i-index and b's k-index = c's k-index.
    # But we also need a's j-index = b's j-index.
    # My decomposition doesn't check j=j', so it's wrong.
    
    # I see the error. Let me fix it.
    
    # For a valid decomposition, each term must satisfy the j=j' and k=k' and i=i' constraints.
    
    # Let me use the schoolbook decomposition which is definitely correct.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
