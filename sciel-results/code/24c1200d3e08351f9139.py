"""Construct a rank-24 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    # Rank-21 core for 3x3 x 3x3 + 3 extra terms for column j=3
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    # Rank-21 decomposition for 3x3 x 3x3 (standard pattern)
    # Using the well-known rank-21 algorithm entries
    # Each term r has U entries (9), V entries (9 for j=0,1,2), W entries (9 for k=0,1,2)
    
    # The rank-21 3x3x3 decomposition (one standard form):
    # We'll use the pattern where terms are indexed and have specific nonzero entries
    
    # Standard rank-21 for 3x3: 21 terms
    # U[i*3+j], V[j*3+k], W[k*3+i] for the 3x3 subproblem
    
    # Here's a known rank-21 decomposition (entries are ±1 or 0):
    # Term structure for 3x3x3 (indices i,j,k in 0..2):
    
    r21_U = [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 0
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 1
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 2
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 3
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 4
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 5
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 6
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],  # term 7
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # term 8
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],  # term 9
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # term 10
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # term 11
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],  # term 12
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],  # term 13
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],  # term 14
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # term 15
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # term 16
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # term 17
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # term 18
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # term 19
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  # term 20
    ]
    # Wait, that's just identity - that's rank 9, not 21.
    # I need the actual rank-21 decomposition. Let me use a known one.
    
    # Actually, let me just use a simple valid construction:
    # For the 3x3 core, use the 9 schoolbook terms (rank 9)
    # For column j=3, use 3 terms
    # Total rank = 12
    # That's valid but not optimal. Let me use a better approach.
    
    # Simplest valid approach: schoolbook for all 12 (j,k) pairs = rank 12
    # But we can do better by sharing structure across k for each (i,j).
    
    # Actually, the cleanest valid rank: for each (i,j) pair, we need to compute
    # C[i,k] = sum_j' A[i,j'] * B[j',k] for all k.
    # We can use rank-9 for the 3x3 part and handle k=3 separately.
    
    # Let me just do the straightforward valid construction:
    # Rank = n * m * p = 36 is schoolbook.
    # But we can do: for each (i,j), one term per k gives 9*4=36.
    # Better: recognize that for fixed (i,j), the output is a vector in k-space.
    
    # Simplest correct approach that beats 36: 
    # Use 9 terms for the 3x3 submatrix (k=0,1,2) with rank-9 (schoolbook)
    # Plus 3 terms for k=3 column, one per row i (since for each i, C[i,3] = sum_j A[i,j]*B[j,3])
    # But that's not how tensor decomposition works - each term is a rank-1 slice.
    
    # Let me just use rank-12: for each (j,k) pair (12 pairs), one term
    # U[i*3+j, term] = A[i,j] coefficient... no, U is fixed coefficients.
    
    # OK, the correct tensor decomposition:
    # C[i,k] = sum_{j,j'} A[i,j] * B[j',k] * delta_{j,j'}
    # = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k,i',j',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}... 
    # No: T[i,j,k] (output) depends on A[i',j'] and B[j'',k']
    # T_{(i,k),(i',j'),(j'',k'')} = delta_{i,i'} * delta_{j',j''} * delta_{k,k''}
    # Wait, the convention: output C[i,k] = sum_{i',j',j'',k''} T[i,k,i',j',j'',k''] A[i',j'] B[j'',k'']
    # So T[i,k,i',j',j'',k''] = delta_{i,i'} * delta_{k,k''} * delta_{j',j''}
    
    # Rank-1 decomposition: T = sum_r U[i,j,r] * V[j',k',r] * W[k'',i',r]... 
    # No, the factorization is: T[i,k, i',j', j'',k''] = sum_r U[i,j,r] V[j'',k',r] W[k'',i',r]?
    # Let me re-read: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U indexed by (i,j), V by (j,k), W by (k,i)
    # Tensor: sum_r U[i,j,r] * V[j',k',r] * W[k'',i',r] should give delta...
    
    # The matrix mult tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # So C[i,k] = sum_{j1,k1,j2,k2} A[i,j1] B[j2,k2] * (delta_{j1,j2} * delta_{i,i} * delta_{k,k})
    # The tensor T[i,k, i2,j1, j2,k2] = delta_{i,i2} * delta_{j1,j2} * delta_{k,k2}
    # Factorization: sum_r U[i,j1,r] V[j2,k2,r] W[k,i2,r]... 
    
    # I think the convention is: sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] 
    # where U contracts with A[i',j']... 
    
    # Let me just verify with the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c=(i,j,k)
    # sum_r U[i,j,r] V[j',k',r] W[k'',i',r] 
    # = sum_{i0,j0,k0} delta_{i,i0}delta_{j,j0} * delta_{j',j0}delta_{k',k0} * delta_{k'',k0}delta_{i',i0}
    # = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # And C[i,k] should = sum_{i',j',j'',k''} T A[i',j']B[j'',k'']
    # So T[i,k,i',j',j'',k''] must satisfy sum T * A * B = C[i,k] = sum_j A[i,j]B[j,k]
    # This means T[i,k,i',j',j'',k''] = delta_{i,i'}delta_{k,k''}delta_{j',j''}... but that's 6 indices.
    
    # The factorization sum_r U[i,j,r]V[j'',k'',r]W[k,i',r] (guessing indices)
    # For schoolbook: = delta_{i,i'}delta_{j,j''}delta_{k,k}... 
    
    # I'll just trust the schoolbook is correct and build on it.
    # Schoolbook rank = 36. I need something less.
    
    # Key insight: for fixed (i), C[i,k] for k=0,1,2,3 can be computed.
    # If I use the rank-9 schoolbook for k=0,1,2 (9 terms) and 3 more for k=3, that's 12.
    # But the 3 extra terms for k=3: for each i, I need C[i,3] = sum_j A[i,j] B[j,3]
    # Term for (i, j=0): U[i*3+0, r]=1, V[0*4+3, r]=1, W[3*3+i, r]=1 -> contributes to C[i,3]
    # But this also needs to not interfere with k=0,1,2 outputs.
    # If I make separate terms, they only contribute to k=3 output.
    
    # So: 9 terms for (i,j) with k=0,1,2 using rank-9 schoolbook on 3x3x3
    # Plus 3 terms for the k=3 column, one per (i) ... but I need sum over j.
    
    # For k=3: C[i,3] = sum_j A[i,j] B[j,3]. 
    # I can use 3 terms (one per j) for each i? No, that's 9 more.
    # Or: for each j, one term that handles all i: 
    #   U[i*3+j, r_j] = 1 for all i, V[j*4+3, r_j] = 1, W[3*3+i, r_j] = 1 for all i
    #   Then sum_r U[i,j,r]V[j,3,r]W[3,i,r] = sum_j 1*1*1 = 3... not right.
    
    # Each rank-1 term: U[i,j,r] * V[j'',k'',r] * W[k,i',r]
    # For term r corresponding to (j0, k0=3): U[i,j0,r]=1, V[j0,3,r]=1, W[3,i,r]=1
    # Contribution to C[i,3]: sum_{i',j'} U[i,j',r] V[j',3,r] W[3,i',r] * A[i',j'] B[j',3]
    # Hmm, I'm confusing myself. Let me just do rank-36 schoolbook to be safe, 
    # or find a clearly valid smaller rank.
    
    # Safest: rank-12. For each (j,k) pair (j in 0..2, k in 0..3), one term:
    # U[i*3+j, r] = 1 for all i, V[j*4+k, r] = 1, W[k*3+i, r] = 1 for all i
    # Check: sum_r U[i,j',r] V[j'',k'',r] W[k,i',r] 
    # For fixed output (i,k): sum_{j',k''} sum_r [U[i,j',r] V[j'',k'',r] W[k,i',r]] A[i',j']B[j'',k'']
    # Term r=(j0,k0): U[i,j0,r]=1, V[j0,k0,r]=1, W[k0,i,r]=1
    # So sum_r U[i,j',r]V[j'',k'',r]W[k,i',r] = sum_{j0,k0} delta_{j',j0} delta_{j'',j0} delta_{k'',k0} delta_{k,k0}
    # = delta_{j',j''} delta_{k,k''}
    # Then C[i,k] = sum_{i',j',j'',k''} delta_{j',j''}delta_{k,k''} A[i',j']B[j'',k'']
    # = sum_{i',j'} A[i',j'] B[j',k]
    # But we need C[i,k] = sum_j A[i,j]B[j,k], not A[i',j']!
    # The i' is free - that's wrong.
    
    # I see the issue. The W factor has index (k,i') where i' is the A row index.
    # So W[k,i',r] couples output row i with input row i'.
    # For the identity to hold, we need delta_{i,i'} in the tensor.
    
    # Schoolbook: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for r=(i,j,k)
    # sum_r U[i,j',r]V[j'',k'',r]W[k,i',r] = delta_{j',j}delta_{j'',j}delta_{k'',k}delta_{i',i} = delta_{i,i'}delta_{j',j''}delta_{k,k''}
    # C[i,k] = sum A[i',j']B[j'',k''] delta_{i,i'}delta_{j',j''}delta_{k,k''} = sum_{j'} A[i,j']B[j',k]. CORRECT.
    
    # So I need delta_{i,i'} in W. That means W[k,i',r] is nonzero only when i' matches some function of r.
    
    # For rank-12 construction to work, I'd need to encode i in each term.
    # That gives rank = n*m*p = 36 again.
    
    # To reduce rank, I need to share terms across i or k or j.
    # The rank-21 for 3x3x3 does this by having terms where U,V,W have multiple nonzeros.
    
    # Given the complexity, let me just submit the schoolbook rank-36.
    # It's valid and gives score 1.0, same as current. Not an improvement but correct.
    
    # Actually wait - can I do rank-18? For each (i,j) pair, handle all 4 k values?
    # No, because W[k,i',r] needs to encode k.
    
    # Let me try: rank-12 where each term handles one (i,j) and all k:
    # Term r=(i0,j0): U[i0,j0,r]=1, V[j0,k,r]=1 for all k, W[k,i0,r]=1 for all k
    # sum_r U[i,j',r]V[j'',k'',r]W[k,i',r] 
    # = sum_{i0,j0} delta_{i,i0}delta_{j',j0} * delta_{j'',j0} * delta_{i',i0}
    # = delta_{i,i'}delta_{j',j''}
    # C[i,k] = sum A[i',j']B[j'',k''] delta_{i,i'}delta_{j',j''} = sum_{j'} A[i,j'] sum_{k''} B[j',k'']
    # That's sum_k'' B[j',k''] which is not right - we need B[j',k] specifically.
    
    # The k'' is summed over in the contraction, so we lose the k index. Wrong.
    
    # I'll stick with rank-36 schoolbook. It's guaranteed correct.
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
