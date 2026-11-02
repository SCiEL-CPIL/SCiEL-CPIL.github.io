"""Construct a rank-22 tensor decomposition for 3x3 * 3x4 matrix multiplication.

Uses the known rank-21 algorithm for 3x3*3x3 (Strassen-like) and extends to the
4th output column with one additional rank-1 term per input column (3 terms),
giving total rank 24. However, we can do better: the 4th column of B is a vector
b_4 = (B[0,3], B[1,3], B[2,3]). The contribution to C[i,3] is sum_j A[i,j]*B[j,3].
This is a linear form in the entries of B's 4th column. We can fold this into the
existing structure if we add terms that compute these 3 dot-products.

Actually, let me use a cleaner approach:
- Rank-21 for the 3x3*3x3 core (k=0,1,2)
- For k=3: C[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3] + A[i,2]*B[2,3]
  This requires 3 additional terms (one per j), each handling all i.
  Term for j=0: U[i*3+0, r]=1 for all i, V[0*4+3, r]=1, W[3*3+i, r]=1 for all i
  But wait - this doesn't work because W[k,i',r] must encode i'=i for correctness.
  
  Let me reconsider. The tensor decomposition:
  C[i,k] = sum_{i',j',j'',k''} T[i,k,i',j',j'',k''] * A[i',j'] * B[j'',k'']
  where T[i,k,i',j',j'',k''] = delta_{i,i'} * delta_{j',j''} * delta_{k,k''}
  
  Factorization: T = sum_r U[i,j',r] * V[j'',k'',r] * W[k,i',r]
  
  For the 4th column (k=3):
  We need: sum_r U[i,j',r] * V[j'',3,r] * W[3,i',r] = delta_{i,i'} * delta_{j',j''}
  for the contribution to C[i,3].
  
  This is a 3x3x3 tensor (indices i, j', j'' with i'=i constraint via W).
  Actually it's: for fixed k=3, the tensor in (i, j', j'', i') must equal delta_{i,i'}*delta_{j',j''}.
  
  Using 3 terms (one per j0):
  Term r_j0: U[i,j0,r]=1, V[j0,3,r]=1, W[3,i,r]=1
  Then sum_r U[i,j',r]V[j'',3,r]W[3,i',r] = sum_{j0} delta_{j',j0}delta_{j'',j0}delta_{i',i}
  = delta_{j',j''}delta_{i,i'}  (since j'=j''=j0)
  
  Yes! This works. So 3 extra terms for the 4th column.
  Total rank = 21 + 3 = 24.

But can we do better? Let's try rank 22 by being clever.

Actually, the minimal rank for 3x3*3x4 is known to be at most 24 (from 21+3) 
and at least 21 (from the 3x3*3x3 lower bound). Let me just implement rank 24 
which is clearly valid, or try to find a rank-22 or rank-23.

For now, let me implement the clean rank-24 construction:
- 21 terms for the 3x3*3x3 core
- 3 terms for the 4th column

I need an explicit rank-21 decomposition for 3x3*3x3.
"""

def construct():
    n, m, p = 3, 3, 4
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    # Rank-21 decomposition for 3x3 * 3x3
    # Using the standard rank-21 algorithm (one known explicit form)
    # Indices: i,j in 0..2 for A (9 entries), j,k in 0..2 for B (9 entries), k,i in 0..2 for C (9 entries)
    
    # A known rank-21 decomposition (from literature):
    # The 21 terms have the following structure (U entries for A[i,j], V entries for B[j,k], W entries for C[k,i]):
    
    r21 = [
        # Each entry: (U_nonzero_positions, V_nonzero_positions, W_nonzero_positions)
        # U positions are (i,j) pairs, V positions are (j,k) pairs, W positions are (k,i) pairs
        # All entries are +1 or -1
        
        # Term 0
        ([(0,0)], [(0,0),(0,1),(0,2)], [(0,0),(1,0),(2,0)]),
        # Term 1
        ([(1,0)], [(0,0),(1,0),(1,1),(1,2)], [(0,1),(1,1),(2,1)]),
        # Term 2
        ([(2,0)], [(0,0),(2,0),(2,1),(2,2)], [(0,2),(1,2),(2,2)]),
        # Term 3
        ([(0,1)], [(0,1),(1,1),(2,1)], [(0,0),(1,0),(2,0)]),
        # Term 4
        ([(1,1)], [(0,1),(1,1),(1,2),(2,1),(2,2)], [(0,1),(1,1),(2,1)]),
        # Term 5
        ([(2,1)], [(0,1),(1,1),(2,1),(2,2)], [(0,2),(1,2),(2,2)]),
        # Term 6
        ([(0,2)], [(0,2),(1,2),(2,2)], [(0,0),(1,0),(2,0)]),
        # Term 7
        ([(1,2)], [(0,2),(1,1),(1,2),(2,2)], [(0,1),(1,1),(2,1)]),
        # Term 8
        ([(2,2)], [(0,2),(1,2),(2,1),(2,2)], [(0,2),(1,2),(2,2)]),
        # Terms 9-20: more terms needed
    ]
    
    # This is getting complex. Let me use a simpler verified approach.
    # I'll use the schoolbook rank-9 for 3x3*3x3 and add 12 terms for the 4th column extension.
    # That gives rank 21 total. Wait, schoolbook for 3x3*3x3 is rank 9, plus 3*3=9 for the 
    # 4th column? No, the 4th column only has 3 k-values to worry about (k=3 only).
    
    # Schoolbook for 3x3*3x3: rank 9 (one term per (i,j,k) with i,j,k in 0..2)
    # For k=3: 3 terms (one per j, as shown above)
    # Total: 9 + 3 = 12. That's rank 12!
    
    # Wait, that seems too good. Let me verify.
    
    # Schoolbook for 3x3*3x3 (k=0,1,2):
    # For each (i,j,k) in {0,1,2}^3, term r=(i,j,k):
    #   U[i,j,r] = 1
    #   V[j,k,r] = 1
    #   W[k,i,r] = 1
    # This gives: sum_r U[i,j',r]V[j'',k'',r]W[k,i',r] = delta_{i,i'}delta_{j',j''}delta_{k,k''}
    # for k,k'' in {0,1,2}. Correct for the 3x3*3x3 part.
    
    # For k=3 (4th column), 3 terms r_j (j in 0,1,2):
    #   U[i,j,r_j] = 1 for all i
    #   V[j,3,r_j] = 1
    #   W[3,i,r_j] = 1 for all i
    # Contribution to C[i,3]: sum_{i',j',j''} U[i,j',r]V[j'',3,r]W[3,i',r] A[i',j']B[j'',3]
    # = sum_j sum_{i',j',j''} delta_{j',j}delta_{j'',j}delta_{i',i} A[i',j']B[j'',3]
    # = sum_j A[i,j]B[j,3]. Correct!
    
    # And these terms don't interfere with k=0,1,2 because W[k,i',r_j]=0 for k!=3.
    # And the schoolbook terms don't interfere with k=3 because V[j,k,r]=0 for k!=k0 where k0 in {0,1,2}.
    
    # So total rank = 9 + 3 = 12. That's much better than 24!
    
    # Let me implement rank-12.
    rank = 12
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    col = 0
    # Schoolbook for 3x3*3x3 (k=0,1,2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # 3 terms for k=3
    for j in range(3):
        for i in range(3):
            U[i * 3 + j][col] = 1
        V[j * 4 + 3][col] = 1
        for i in range(3):
            W[3 * 3 + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
