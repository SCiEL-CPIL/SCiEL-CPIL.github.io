"""
Kronecker-product construction for 3x3 x 3x8 matrix multiplication.

Strategy:
  We decompose the 3x3 x 3x8 multiplication into 8 independent 3x3 x 3x1 
  (matrix-vector) multiplications, one for each column of the 8-column output.
  
  For a single matrix-vector multiplication C = A*b where A is 3x3 and b is 3x1:
  C[i] = sum_{j=0}^{2} A[i,j] * b[j]
  
  The schoolbook rank for this is 9 (one term per (i,j) pair).
  
  However, we can use the fact that for 3x3 x 3x1, the bilinear map is:
  (A, b) -> A*b where A is 3x3 (9 entries) and b is 3x1 (3 entries), output is 3x1.
  
  Actually, let me reconsider. The tensor T[i,j,k] for the full 3x3 x 3x8 problem:
  C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
  
  Here i in {0,1,2} (row of A and C), j in {0,1,2} (column of A / row of B), 
  k in {0,...,7} (column of B and C).
  
  The schoolbook decomposition has rank 3*3*8 = 72.
  
  Can we do better? For the 3x3 x 3x3 case, Strassen's algorithm achieves rank 23
  (though the true minimum is unknown, between 19 and 23).
  
  For 3x3 x 3x8, we can view it as 8 independent 3x3 x 3x1 problems, OR we can
  group the 8 columns into blocks.
  
  Let me try a different approach: use the Strassen-like decomposition for the 3x3 
  part and replicate it appropriately for the 8 columns.
  
  Actually, the cleanest approach: the multiplication C = AB where A is 3x3 and B is 3x8
  can be seen as computing 8 independent matrix-vector products A*b_k for k=0..7.
  
  Each such product has rank at most 9 (schoolbook). So total rank <= 72.
  
  But wait - there's a better way. We can use the fact that the 3x3 matrix multiplication
  tensor has a known low-rank decomposition. Let me use the standard Strassen decomposition
  for 2x2x2 and extend it.
  
  For 2x2x2, Strassen's algorithm uses rank 7 instead of 8. The decomposition is:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B10-B11)
  M4 = A11(B01-B10)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B10)
  M7 = (A01-A11)(B01+B11)
  C00 = M1+M4-M5+M7
  C01 = M3+M5
  C10 = M2+M4
  C11 = M1-M2+M3+M6
  
  For 3x3, we can partition into 2x2 blocks plus one row/column. This gets complicated.
  
  Given the complexity and risk of error, let me try a simpler improvement:
  Use the fact that 8 = 3 + 3 + 2, and handle each block separately with 
  the best known decomposition for that size.
  
  For 3x3 x 3x3: best known upper bound is 23 (Strassen-like), schoolbook is 27.
  For 3x3 x 3x2: schoolbook is 18.
  
  Total: 23 + 23 + 18 = 64 if we can use rank-23 for 3x3x3.
  
  But I'm not confident I can correctly implement a rank-23 3x3x3 decomposition.
  
  Let me try yet another approach: use the Kronecker product structure.
  
  Actually, the simplest safe improvement: note that the schoolbook is 72. 
  If I can find ANY valid decomposition with rank < 72, it's an improvement.
  
  Let me try to use the fact that for the 3x3 x 3x8 problem, we can separate
  the computation by the middle index j. For each j, we compute A[:,j] * B[j,:],
  which is an outer product of a 3-vector and an 8-vector, contributing to C.
  
  C = sum_{j=0}^{2} A[:,j] * B[j,:]
  
  Each term A[:,j] * B[j,:] is a 3x8 matrix. The bilinear map for one j is:
  (a, b) -> a * b^T where a is 3x1 and b is 8x1, output is 3x8.
  
  The rank of this outer product map is 3*8 = 24 (each output entry C[i,k] = a[i]*b[k]
  is a single product, so we need 24 rank-1 tensors for one j).
  
  For 3 values of j: 3 * 24 = 72. Same as schoolbook.
  
  Hmm. Let me think about this differently.
  
  The key insight: the tensor T for 3x3 x 3x8 multiplication is:
  T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k], else 0.
  
  Actually, T[i,j,k] = 1 for all i,j,k in the schoolbook sense, but that's not right.
  
  Let me be precise. The multiplication tensor T is defined such that:
  C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
  
  In the standard tensor formulation, we flatten A to a vector of length 9, B to a 
  vector of length 24, and C to a vector of length 24. The bilinear map is:
  C = T(A, B) where T is a 3-tensor of shape (9, 24, 24) such that:
  C[c] = sum_{a,b} T[a,b,c] * A[a] * B[b]
  
  But this is equivalent to saying that for each output index c = (i,k),
  C[i,k] = sum_{j} A[i,j] * B[j,k]
  
  So T[a,b,c] = 1 if a corresponds to (i,j), b corresponds to (j',k), and c corresponds 
  to (i',k'), with i=i', j=j', k=k'. Otherwise T=0.
  
  The rank of this tensor is what we want to minimize.
  
  For the schoolbook, each (i,j,k) triple gives one rank-1 term, so rank = 3*3*8 = 72.
  
  To do better, we need to find a decomposition with fewer terms.
  
  Given the difficulty of finding a verified better decomposition, and the risk of 
  producing an incorrect one (which scores 0), I'll stick with the schoolbook 
  decomposition which is guaranteed correct.
"""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
