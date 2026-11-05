"""
Matrix multiplication 3x3 * 3x7 via Kronecker substitution + Strassen.

Idea:
  Map A (3x3) to a polynomial a(x) of degree < 3 by packing rows:
      a(x) = sum_{i=0}^{2} A[i,0] * x^{i*3} + A[i,1] * x^{i*3+1} + A[i,2] * x^{i*3+2}
  Map B (3x7) to a polynomial b(x) of degree < 3 by packing rows:
      b(x) = sum_{j=0}^{2} B[j,0] * x^{j*3} + B[j,1] * x^{j*3+1} + ... + B[j,6] * x^{j*3+6}
  
  The product c(x) = a(x) * b(x) has coefficients that encode C[i,k] at specific positions,
  but with overlaps. This direct packing doesn't work cleanly because of carry/overlap issues.

Alternative cleaner approach:
  Use the standard Kronecker substitution for matrix multiplication:
  For A (n x m) and B (m x p), define:
    a(x) = sum_{i=0}^{n-1} sum_{j=0}^{m-1} A[i,j] * x^{i*(m+p) + j}  ... this gets complicated.

Let me use a different, well-known approach:
  The rank of matrix multiplication <n x m, m x p> is at most the rank of the tensor.
  
  For 3x3 * 3x7, let's try a direct construction using the fact that we can use
  Strassen's algorithm on a 2x2 subblock and handle the rest with schoolbook.

Actually, let me try a simpler but effective approach:
  Use the "border rank" or a known decomposition.

Given the complexity, let me try a heuristic: use the schoolbook but see if we can
merge some terms. In the schoolbook, we have 63 terms. Each term is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].

Let me try to use a rank-42 decomposition by grouping the j-index differently.

Actually, here's a cleaner idea:
  For each pair (i, k), C[i,k] = sum_j A[i,j] * B[j,k].
  We can write this as an inner product of the vector (A[i,0], A[i,1], A[i,2]) and (B[0,k], B[1,k], B[2,k]).
  
  The tensor for the inner product of two 3-vectors has rank 3 (it's just the standard dot product).
  But we have 21 such pairs (i,k), so naively rank 63.

  However, the vectors for different (i,k) are not independent in a way that allows sharing.

Let me just implement a verified rank-50 decomposition using a structured approach.

Actually, I'll use the following approach:
  Pad the 3x3 * 3x7 problem to a 4x4 * 4x8 problem, use a known low-rank decomposition for that,
  then restrict back. But this is complex.

Given the time constraints, let me implement a simple but valid improvement:
  Use the schoolbook but with a different indexing that might allow some terms to be shared.

Actually, let me just return the schoolbook decomposition cleanly. The score will be 1.0,
but it's correct. Improving it requires a genuine algorithmic insight.

Wait, I just realized: I can try to use a rank-42 decomposition by noting that
for fixed j, the contribution is A[:,j] * B[j,:] which is an outer product.
There are 3 such j-values, each giving a 3x7 outer product.
Each outer product of a 3-vector and a 7-vector has rank 21 in the tensor sense? No,
each is a single rank-1 tensor in the appropriate space.

Hmm, this is getting confusing. Let me just implement the schoolbook correctly.
"""

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    rank = n * m * p  # 63
    U = [[0] * rank for _ in range(n * m)]  # 9 x 63
    V = [[0] * rank for _ in range(m * p)]  # 21 x 63
    W = [[0] * rank for _ in range(p * n)]  # 21 x 63
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}
