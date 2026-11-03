"""Rewritten program for matrix multiplication 3x4x6 tensor decomposition.

Approach:
- The standard schoolbook decomposition has rank n*m*p = 72.
- For matrix multiplication, the tensor rank is known to be bounded above by the
  number of non-zero entries in the tensor (which is n*m*p) but can be lower due
  to algebraic structure.
- For small dimensions like 3x4x6, we can try to find a rank-minimal decomposition
  by exploiting the fact that matrix multiplication is a bilinear map.
- The key insight is that the matrix multiplication tensor T[i,j,k] = e_{ik} (in some sense)
  can be decomposed using fewer than n*m*p terms if there are linear dependencies.
- However, for generic matrix multiplication, the rank is exactly n*m*p unless special
  structure exists.
  
- Actually, let me reconsider. The matrix multiplication tensor for A (n x m) * B (m x p)
  = C (n x p) has entries C_{i,k} = sum_j A_{i,j} * B_{j,k}.
  
- The tensor is T_{i,j,k} where the multiplication is: sum over j of U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
  Wait, let me re-read the trace convention.
  
- The task says: output factor follows the official trace convention W[k*n+i,r].
  So W has shape (p*n, rank), indexed by [k*n + i, r].
  
- The decomposition should satisfy:
  For all i in [0,n), k in [0,p):
    sum_{j=0}^{m-1} A[i,j] * B[j,k] = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
  Wait, that's not quite right. Let me think again.
  
- The standard form: C[i,k] = sum_j A[i,j] B[j,k]
  We want: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
  
  Hmm, that's a product of three terms which doesn't directly work. Let me reconsider.

- Actually, the correct formulation for tensor decomposition of matrix multiplication:
  The bilinear map (A, B) -> C where C = A*B can be written as:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  This is a sum of n*m*p rank-1 tensors (in the trilinear sense):
  C[i,k] = sum_{i,j,k} A[i,j] * B[j,k] * delta_{i,k}... no that's not right either.
  
- Let me use the standard approach: The matrix multiplication tensor is:
  T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^* \otimes e_j \otimes (e_k \otimes e_i)
  
  Wait, I think the correct way is:
  The bilinear form is: <A, B> -> C where C_{ik} = sum_j A_{ij} B_{jk}
  
  This can be written as sum over (i,j,k) of (A_{ij}) (B_{jk}) (1 if we're computing C_{ik})
  
  So the tensor has components T_{i,j,k}^{(i,k)} = 1, meaning for each triple (i,j,k),
  we contribute to C_{i,k} the product A_{i,j} * B_{j,k}.
  
- In the factorization framework:
  U has shape (n*m, rank), indexed by (i,j)
  V has shape (m*p, rank), indexed by (j,k)  
  W has shape (p*n, rank), indexed by (k,i) [note: k*n+i]
  
  The condition is: for each (i,k), sum_j A_{ij} B_{jk} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} * A_{ij}) * (sum_j V_{j*p+k, r} * B_{jk})
  
  Hmm, that's still a product of three things. I think I'm confusing myself.
  
- Let me look at this differently. The standard rank-1 decomposition for bilinear maps:
  A bilinear map f: X x Y -> Z can be written as sum_r alpha_r(x) * beta_r(y) * z_r
  where alpha_r: X -> R, beta_r: Y -> R, z_r in Z.
  
  For matrix multiplication: X = R^{n*m} (matrix A), Y = R^{m*p} (matrix B), Z = R^{n*p} (matrix C)
  f(A,B) = A*B
  
  A rank-1 term is: (u^T A) * (v^T B) * w where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}
  
  So we need: A*B = sum_r (u_r^T A) * (v_r^T B) * w_r
  
  In terms of entries:
  (A*B)_{ik} = sum_j A_{ij} B_{jk} = sum_r (sum_{i',j'} u_r[i',j'] A[i',j']) * (sum_{j',k'} v_r[j',k'] B[j',k']) * w_r[i,k]
  
  For this to hold for all A, B, we need specific structures on u_r, v_r, w_r.
  
- The schoolbook solution uses rank-1 terms where:
  u_r is the standard basis vector e_{i,j} (1 at position (i,j), 0 elsewhere)
  v_r is the standard basis vector e_{j,k} (1 at position (j,k), 0 elsewhere)  
  w_r is the standard basis vector e_{i,k} (1 at position (i,k), 0 elsewhere)
  
  This gives: (e_{ij}^T A) * (e_{jk}^T B) * e_{ik} = A_{ij} * B_{jk} * e_{ik}
  Summing over all (i,j,k) gives sum_{i,j,k} A_{ij} B_{jk} e_{ik} = (A*B) with the correct entries.
  
  So rank = n*m*p = 72 for the schoolbook.

- Can we do better? For matrix multiplication, the tensor rank is a well-studied problem.
  For 2x2x2, the rank is 7 (Strassen's algorithm) instead of 8.
  For larger matrices, the rank is generally believed to be close to n*m*p but can be slightly lower.
  
- For 3x4x6, I don't know of a known lower rank. Let me just implement the schoolbook and see if
  there's any obvious improvement. Given the complexity, I'll stick with the schoolbook for now,
  but I'll write it cleanly.

- Actually, wait. Let me reconsider whether there's a simple algebraic improvement.
  The matrix multiplication tensor for dimensions n x m x p has been studied. For small cases:
  - 2x2x2: rank 7 (Strassen)
  - 3x3x3: rank at most 23 (unknown if minimal is 23)
  
  For 3x4x6, I don't have specific knowledge. Let me just provide the schoolbook which is guaranteed correct.
"""

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: each (i,j,k) triple gives one rank-1 term
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i*m + j), representing the (i,j) entry of A
                U[i * m + j][column] = 1
                # V is indexed by (j*p + k), representing the (j,k) entry of B
                V[j * p + k][column] = 1
                # W is indexed by (k*n + i), representing the (k,i) position in C
                # (note: C has shape n x p, but W uses k*n + i indexing)
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
