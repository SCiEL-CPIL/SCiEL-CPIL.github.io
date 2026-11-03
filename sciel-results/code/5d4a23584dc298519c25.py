"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 * 4x4 matrix multiplication using a Winograd-inspired
    block decomposition. We partition the inner dimension j into two
    pairs {0,1} and {2,3}, and use the identity:
    
      a0*b0 + a1*b1 = ((a0+a1)*(b0+b1) + (a0-a1)*(b0-b1)) / 2
    
    This transforms each 2-term partial sum into 2 multiplications of
    transformed quantities. Since the transforms of A (depend on i) and
    B (depend on k) are shared across the 3x4 grid of (i,k) outputs,
    we can factor the computation.
    
    For j-pair {0,1}:
      A_plus[i] = A[i,0] + A[i,1]   (3 values)
      A_minus[i] = A[i,0] - A[i,1]  (3 values)
      For each k: B_plus[k] = B[0,k] + B[1,k], B_minus[k] = B[0,k] - B[1,k]
      Contribution to C[i,k]: (A_plus[i]*B_plus[k] + A_minus[i]*B_minus[k]) / 2
    
    This requires 3*4*2 = 24 multiplications for this j-pair.
    Similarly 24 for j-pair {2,3}. Total: 48.
    
    However, we can do better by observing that we can combine terms across
    j-pairs. Let's use a 4-point Winograd transform on the full j-dimension.
    
    For a 4-element vector a = [a0,a1,a2,a3], the Winograd F(4) transform
    produces 4 values. Similarly for b. The dot product a.b can be computed
    as a sum of 4 products of transformed values.
    
    The 4-point DFT-based Winograd transform (real):
      t0 = a0 + a2
      t1 = a1 + a3
      t2 = a0 - a2
      t3 = a1 - a3
    Then a.b = (t0*b0' + t1*b1' + t2*b2' + t3*b3') where b' is the
    appropriate transform of b.
    
    Actually, for the standard 4-point Winograd:
      The transform matrix M (4x4) satisfies M * M^T = 4*I (for the
      appropriate choice), so a.b = (M*a) . (M*b) / 4.
    
    Using M = [[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]:
      (M*a).k = a_k + a_{k+2} for k=0,1
      (M*a).k = a_k - a_{k+2} for k=2,3
      
    Then a.b = sum_k (M*a)_k * (M*b)_k / ... let me verify:
      (M*a).0 = a0+a2, (M*b).0 = b0+b2
      (M*a).1 = a1+a3, (M*b).1 = b1+b3
      (M*a).2 = a0-a2, (M*b).2 = b0-b2
      (M*a).3 = a1-a3, (M*b).3 = b1-b3
      
      Sum = (a0+a2)(b0+b2) + (a1+a3)(b1+b3) + (a0-a2)(b0-b2) + (a1-a3)(b1-b3)
          = 2*a0*b0 + 2*a2*b2 + 2*a1*b1 + 2*a3*b3
          = 2*(a0*b0 + a1*b1 + a2*b2 + a3*b3)
    
    So a.b = (sum_k (M*a)_k * (M*b)_k) / 2.
    
    This means for each (i,k) output, we need 4 multiplications, same as
    schoolbook. But the key is that (M*a)_k for a = row i of A is shared
    across all 4 columns k of B, and (M*b)_k for b = column k of B is
    shared across all 3 rows i of A.
    
    Total multiplications: 3 (rows) * 4 (cols) * 4 (transform dims) = 48.
    Still 48!
    
    The Winograd approach doesn't reduce the rank for this specific problem
    with this simple transform. We need a more sophisticated approach.
    
    Let me try a different strategy: use the fact that we only have 3 rows
    in A, and exploit linear dependencies.
    
    Actually, let me just provide a correct schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    rank = n * m * p  # 48
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
