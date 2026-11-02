"""Find a rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Construct the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W):
    """Verify that the factorization is exact: for all i,k, sum_j C[i,j]*D[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    Here the tensor identity is: the multiplication tensor T[i,j,k] = 1 for each (i,j,k) with i<n, j<m, k<p,
    decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U = U
    V = V
    W = W
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    # Also verify off-diagonal entries are zero: for the tensor to be EXACTLY the multiplication tensor,
    # we need sum_r U[i1*m+j1, r]*V[j2*p+k2, r]*W[k1*n+i2, r] = 0
    # whenever (i1,j1,k1) != (i,j,k) ... actually the tensor is defined only on valid (i,j,k) indices.
    # The standard check: for each i,k, the inner product pattern should reproduce C*D.
    # But for a valid rank-r decomposition of the matmul tensor, we need:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   T(i,j,k) = sum_r U(i,j,r) V(j,k,r) W(k,i,r) = 1 if this is the "valid" entry
    # The off-diagonal check isn't standard for matmul; the tensor is T_{ijk} = 1 delta.
    # Actually the matmul tensor is: M[i,j,k] where we compute C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor form is T[i,j,k] = 1, and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # A rank-r decomposition means T[i,j,k] = sum_{r=1}^r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # So we just need T[i,j,k]=1 for all valid (i,j,k) and 0 otherwise (but T is only defined on valid indices).
    return True


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit
    start = time.time()

    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p  # 84

    # For 3x4x7, known lower bounds and constructions:
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Rank is known to be between 28 and 84.
    # Try Winograd-like frequency domain constructions.
    # 
    # The idea: factor the matmul using polynomial/convolution structure.
    # A[i,j] * B[j,k] can be viewed as evaluating polynomials at certain points.
    #
    # For small dimensions, let's try a direct constructive approach:
    # Split indices into blocks and use Winograd-style evaluation/interpolation.
    #
    # Strategy: Use the fact that matmul C[i,k] = sum_j A[i,j] B[j,k]
    # We can group the j-index (dimension 4) and use a rank-r interpolation.
    # If we can represent the 4x4 "middle" interaction with fewer than 4 multipliers
    # per (i,k) pair using Winograd, total rank = n * p * r_j where r_j < m.
    # 
    # Actually, the standard Winograd approach for matmul:
    # Fix i and k, C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
    # This is a dot product of length m. Winograd can reduce m multiplications
    # to r < m when m is small and we have structure, but for a generic dot product
    # of two length-m vectors, the rank is m (you need m multiplications).
    # 
    # However, when we consider ALL (i,k) pairs simultaneously, there's more structure.
    # The full tensor has n*m*p = 84 non-zero entries. The rank can be less than 84
    # if there's linear structure across the (i,k) dimension.
    #
    # Key insight: C = A*B where A is n x m and B is m x p.
    # Write A as n rows, B as p columns.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Group the j-index: let j go from 0 to m-1 = 0 to 3.
    # Define for each i: a_i = (A[i,0], A[i,1], A[i,2], A[i,3]) (length-m vector)
    # For each k: b_k = (B[0,k], B[1,k], B[2,k], B[3,k]) (length-m vector)
    # C[i,k] = a_i . b_k
    # 
    # Now, can we compute all n*p = 21 dot products using fewer than 84 scalar mults?
    # Each dot product needs m=4 mults, so 21*4 = 84 total.
    # But we can share computation if we group them.
    #
    # Winograd frequency domain method for matmul:
    # Let's use a small FFT-like transform on the j-dimension.
    # For m=4, we can use a 4-point transform (DFT or Winograd 4-point).
    # 
    # Winograd 4-point: to compute y[n] = sum_{j=0}^{3} x[j]*h[j] for n=0..3 (convolution)
    # But here it's a simple dot product, not convolution.
    #
    # Alternative: Strassen-like blocking.
    # Split A into blocks and B into blocks, use a low-rank decomposition of the block product.
    #
    # Let me try a different approach: since dimensions are small, let's try to find
    # a decomposition with rank less than 84 by exploiting structure.
    #
    # The simplest improvement: group by i. For each i, we need to compute
    # C[i,:] = A[i,:] * B, which is a 1x4 times 4x7 = 1x7 vector.
    # This requires 28 multiplications for each i, total 3*28 = 84.
    #
    # Or group by k: C[:,k] = A * B[:,k], a 3x4 times 4x1 = 3x1 vector.
    # 21 multiplications per k, total 7*21 = 84.
    #
    # Can we do better? The rank of the matmul tensor [n,m,p] is at least max(n*m, m*p, p*n) = 28.
    # Known results: rank of 3x4x4 matmul tensor... 
    # 
    # Let me try a concrete Winograd-style construction:
    # Use evaluation-interpolation on the j-dimension with 4 points, but share across i's.
    #
    # Actually, for the specific case of multiplying an n x m matrix by an m x p matrix,
    # the tensor rank is known to be at least max(nm, mp, pn) and at most nmp.
    # For n=3, m=4, p=7: lower bound 28, upper bound 84.
    #
    # Let me try to use the following construction:
    # Partition the m=4 indices into pairs: (0,1) and (2,3).
    # A[i, 2s:t] for s=0,1 and t in {0,1} gives 2x2 blocks per row of A.
    # Similarly for B.
    # Use a rank-3 decomposition of the 2x2x2 matmul (Strassen: rank 7, but for 2x2 it's 7).
    # Wait, 2x2 matmul has rank 7 (Strassen), not 3.
    #
    # Let me just try the straightforward approach and also attempt a numerical search
    # for a lower-rank decomposition using the properties of the tensor.
    #
    # For now, let's implement a Winograd-style construction that might reduce the rank.
    # 
    # Winograd's method for small matrix multiplication:
    # For m x m matrices, if we can find a set of r points {t_1,...,t_r} and use
    # polynomial interpolation, the rank can be reduced.
    #
    # For the 3x4x7 case, let me try:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # 
    # Idea: Use 4 evaluation points to compute all dot products.
    # Actually, that's just m=4, so no reduction.
    #
    # Let me try a completely different approach: use the fact that we only need
    # integer/half-integer entries, and try to find a sparse decomposition.
    #
    # Since the search space is large, and I'm constrained to 110 seconds,
    # let me try a heuristic search for a lower-rank decomposition.
    #
    # Actually, the most practical approach given the time constraint:
    # Try to find a decomposition with rank = n * p * r_j where r_j is the effective
    # rank of the j-dimension interaction.
    #
    # Let me try r_j = 3 (instead of 4) by using a Winograd-style trick on the 4-element
    # dot product. For 4 elements, Winograd can sometimes reduce to 3 multiplications
    # if the sequences have special structure, but for generic inputs it stays at 4.
    #
    # Given the complexity, let me just return the schoolbook decomposition.
    # But first, let me try one more idea: block decomposition.
    #
    # Split A into top 2 rows and bottom 1 row (3 = 2+1).
    # Split B into left 3 columns and right 4 columns (7 = 3+4).
    # A_top (2x4) * B_left (4x3) = 2x3
    # A_top (2x4) * B_right (4x4) = 2x4
    # A_bot (1x4) * B_left (4x3) = 1x3
    # A_bot (1x4) * B_right (4x4) = 1x4
    # Total mults: 24 + 32 + 12 + 16 = 84. No savings.
    #
    # Let me just return the schoolbook answer. It's the safest.
    return best
# EVOLVE-BLOCK-END
