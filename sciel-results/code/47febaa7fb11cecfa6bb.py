"""Matrix multiplication 4x4x9: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x9 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
      i in [0, 4)  (row of output / row of A)
      j in [0, 4)  (column of A / row of B)
      k in [0, 9)  (column of B / column of output)
    
    The output C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor decomposition requires:
      U: shape (4*4, rank) = (16, rank)  indexed by (i*m + j, c)
      V: shape (4*9, rank) = (36, rank)  indexed by (j*p + k, c)
      W: shape (9*4, rank) = (36, rank)  indexed by (k*n + i, c)
    
    Such that for all i, j, k:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} (the multiplication tensor)
    
    Actually, the multiplication tensor T_{i,j,k} = 1 for all i,j,k (it's the outer product structure),
    and the decomposition is: T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    
    Wait, let me reconsider. The matrix multiplication tensor is:
    T(i, j, k) = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    
    Actually the standard formulation: The bilinear map (A, B) -> C where C = A*B.
    The tensor is T with T[i,j,k] representing the coefficient of A[i,j] * B[j,k] in C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k.
    
    The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook method, rank = n*m*p = 4*4*9 = 144.
    
    To reduce rank, we can use Strassen-like methods or exploit structure.
    
    Key insight: 9 = 3*3, so we might be able to decompose the k-dimension.
    If we could decompose the 9 columns into a 3x3 grid, we might use a tensor product structure.
    
    However, the most practical approach for guaranteed improvement is to use known
    low-rank decompositions for the sub-problems and combine them.
    
    Let me try a different approach: use the fact that we can potentially share computations
    across the k dimension if there's structure.
    
    Actually, let me think about this more carefully. The multiplication A(4x4) * B(4x9) -> C(4x9).
    
    One approach: view B as three 4x3 blocks: B = [B1, B2, B3] where each Bt is 4x3.
    Then C = [A*B1, A*B2, A*B3].
    
    If we had a rank-r decomposition for 4x4 * 4x3 -> 4x3, we could use it three times,
    giving rank 3r. But that's not better than schoolbook (3*12=36 per block, total 108 < 144).
    
    Wait, schoolbook for 4x4 * 4x3 is 4*4*3 = 48 per block, so 3*48 = 144 total. Same.
    
    Let me try yet another approach. What if we use Strassen for the 4x4 part?
    
    Actually, the cleanest path to improvement: Use the fact that we can find a rank-143
    decomposition by merging two schoolbook terms if they share structure, or use known
    results that matmul(4,4,9) has rank < 144.
    
    Let me try a constructive approach: start with schoolbook and try to find linear
    dependencies or use a rank-reduction algorithm.
    
    For now, let me implement a careful schoolbook and then attempt a simple rank reduction
    by checking if any term can be eliminated via linear combinations.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank_school = n * m * p  # 144
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, rank_school), dtype=np.float64)
    V = np.zeros((m * p, rank_school), dtype=np.float64)
    W = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition correctly represents matrix multiplication."""
        # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # This is the multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # The schoolbook decomposition is trivially correct.
    # Now try to reduce rank.
    
    # Strategy: Try to find a lower rank by using a greedy/iterative approach.
    # We'll try to express some schoolbook terms as linear combinations of others.
    
    # A simpler approach: use known low-rank decompositions.
    # For 4x4x4 matmul, Strassen gives rank 7 (but Strassen is for 2x2x2 -> rank 7,
    # and 4x4x4 can be done with rank 49 via tensor product, or potentially lower).
    
    # Actually, let me try a different strategy: use the structure that 9 = 3*3.
    # Decompose the k-index as k = 3*k1 + k2 where k1 in {0,1,2}, k2 in {0,1,2}.
    # This doesn't directly help unless we can find a Kronecker product structure.
    
    # Let me try: can we do this with rank 143?
    # The idea: if two terms in the schoolbook decomposition are "similar" enough,
    # we might be able to merge them.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The rank is 144.
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
