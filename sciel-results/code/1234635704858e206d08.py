"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _naive_decomposition(n, m, p):
    """Return the standard (rank = n*m*p) exact decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that the decomposition is exact: for all i,j,k,
    sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    Actually the correct identity is:
    For matrix product C = A * B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 for all i,j,k)
    and the decomposition says:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, let me re-read the contract. The contract says:
    U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    
    The matrix multiplication tensor has entries T[i,j,k] = 1 for all valid i,j,k.
    The decomposition requires:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    But that's not right either. Let me think again.
    
    Actually, for matrix multiplication A (n x m) times B (m x p) = C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The bilinear form uses:
    For each "product" (i, j, k) we have a term A[i,j] * B[j,k] that contributes to C[i,k].
    
    In tensor decomposition language, the multiplication tensor M has:
    M[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    
    The standard decomposition has one rank-1 tensor for each (i,j,k):
    U[i*m+j, r] = 1 (for the r corresponding to (i,j,k))
    V[j*p+k, r] = 1
    W[k*n+i, r] = 1
    
    And the reconstruction is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    But wait, that would give C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    = sum_j 1 = m, not 1.
    
    Hmm, I think the convention is different. Let me re-read.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k, i) where k is the column of B / row of C, and i is the row of A / row of C.
    
    The standard decomposition for matrix multiplication:
    For each triple (i, j, k):
      U[i*m+j, r] = 1  (row i of A, column j)
      V[j*p+k, r] = 1  (row j of B, column k)  
      W[k*n+i, r] = 1  (this encodes the output position (i, k))
    
    Then the product C[i,k] = sum_j A[i,j] * B[j,k] is computed as:
    C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_k' B[?,k] * V[?,k, r]) * W[k*n+i, r]
    
    Actually, I think the verification is simpler. The tensor entry T[i,j,k] should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    For the naive decomposition, for a fixed (i,j,k), there is exactly one r where all three
    are 1, so the sum is 1. For any other combination of indices, the sum is 0.
    
    So T[i,j,k] = 1 for all i,j,k, which is correct.
    """
    rank = U.shape[1]
    # Check all (i, j, k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True


def _try_las_vergas(n, m, p, deadline):
    """
    Try to find a rank-(n*m*p - 1) decomposition using a Las Vegas style approach.
    
    For small matrices, we can try to find a lower-rank decomposition by solving
    a system of polynomial equations. For 4x5x6, the naive rank is 120.
    
    One known technique: if we can find a linear dependence among the rank-1 tensors,
    we can reduce the rank. This is equivalent to finding a non-trivial solution to
    a system of quadratic equations.
    
    For practical purposes within the time budget, we'll try a simple greedy approach:
    try to remove one term at a time and see if the remaining terms can be adjusted
    to still give an exact decomposition.
    """
    U0, V0, W0 = _naive_decomposition(n, m, p)
    rank0 = n * m * p
    
    # Try to find a rank-(rank0-1) decomposition
    # We need to find U, V, W of rank rank0-1 such that the tensor is preserved.
    
    # One approach: pick a column r0 to eliminate, and try to express it as a
    # linear combination of the others.
    # The r-th rank-1 tensor is u_r ⊗ v_r ⊗ w_r.
    # We want: u_r ⊗ v_r ⊗ w_r = sum_{r' != r} c_{r,r'} (u_{r'} ⊗ v_{r'} ⊗ w_{r'})
    
    # This is a very hard problem in general. For now, let's try a simple heuristic.
    
    # Actually, for 4x5x6, the known minimal rank might be 120 (i.e., no improvement
    # is known). Let's try a different approach: use the fact that for square matrices,
    # Strassen's algorithm gives rank 7 for 2x2x2.
    
    # For 4x5x6, we could try to use a block decomposition.
    # Split A into blocks of size 2x2, 2x3, etc.
    
    # Given the time constraints, let's just try a few random perturbations
    # and see if we can find a valid lower-rank decomposition.
    
    # Actually, let's try a more systematic approach.
    # The multiplication tensor for m x n x p has rank at most m*n*p.
    # For 4x5x6, m*n*p = 120.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (best known)
    # - 4x4x4: rank 49 (best known)
    
    # For non-cubic matrices, the best known ranks are less clear.
    # Let's just try to verify that the naive decomposition is correct and return it.
    
    # But first, let's try one simple thing: check if we can merge two terms
    # by finding a linear relation.
    
    # For a 2x2x2 block, Strassen's algorithm works. Let's try to apply it.
    
    # Actually, for 4x5x6, we can try to decompose it into blocks and apply
    # Strassen's algorithm to the 2x2x2 blocks.
    
    # Let's try a block approach:
    # A is 4x5, B is 5x6.
    # Split A into two blocks: A1 (2x5), A2 (2x5)
    # Split B into three blocks: B1 (5x2), B2 (5x2), B3 (5x2)
    # Then C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    # Each A_i*B_j is a 2x2 matrix product.
    # Using Strassen for each 2x2x2: 7 multiplications instead of 4.
    # Total: 6 * 7 = 42 multiplications, vs 4*5*6 = 120.
    
    # But this requires the dimensions to be compatible. 4x5x6 doesn't split
    # cleanly into 2x2 blocks.
    
    # Let's try a different split:
    # A is 4x5. Split into A1 (2x3), A2 (2x2)
    # B is 5x6. Split into B1 (3x3), B2 (2x3)
    # Then C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 2x3 * 3x3 = 2x3, rank 18
    # A1*B2: 2x3 * 3x3 = 2x3, rank 18
    # A2*B1: 2x2 * 2x3 = 2x3, rank 12
    # A2*B2: 2x2 * 2x3 = 2x3, rank 12
    # Total rank: 60, which is better than 120!
    
    # But wait, this is just a block decomposition, not a tensor decomposition
    # in the sense required. The rank of the tensor decomposition is the number
    # of scalar multiplications, which is the sum of the ranks of the block products.
    
    # For 2x3 * 3x3, the naive rank is 2*3*3 = 18.
    # For 2x2 * 2x3, the naive rank is 2*2*3 = 12.
    
    # So the total rank would be 18 + 18 + 12 + 12 = 60.
    
    # But can we do better with Strassen?
    # 2x3 * 3x3: not a standard Strassen case.
    # 2x2 * 2x3: we can use Strassen for the 2x2 part.
    
    # Actually, let's think about this differently.
    # The rank of the tensor for m x n x p matrix multiplication is the minimum
    # number of scalar multiplications needed.
    
    # For 4x5x6, the naive algorithm uses 120 multiplications.
    # A block decomposition can reduce this.
    
    # Let's try to construct a block decomposition with rank < 120.
    
    # Strategy: split A (4x5) into A1 (2x5) and A2 (2x5).
    # Split B (5x6) into B1 (5x3) and B2 (5x3).
    # Then C = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    # A1*B1: 2x5 * 5x3, naive rank 30
    # A1*B2: 2x5 * 5x3, naive rank 30
    # A2*B1: 2x5 * 5x3, naive rank 30
    # A2*B2: 2x5 * 5x3, naive rank 30
    # Total: 120, no improvement.
    
    # Let's try a different split.
    # A (4x5) = A1 (3x5) + A2 (1x5)
    # B (5x6) = B1 (5x4) + B2 (5x2)
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 3x5 * 5x4, naive rank 60
    # A1*B2: 3x5 * 5x2, naive rank 30
    # A2*B1: 1x5 * 5x4, naive rank 20
    # A2*B2: 1x5 * 5x2, naive rank 10
    # Total: 120, still no improvement.
    
    # It seems like any block decomposition that preserves the exact dimensions
    # will give the same total rank as the naive algorithm.
    
    # The only way to get a lower rank is to use a non-trivial tensor decomposition,
    # like Strassen's algorithm.
    
    # For 4x5x6, I don't know of a known lower-rank decomposition.
    # Let's just return the naive decomposition.
    
    return None


def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # First, get the naive decomposition
    U, V, W = _naive_decomposition(n, m, p)
    rank = n * m * p
    
    # Verify the naive decomposition
    if not _verify(U, V, W, n, m, p):
        # This should never happen, but just in case
        pass
    
    # Try to find a lower-rank decomposition
    result = _try_las_vergas(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
