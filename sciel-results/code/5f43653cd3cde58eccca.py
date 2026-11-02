"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The matrix multiplication tensor has entries T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We'll use a structured approach: try to find a low-rank decomposition using
    # the fact that this is a "simple" tensor (all entries are 1).
    
    # One known approach: use the "block diagonal" or "Kronecker" structure.
    # For multiplying an n x m matrix by an m x p matrix, the tensor is:
    # T = sum over (i,j,k) of e_i^T ⊗ e_j ⊗ e_k
    # This is exactly the tensor that represents matrix multiplication.
    
    # Known lower bounds: rank >= max(n*p, n*m, m*p) for certain structured decompositions
    # But the true minimum rank for 3x5x7 is unknown in general.
    
    # Let's try a constructive approach using circulant matrices or other structured forms.
    
    # Approach: Use the fact that we can embed the multiplication into a larger problem
    # and use FFT-like decompositions.
    
    # For now, let's try to find a decomposition of rank < schoolbook_rank using
    # a systematic search with structured patterns.
    
    # Key insight: The tensor T[i,j,k] = 1 for all valid (i,j,k) can be written as
    # T = sum_r u_r ⊗ v_r ⊗ w_r where u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
    
    # Let's try a decomposition where we use the structure of the indices.
    # If we can find rank-1 tensors that sum to the all-ones tensor, we're done.
    
    # A simple observation: T[i,j,k] = 1 = 1*1*1, so the schoolbook decomposition works.
    # To do better, we need to "share" the rank-1 terms.
    
    # Let's try using the identity: for any a, b, c with a*b*c = 1, we can write
    # 1 = a * (1/a) * 1, but this doesn't help directly.
    
    # Better approach: Use the fact that the tensor is a "product" tensor in some sense.
    # Actually, the matrix multiplication tensor is NOT a product tensor, so we need
    # multiple rank-1 terms.
    
    # Let's try a different strategy: use random initialization and gradient-free optimization
    # to find a low-rank decomposition.
    
    # First, let's establish that the schoolbook solution works, then try to improve.
    
    # For a 3x5x7 multiplication, the schoolbook rank is 3*5*7 = 105.
    # The Strassen algorithm gives rank 7 for 2x2x2, which is better than 8.
    # For our case, we can try to use a "block Strassen" approach.
    
    # Let's try to construct a decomposition using the following idea:
    # Split the matrices into blocks and use a recursive or structured decomposition.
    
    # Actually, let's just try to find ANY decomposition with rank < 105 using a simple heuristic.
    
    # We'll use the following approach:
    # 1. Start with the schoolbook decomposition (rank 105)
    # 2. Try to "merge" pairs of rank-1 terms if their sum can be expressed with fewer terms
    
    # For simplicity and to ensure correctness, let's return the schoolbook decomposition
    # but also try a quick search for a better one.
    
    # Let's try a structured decomposition based on the following idea:
    # Use the fact that 3, 5, 7 are all odd, and try to use a "cyclic" structure.
    
    # Actually, let's just implement a basic search that tries to find a decomposition
    # with rank slightly below 105, and if it fails quickly, return the schoolbook result.
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find a decomposition of rank r for r from 100 down to some lower bound
    # This is a very rough heuristic; in practice, we'd need a more sophisticated approach.
    
    # For now, let's just return the schoolbook decomposition to ensure correctness.
    # A more advanced version would try to find a better decomposition.
    
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
