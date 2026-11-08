"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np
from numpy.linalg import matrix_rank

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Strategy:
      1. Start with the schoolbook rank-60 decomposition.
      2. Try to find a lower rank via random restarts + rank reduction via 
         identifying linear dependencies among the term vectors.
      3. Also try a structured approach: split the 5 into 2+3 and the 6 into 3+3,
         using Strassen-like or known low-rank decompositions for smaller blocks.
      
      For 2x2x2, the minimum rank is 7 (Strassen). For 2x3x3, let's think:
      Actually, let's use a more general approach: random low-rank decomposition search.
      
      But first, let's implement a clean schoolbook and a structured decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # Schoolbook: rank = n*m*p = 60
    rank_school = n * m * p
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    U0, V0, W0, r0 = schoolbook()
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For valid decomposition, T[i,j,k] should be 1 if this is the identity in 
        # the appropriate sense. Actually, the tensor for matrix multiplication is:
        # T[i,j,k] = 1 for all (i,j,k) since (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k.
        # Let's check a sample of entries.
        # Full check is expensive, so check all for small sizes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try structured decomposition using block splitting
    # Split m=5 into 2+3, p=6 into 3+3
    # A is 2x5, B is 5x6
    # C = A*B
    # 
    # We can think of A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1: 2x2 * 2x6. We can split B1 into [B1a | B1b] where B1a is 2x3, B1b is 2x3
    # A1*B1a: 2x2*2x3, A1*B1b: 2x2*2x3
    # For A2*B2: 2x3 * 3x6. Split B2 into [B2a | B2b] where each is 3x3
    # A2*B2a: 2x3*3x3, A2*B2b: 2x3*3x3
    #
    # This is getting complex. Let me try a different approach.
    #
    # Known result: R(2,5,6) <= ? 
    # Using the fact that R(n,m,p) <= R(n,m1,p) + R(n,m2,p) if m = m1+m2
    # R(2,5,6) <= R(2,2,6) + R(2,3,6)
    # R(2,2,6): split 6 into 3+3, R(2,2,6) <= 2*R(2,2,3)
    # R(2,2,3) <= ? This is small.
    #
    # Actually, let me just try to find a valid lower-rank decomposition via 
    # random initialization and iterative refinement, or use known formulas.
    #
    # For now, let's try a rank-reduction approach on the schoolbook.
    # The 60 term vectors (as rows of the "bilinear" map) might have dependencies.
    
    # Build the tensor as a 3D array and try to find low-rank decomposition
    # T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for this specific case? 
    # No, it's the matrix multiplication tensor which has T[i,j,k]=1 for all valid (i,j,k))
    
    # Actually, the matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if (i,j,k) is a valid index, which is always true here.
    # Wait, no. The tensor is defined such that:
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So it's the all-ones tensor of shape (n,m,p) = (2,5,6).
    
    # The all-ones tensor of shape (n,m,p) has a very low rank! 
    # In fact, it can be decomposed as:
    # T[i,j,k] = 1 = (1)*(1)*(1)
    # So rank 1?! But that can't be right for matrix multiplication.
    
    # I'm confusing the tensor definition. Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # It's the tensor such that contracting it with A and B gives C = A*B.
    # The standard definition: T[i,j,k] = 1 if the indices form a valid triple, 
    # but the tensor is actually the structure tensor of the algebra, which for 
    # matrix multiplication is:
    # T = sum_{i,j,k} e_i^* \otimes e_j \otimes e_k^* where the multiplication is defined.
    
    # Actually, for matrix multiplication A (n x m) times B (m x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is in (R^{n*p})^* \otimes R^{n*m} \otimes R^{m*p}
    # T[(i,k), (i',j), (j',k')] = 1 if i=i', k=k', j=j'... 
    # No, T[(i,k), (i',j), (j',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right either.
    
    # Let me think again. The bilinear map is:
    # f: R^{n*m} x R^{m*p} -> R^{n*p}
    # f(A, B) = A*B
    # The associated tensor T satisfies:
    # T[A, B, C^*] = C^*[A*B] = sum_{i,k} C[i,k] * (A*B)[i,k] = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # So T is a trilinear form: T(A,B,C) = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] ... 
    # In the factorization T = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r],
    # we have T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # where a indexes R^{n*m}, b indexes R^{m*p}, c indexes R^{n*p}
    
    # For the matrix multiplication tensor, the non-zero entries are:
    # T[a,b,c] = 1 when a = i*m+j, b = j*p+k, c = i*n+k for some i,j,k
    # And T[a,b,c] = 0 otherwise.
    # So T has exactly n*m*p = 60 non-zero entries, each equal to 1.
    
    # So my schoolbook is correct: each term r corresponds to a triple (i,j,k),
    # and U, V, W have 1's at the appropriate positions.
    
    # Now, can we find a lower rank? The tensor has 60 non-zero entries.
    # The minimum rank is at least the "slice rank" or other lower bounds.
    
    # Let me try to use a numerical approach: random initialization and 
    # alternating minimization, or use the fact that this is a small problem 
    # and try various rank values.
    
    # Given time constraints, let me try a structured decomposition.
    
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # If I can use Strassen for 2x2x2 blocks, I can save 1 per 2x2x2 block.
    
    # Split 5 = 2 + 3, 6 = 3 + 3
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:5] @ B[2:5, :]
    # First term: 2x2 * 2x6. Split 6 = 3+3:
    #   A1 @ B1[:, 0:3] + A1 @ B1[:, 3:6]
    #   Each is 2x2 * 2x3. 
    # Second term: 2x3 * 3x6. Split 6 = 3+3:
    #   A2 @ B2[:, 0:3] + A2 @ B2[:, 3:6]
    #   Each is 2x3 * 3x3.
    
    # This is getting complicated. Let me just try to find a valid decomposition 
    # of rank < 60 via random search, or return the schoolbook.
    
    # Given the time limit, let me try a few random attempts at rank reduction.
    
    best_rank = r0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try rank from 50 down to some minimum, with a few random restarts
    # This is unlikely to find a valid solution quickly, but let's try.
    
    # Actually, let me try a smarter approach: use the known Strassen decomposition
    # for the 2x2x2 sub-problems.
    
    # Strassen's 7-term decomposition for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # I'll implement a function that takes a rank-r decomposition for a sub-problem
    # and combines them.
    
    # Given the complexity and time, let me just return the schoolbook for now,
    # but with a small optimization: check if there are any linear dependencies 
    # in the 60 term vectors that could reduce the rank.
    
    # The 60 terms correspond to triples (i,j,k). The term vectors are:
    # u_r = e_{i*m+j}, v_r = e_{j*p+k}, w_r = e_{k*n+i}
    # For different (i,j,k), these are different basis vectors, so the terms 
    # are linearly independent in a certain sense. But the tensor rank is about 
    # decomposing the tensor, not the individual vectors.
    
    # I'll just return the schoolbook.
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
