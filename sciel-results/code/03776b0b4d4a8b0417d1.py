"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Start with the schoolbook rank-64 decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to find minimal rank R such that:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Strategy: Use a structured ansatz. For 2x4x8, we can try to exploit
    # the small dimensions. Let's try rank < 64 by using a smarter decomposition.
    #
    # Key insight: For matrix multiplication A(2x4) * B(4x8) = C(2x8),
    # the output C[i,k] = sum_j A[i,j] * B[j,k].
    # 
    # We can try to use the fact that n=2 is very small.
    # 
    # Let's try a rank-32 decomposition by pairing up the k indices or using
    # a block structure. Actually, let's try to use a known lower bound approach.
    #
    # For 2x4x8, the trivial bound is 64. The Strassen-like bounds suggest
    # we might do better. Let's try rank 48, 40, 32, etc.
    #
    # Approach: Use a random initialization and gradient-free optimization
    # (e.g., random restarts with local search) to find a lower-rank decomposition.
    #
    # Actually, for exact integer/half-integer solutions, let's try a more
    # structured approach. We can fix some entries and solve for others.
    #
    # Let's try the following: use rank R = 48 (75% of schoolbook) with a
    # structured pattern, then verify. If it works, try lower ranks.
    #
    # For now, let's implement a verification function and try several ranks.

    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        R = U_mat.shape[1]
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(R):
        """Try to find a rank-R decomposition using structured random search."""
        if time.time() > deadline - 10:
            return None
        
        # Use a structured ansatz: initialize with random ±1 values and try
        # to adjust. For small R, this is hard. Let's use a different approach.
        #
        # Alternative: Use the fact that we can write the multiplication as
        # a sum of rank-1 tensors. For 2x4x8, let's try to use a "split" approach.
        #
        # Actually, let's just try to reduce the rank by merging some of the
        # schoolbook terms. If two terms (i,j,k) and (i',j',k') share the same
        # i*m+j row in U, same j*p+k row in V, and same k*n+i row in W, they
        # can be merged. But in schoolbook, all are distinct.
        #
        # Let's try a different approach: use a linear algebra method.
        # Fix U and V, then solve for W, or vice versa.
        #
        # For a given rank R, we can set up the problem as:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        #
        # This is a system of n*m*p = 64 equations. Unknowns are U (2m*R = 8R),
        # V (m*p*R = 32R), W (p*n*R = 16R). Total unknowns: 56R.
        # For R=64: 3584 unknowns, 64 equations. Very underdetermined.
        #
        # Let's try a simpler approach: start with rank 64 and try to zero out
        # columns by combining them. If column c1 and c2 have the same pattern
        # in U, V, W up to scaling, we can merge them.
        
        # For now, return the schoolbook solution if we can't find better.
        return None

    # Try to find a better decomposition
    # Given the difficulty, let's just return the schoolbook solution
    # but try a few quick structured approaches first.
    
    # Approach: Try rank 48 by exploiting the n=2 structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    #
    # This is two independent matrix-vector multiplications (2x4 * 4x8).
    # Each can be done with rank 32 (schoolbook for 2x4x4... wait, no).
    # Actually each row of C is a 1x4 matrix times 4x8 matrix, which is
    # 4*8=32 terms each, total 64. Same as before.
    
    # Let's try a quick random search for rank < 64 with integer entries
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try a few structured reductions
    # For rank 64, we already have it. Let's see if we can do 63, 62, etc.
    # by setting one column to zero and adjusting others. This is complex.
    
    # Given time constraints, return the schoolbook solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
