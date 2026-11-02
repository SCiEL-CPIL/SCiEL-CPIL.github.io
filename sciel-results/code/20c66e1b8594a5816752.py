"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 75).
    2. Use a structured Kronecker product idea: since n=3, m=5, p=5, we can think
       of the multiplication as involving a 3x1 block structure.
       
       However, the most effective approach for small matrices is to use known
       results or attempt rank reduction via algebraic methods.
       
       For 3x5 * 5x5, the trivial bound is 75. The Strassen-like improvements
       or other bilinear algorithms might reduce this, but for small matrices
       exact low-rank decompositions are hard to find.
       
       Let's try a different approach: use the fact that we can sometimes
       combine terms. For matrix multiplication C = A*B where A is 3x5 and B is 5x5,
       C[i,k] = sum_j A[i,j] * B[j,k].
       
       The tensor T[i,j,k] = e_i^T * e_j^T * e_k^T (in appropriate bases) has rank
       at least max(n,m,p) and at most n*m*p.
       
       For 3x5x5, let's see if we can find a rank < 75 decomposition.
       
       Actually, let's use a randomized approach with integer/half-integer constraints
       to try to find a lower rank decomposition. But this is very hard.
       
       Alternative: Use the known fact that for n x m * m x p, if we can find a
       rank-r decomposition, it corresponds to r "multiply-add" operations.
       
       For small matrices, let's try to use a greedy rank reduction approach:
       Start with the schoolbook basis and try to find linear dependencies.
       
       Actually, the most practical approach within time limits is to:
       1. Return the schoolbook decomposition (rank 75) as a fallback.
       2. Attempt a simple rank reduction by checking if any combination of the
          75 terms can be eliminated.
       
       Given the difficulty, let's implement a search for rank < 75 using
       integer programming or a heuristic.
       
       For now, let's implement a heuristic that tries to find a rank-15 decomposition
       based on the Kronecker product structure mentioned in the strategy.
       
       The idea: If we write A as a 3x1 block of 1x5 rows, and B as a 5x1 block of
       1x5 rows, the multiplication involves 3*5 = 15 "row-column" products, but
       each product is a scalar times a 1x5 vector, so we need 5 operations per
       such product, giving 75 again.
       
       However, if we use a rank-3 decomposition for the 3x1 part and rank-5 for
       the 5x5 part, we might get rank 15. But this requires the 3x1 and 5x5
       multiplications to be separable, which they are not in general.
       
       Let's try a different approach: use the fact that for 3x5 * 5x5, we can
       view it as 3 independent 1x5 * 5x5 multiplications. Each 1x5 * 5x5
       multiplication computes 5 outputs, each being a dot product of a 5-vector
       with 5 rows of B. This can be done with 5 multiplications per output,
       so 25 per row of A, and 3 rows, giving 75.
       
       Without a breakthrough, let's return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Attempt to find a lower rank decomposition
    # Try ranks from 15 to rank_schoolbook - 1
    best_result = {'U': U, 'V': V, 'W': W}
    best_rank = rank_schoolbook
    
    # For each candidate rank, try to find a decomposition
    # This is a very hard problem, so we'll use a heuristic approach
    
    # Heuristic: Try to find a rank-r decomposition by solving a system of
    # polynomial equations. For small r, this might be feasible.
    
    # Let's try a simple approach: use random initializations and gradient descent
    # to minimize the error, then round to nearest half-integer and check.
    
    def check_decomposition(U_mat, V_mat, W_mat, n, m, p, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        # The tensor is T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
        # In our convention: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The decomposition gives: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # The tensor entry is 1 for all valid (i,j,k)
                    actual = 0
                    for c in range(rank):
                        actual += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition using a simple search
    # This is very expensive, so we'll only try a few ranks
    
    for candidate_rank in range(15, min(rank_schoolbook, 40)):
        if time.time() > deadline - 10:
            break
        
        # Try a random initialization approach
        np.random.seed(42)
        
        U_rand = np.zeros((n * m, candidate_rank))
        V_rand = np.zeros((m * p, candidate_rank))
        W_rand = np.zeros((p * n, candidate_rank))
        
        # Initialize with random half-integers
        for _ in range(10):  # 10 restarts
            if time.time() > deadline - 5:
                break
            
            U_rand = np.round(np.random.randn(n * m, candidate_rank) * 2) / 2
            V_rand = np.round(np.random.randn(m * p, candidate_rank) * 2) / 2
            W_rand = np.round(np.random.randn(p * n, candidate_rank) * 2) / 2
            
            # Check if this random initialization works
            if check_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p, candidate_rank):
                return {'U': U_rand.tolist(), 'V': V_rand.tolist(), 'W': W_rand.tolist()}
        
        # If random doesn't work, try a structured approach
        # For rank 15, use the Kronecker product idea
        if candidate_rank == 15:
            # Try to use a 3x5 structure
            # This is a guess, might not work
            pass
    
    # If we didn't find a better decomposition, return the schoolbook
    return best_result
# EVOLVE-BLOCK-END
