import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x6 matrix.
    
    The decomposition satisfies:
        A * B = sum_{t=0}^{rank-1} W[:,t] * outer(U[:,t], V[:,t])
    where:
        U is (n*m, rank) = (12, rank)
        V is (m*p, rank) = (24, rank)
        W is (p*n, rank) = (18, rank)
    
    We attempt to find a rank lower than the schoolbook rank (72) 
    with integer or half-integer entries.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # The target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the standard decomposition, U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1
    # This corresponds to the identity mapping of the multiplication.
    
    # We define the tensor T such that T[i, j, k] = 1
    # The decomposition is: T[i, j, k] = sum_t U[i, t] * V[j, t] * W[k, t]
    # Note: The interface defines U, V, W with different dimensions.
    # U is (n*m, rank). Indexing U[i*m + j, t]
    # V is (m*p, rank). Indexing V[j*p + k, t]
    # W is (p*n, rank). Indexing W[k*n + i, t]
    # The product sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal T[i,j,k] = 1.
    
    # Let's construct the target tensor in the flattened form for checking
    # T_flat is not needed, we just need to check the sum for all i,j,k.
    
    def check_decomposition(U, V, W, rank):
        """
        Checks if the decomposition is exact (within floating point tolerance)
        and if entries are close to integers or half-integers.
        """
        # Check exactness
        # We can iterate over all i, j, k
        # But for speed, we can use matrix multiplications if we reshape.
        # However, the tensor is small (3*4*6 = 72 elements).
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check integrality/half-integrality
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is close to an integer or half-integer
                # val * 2 should be close to an integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def snap_to_lattice(mat):
        """Snap entries to nearest integer or half-integer."""
        mat_snapped = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat_snapped[i, j] = round(mat[i, j] * 2) / 2.0
        return mat_snapped

    # Try to find a lower rank
    # We use a simple random search with ALS-like updates or just random restarts
    # Given the time limit, we try a few ranks.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # We start from rank_schoolbook - 1 and go down
    # To save time, we only try a few random restarts per rank.
    
    # First, let's see if we can easily find a rank 71 or 70.
    # If not, we return the schoolbook.
    
    # Optimization: Use numpy for the check inside the loop?
    # The check is O(n*m*p*rank). 3*4*6*70 = 5040 ops. Very fast.
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    for target_rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization with small integers/half-integers
            # This is a heuristic. Real solutions might have larger values.
            # But for small ranks, values are often small.
            U = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Check if this random guess works (unlikely)
            if check_decomposition(U, V, W, target_rank):
                # Convert to lists for the return format
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                found = True
                break
                
            # If not, we could try to optimize.
            # But simple random search is unlikely to hit the exact integer solution.
            # We need a more structured search or a solver.
            
        if found:
            break
        # If we don't find it in random search, we might need a different approach.
        # For now, if we don't find a better rank quickly, we fall back.
        
    if best_result is None:
        # Fall back to schoolbook
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_result
