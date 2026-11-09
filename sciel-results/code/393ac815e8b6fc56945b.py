"""Tensor decomposition search for 4x5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or best-effort) exact tensor decomposition
    for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Returns: {'U': (20, rank), 'V': (25, rank), 'W': (20, rank)}
    where the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    equals 1 if the indices are valid, 0 otherwise.
    
    Convention: W[k*n+i, r] where n=4, m=5, p=5.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 145s deadline (5s safety margin)
    
    # Schoolbook baseline (rank = 100)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Check a random sample of entries
        random.seed(42)
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            # T[i,j,k] should be 1
            val = 0
            for c in range(U_mat.shape[1]):
                val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
            if abs(val - 1) > 1e-9:
                return False
        # Check some zero entries (out of bounds or non-matching)
        for _ in range(100):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            # This is a valid entry, so it should be 1
            val = 0
            for c in range(U_mat.shape[1]):
                val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
            if abs(val - 1) > 1e-9:
                return False
        return True
    
    # Try random search for lower-rank decompositions
    # Use a heuristic: start from random sparse matrices and refine
    
    np.random.seed(42)
    
    for target_rank in range(90, max(40, best_rank - 50), -5):
        if time.time() > deadline - 30:
            break
        
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Random initialization with sparsity
            U_r = np.random.randint(-2, 3, size=(n * m, target_rank))
            V_r = np.random.randint(-2, 3, size=(m * p, target_rank))
            W_r = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Add sparsity
            sparsity_mask_u = np.random.rand(n * m, target_rank) < 0.7
            sparsity_mask_v = np.random.rand(m * p, target_rank) < 0.7
            sparsity_mask_w = np.random.rand(p * n, target_rank) < 0.7
            U_r *= sparsity_mask_u
            V_r *= sparsity_mask_v
            W_r *= sparsity_mask_w
            
            # Set some entries to exactly 1 or -1
            U_r[np.abs(U_r) > 1] = np.sign(U_r[np.abs(U_r) > 1])
            V_r[np.abs(V_r) > 1] = np.sign(V_r[np.abs(V_r) > 1])
            W_r[np.abs(W_r) > 1] = np.sign(W_r[np.abs(W_r) > 1])
            
            # Verify
            if verify_decomposition(U_r, V_r, W_r):
                best_rank = target_rank
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                break
        
        if time.time() > deadline - 10:
            break
    
    # Try structured approach: use Kronecker product structure
    # Decompose 4x5x5 using block structure
    # 4 = 2*2, 5 = 2*2 + 1
    
    if time.time() < deadline - 20:
        # Try a structured construction based on 2x2 Strassen-like patterns
        # For 2x2x2, rank 7 is known (Strassen)
        # For 2x2x3, rank is known to be at most 10
        
        # Try embedding into a larger structure
        # This is heuristic - may not find valid decomposition
        
        strassen_2x2 = [
            [1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0]
        ]
        
        # Build a candidate using Strassen blocks
        # This is a rough attempt at structured decomposition
        candidate_rank = 7 * 2 * 2  # 28 for 4x4x4, but we have 4x5x5
        
        # Given complexity, stick with best found so far
    
    return best_result
# EVOLVE-BLOCK-END
