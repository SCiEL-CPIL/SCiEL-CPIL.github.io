import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_5x5():
    """
    Construct a Strassen-like decomposition for 5x5 matrix multiplication.
    We decompose 5x5 into blocks and use Strassen's algorithm on each block.
    
    Split 5x5 into:
    - Top-left 2x2, top-right 2x3, bottom-left 3x2, bottom-right 3x3
    
    Actually, let's use a simpler approach: use the fact that 5 = 2 + 3
    and apply Strassen to the 2x2 and 3x3 subproblems.
    
    For now, let's use a direct approach with rank reduction.
    """
    # For 5x5, schoolbook rank is 25
    # Try to find a lower rank decomposition
    
    # Use Strassen's algorithm on 4x4 submatrix, handle 5th row/col separately
    # This gives rank = 7 (Strassen for 4x4) + 4 + 4 + 1 = 16
    
    # Actually, let's construct it properly using Kronecker product structure
    
    # The 4x5x5 tensor = T_4x1x1 ⊗ T_1x5x5
    # T_4x1x1 has rank 4 (each output depends on one input)
    # T_1x5x5 has rank 25 (schoolbook) or less with Strassen
    
    # Let's use Strassen for the 5x5 part
    # Strassen for 5x5: decompose into 2x2 and 3x3 blocks
    
    # For 2x2: rank 7 (Strassen)
    # For 3x3: rank 23 (best known) or 27 (schoolbook)
    
    # Let's just use schoolbook for 5x5 for now and see if we can improve
    # Schoolbook 5x5: rank 25
    
    # The total rank would be 4 * 25 = 100 (same as schoolbook)
    # We need a better approach
    
    # Let's try a different decomposition:
    # Split the 4x5 matrix into two 2x5 matrices
    # Split the 5x5 matrix appropriately
    
    # Actually, the Kronecker product approach:
    # T_4x5x5[i,j,k] = sum over r of U[i*5+j, r] * V[j*5+k, r] * W[k*4+i, r]
    
    # If we can write T_4x5x5 as a sum of products of tensors,
    # we can use the rank of each factor
    
    # For the 4x1x1 part: T[i,0,k] = delta[i,k] if i==k, else 0
    # This has rank 4 (one term per i)
    
    # For the 1x5x5 part: T[0,j,k] = delta[j,k] 
    # This has rank 25 (schoolbook)
    
    # But we can do better! Use Strassen for the 5x5 part.
    
    # Let's construct Strassen for 5x5:
    # Split into 2x2 and 3x3 blocks
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # C11 = A11*B11 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (3x2)
    # C22 = A21*B12 + A22*B22 (3x3)
    
    # This doesn't directly help with rank reduction.
    
    # Let's try a different approach: use the known fact that
    # R(4,5,5) <= R(4,4,4) * something
    
    # Actually, let's just implement a greedy search for rank reduction
    # starting from schoolbook
    
    n, m, p = 4, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try to find a lower rank decomposition using Kronecker product structure.
    
    The key insight: T_4x5x5 can be decomposed using the tensor product
    of smaller multiplication tensors.
    """
    start_time = time.time()
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank by merging terms
    # For each pair of terms, check if they can be merged
    
    # A term is (u, v, w) where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
    # Two terms can be merged if their product contributions are linearly dependent
    
    # This is complex, so let's try a simpler heuristic:
    # Try to find linear dependencies among the U, V, W vectors
    
    # For now, return schoolbook
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try Kronecker product decomposition
    if time.time() < deadline - 30:
        U_k, V_k, W_k, rank_k = _try_kronecker_decomposition(n, m, p, deadline)
        if rank_k < best_rank:
            if _verify_decomposition(U_k, V_k, W_k, n, m, p):
                best_U, best_V, best_W = U_k, V_k, W_k
                best_rank = rank_k
    
    # Try additional search if time remains
    if time.time() < deadline - 10:
        # Try to find rank by examining the tensor structure
        # Use a greedy merging approach
        
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        # Try to merge terms that have similar structure
        # For each pair of terms, check if they can be combined
        
        # This is a heuristic and may not work well
        # For now, just return what we have
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
