"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = delta[i,k] (for C[i,k] = sum_j A[i,j] B[j,k])
    # Our decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0 if i == k else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - target) > tol:
                    return False
    return True

def _construct_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    
    return U, V, W

def _try_strassen_3x4x7():
    """
    Attempt to construct a lower-rank decomposition using Strassen-style
    block decomposition.
    
    Strategy: Split 3x4x7 into blocks and use Strassen for 2x2x2 sub-blocks.
    
    3x4x7 = C(3x7) = A(3x4) * B(4x7)
    
    Split rows of C: 3 = 2 + 1
    Split cols of C: 7 = 2 + 2 + 2 + 1
    Split cols of A / rows of B: 4 = 2 + 2
    
    Block structure:
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  (2x2x2 blocks)
    C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    C[0:2, 6:7] = A[0:2,0:2]*B[0:2,6:7] + A[0:2,2:4]*B[2:4,6:7]
    C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6]
    C[2:3, 6:7] = A[2:3,0:2]*B[0:2,6:7] + A[2:3,2:4]*B[2:4,6:7]
    
    For 2x2x2 blocks, use Strassen (rank 7 instead of 8).
    For 1x2x2 or 2x1x2 or 1x1x2 etc., use schoolbook.
    
    This is complex. Let me try a simpler approach: use the known rank 23
    for 3x3x3 and extend.
    
    Actually, let me try a direct numerical search approach.
    """
    n, m, p = DIMENSIONS
    
    # Try random initialization with iterative refinement
    # We want to find rank-r decomposition where r < n*m*p
    
    best_rank = n * m * p
    best_U, best_V, best_W = _construct_schoolbook()
    
    # Try ranks from best_rank-1 down to some minimum
    # For each rank, try random initialization and gradient descent
    
    deadline = time.time() + 120
    
    for rank in range(best_rank - 1, max(20, best_rank // 2), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple iterative refinement
            # The objective is to minimize ||T - sum_c U_c ⊗ V_c ⊗ W_c||^2
            # This is a non-convex problem.
            
            # Use a simple approach: alternating minimization
            # For each c, the term is U[:,c] * V[:,c] * W[:,c]
            
            # This is complex. Let me try a different approach.
            # Use the fact that for matrix multiplication, we can use
            # Winograd-like methods.
            
            pass  # Skip for now, return schoolbook
    
    return best_U, best_V, best_W

def _try_lanczos_style():
    """
    Try a Lanczos-style or Winograd-style decomposition.
    
    For 3x4x7, we can try to use known efficient algorithms.
    
    One approach: 3x4x7 can be related to 3x3x3 (rank 23) and extensions.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is 84
    # Try to find a decomposition with rank < 84
    
    # For now, return schoolbook
    U, V, W = _construct_schoolbook()
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for 3x4x7 matrix multiplication.
    
    Returns a dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook()
    
    # Try to find a better decomposition
    # For now, just return schoolbook
    # A proper search would go here
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
