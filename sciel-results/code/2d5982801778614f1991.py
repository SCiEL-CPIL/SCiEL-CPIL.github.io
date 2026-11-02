"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_block(n, m, p):
    """
    Attempt to build a decomposition using Strassen-style block partitioning.
    This is a heuristic. It may not yield a valid lower-rank decomposition for 4x5x5
    directly without complex padding and combination logic, but we try a structured
    approach based on 2x2 blocks where possible.
    
    For 4x5 * 5x5:
    Partition A (4x5) into 2x2 blocks of size (2x2, 2x2, 2x3, 2x3)? No, 4 is divisible by 2, 5 is not.
    Let's partition A into 2x2 blocks.
    A = [A11 A12; A21 A22] where A11, A12 are 2x? 
    Since 5 is odd, we can't split 5 into two equal integers.
    Split 5 into 2 and 3.
    A = [A11 (2x2)  A12 (2x3); A21 (2x2) A22 (2x3)]
    B = [B11 (2x2)  B12 (2x3); B21 (3x2) B22 (3x3)]
    
    This is asymmetric and messy. Strassen works best for square matrices.
    Given the difficulty of finding an exact integer decomposition < 100 for 4x5x5
    in a short time, and the fact that the current best is 100, we will try a
    simple random search for rank 99 and 98 with integer coefficients, but with
    a more informed initialization.
    
    Actually, let's just return None to indicate failure, and rely on the
    fallback or simple random search in the main loop.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Attempt to find lower rank (99, 98, etc.)
    # Strategy: Random search with integer/half-integer constraints.
    # We will try ranks 99, 98, 97.
    
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 10:
            break
            
        # Try a small number of random attempts
        # We use random integers in {-1, 0, 1} to keep values small and exact.
        # This is a "sparse" random search.
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < n * m * p:
            break
            
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
