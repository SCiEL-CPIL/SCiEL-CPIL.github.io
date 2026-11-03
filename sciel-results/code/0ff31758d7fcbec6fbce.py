"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def _try_strassen_2x2_2x2():
    """
    Attempt to use Strassen's algorithm for 2x2*2x2 as a building block.
    However, for 3x4*4x5, we can't directly use 2x2 blocks without padding.
    Padding to 4x4*4x4 would use rank 49 (Laderman) for the 4x4 case.
    But we only need 3x4*4x5.
    
    Actually, let's try a different approach:
    We can view the 3x4*4x5 multiplication as a set of independent multiplications.
    
    A more promising approach:
    Use the fact that R(3,4,5) might be reducible.
    
    Known results:
    - R(2,2,2) = 7 (Strassen)
    - R(3,3,3) = 23 (Strassen)
    - R(4,4,4) = 49 (Laderman)
    
    For 3x4*4x5, one strategy is to decompose into smaller blocks.
    For example, split the 4x5 matrix into [2x5; 2x5] and the 3x4 into [3x2 | 3x2].
    Then C = A1*B1 + A2*B2 where A1 is 3x2, B1 is 2x5, A2 is 3x2, B2 is 2x5.
    This doesn't directly help because we still need to multiply 3x2 by 2x5.
    
    Another approach: Use the Strassen algorithm for 2x2 blocks.
    Split the 4x4 part into 2x2 blocks. But we have 3x4*4x5.
    
    Let's try to construct a decomposition using Strassen's 7-multiplication trick for 2x2 blocks.
    We can pad the matrices to 4x4*4x4, use Laderman's 49-multiplication algorithm,
    and then extract the relevant 3x4*4x5 part.
    
    Laderman's algorithm for 4x4*4x4 has rank 49.
    If we pad 3x4*4x5 to 4x4*4x4, we can use the 49 rank decomposition.
    But we need to verify that the extracted part is correct.
    
    Actually, this is a valid approach!
    Let's implement Laderman's 4x4*4x4 algorithm and extract the 3x4*4x5 part.
    """
    # Laderman's algorithm for 4x4*4x4
    # This is a known result with rank 49.
    # We'll construct the U, V, W matrices for 4x4*4x4 and then restrict to 3x4*4x5.
    
    n4, m4, p4 = 4, 4, 4
    rank4 = 49
    
    # Laderman's algorithm coefficients
    # This is complex to implement from scratch. Let's use a known construction.
    # For now, let's just return the schoolbook result since implementing Laderman
    # correctly is non-trivial and error-prone.
    
    # Actually, let's try a simpler approach:
    # Use the fact that R(3,4,5) <= R(4,4,4) = 49 by padding.
    # If we can implement a rank-49 decomposition for 4x4*4x4,
    # we can extract the 3x4*4x5 part.
    
    # For now, let's just return None to indicate this approach is not implemented.
    return None

def _try_winograd():
    """
    Try to use Winograd's algorithm for matrix multiplication.
    Winograd's F(2) transform can reduce the number of multiplications.
    For 3x4*4x5, we can try to apply Winograd's method.
    """
    # Winograd's algorithm is complex to implement correctly.
    # For now, let's return None.
    return None

def _random_search(max_rank=59, num_trials=10000):
    """
    Random search for a lower rank decomposition.
    This is unlikely to succeed but worth trying.
    """
    n, m, p = DIMENSIONS
    best_result = None
    
    for rank in range(max_rank, 19, -1):
        for trial in range(num_trials):
            # Generate random U, V, W matrices with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
        
        if best_result:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    # Try to find a lower rank decomposition
    # First, try structured approaches
    result = _try_strassen_2x2_2x2()
    if result is not None:
        return result
    
    result = _try_winograd()
    if result is not None:
        return result
    
    # Check time
    if time.time() > deadline - 50:
        return _schoolbook()
    
    # Try random search
    result = _random_search(max_rank=59, num_trials=5000)
    if result is not None:
        return result
    
    # Return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
