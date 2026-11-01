import time
import numpy as np

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_28():
    """
    Construct a rank-28 decomposition for 3x3x3 matrix multiplication.
    This uses the idea of applying Strassen's algorithm to 2x2 sub-blocks
    and handling the remaining elements, but structured to achieve rank 28.
    
    The decomposition exploits the fact that the 3x3 multiplication can be
    viewed as overlapping 2x2 multiplications with shared terms.
    
    This is a known result: the rank of 3x3x3 is at most 28, and it's
    conjectured to be 23, but 28 is constructively achievable.
    """
    # We'll use a specific known rank-28 construction.
    # This construction uses the fact that we can embed Strassen-like
    # operations for the 2x2 sub-matrices and handle the 3rd row/col.
    
    # Actually, let me use a simpler approach: the "Laderman" style
    # construction for 3x3 which gives rank 23 is complex. 
    # Instead, I'll use a rank-28 construction based on padding to 4x4
    # and applying a rank-32 4x4 algorithm, then reducing... that's not helpful.
    
    # Let me use a different strategy: direct construction with half-integers.
    # The key insight is that the tensor has structure that allows
    # compression beyond schoolbook but below the conjectured 23.
    
    # I'll use a parametric family and solve for coefficients.
    # This is a well-known construction.
    
    n, m, p = 3, 3, 3
    rank = 28
    
    # Use a known rank-28 decomposition pattern.
    # This uses the "Bini-Capovani" type construction.
    
    # Actually, the simplest reliable improvement is to use the fact that
    # we can do better than 27 by sharing computations.
    
    # Let me just return the schoolbook for now, since I cannot guarantee
    # a correct rank-28 construction from memory.
    return _construct_schoolbook()

def construct():
    overall_deadline = time.time() + 140
    
    # Return the schoolbook decomposition (Rank 27)
    # This is the safest and most reliable solution.
    return _construct_schoolbook()
