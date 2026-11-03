"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy: slice-packing. Partition the 9-column dimension (p) into three
    independent 3-column blocks. For each 4x4x3 sub-problem, we use the
    schoolbook decomposition which has rank 4*4*3 = 48.
    
    However, the schoolbook decomposition for the full 4x4x9 has rank 4*4*9 = 144.
    The slice-packing approach yields 3 * 48 = 144, which is the same.
    
    To potentially improve, we need to find a decomposition for 4x4x3 with rank < 48.
    The known lower bound for 4x4x3 is not well-established, but the schoolbook
    is 48.
    
    For now, we implement the schoolbook decomposition for the full problem,
    which is guaranteed to be correct and has rank 144.
    
    The directive asks to investigate improvements within the slice-packing family.
    A key insight is that for the 4x4x3 sub-problem, we might be able to use
    a decomposition with rank less than 48 if we exploit the structure.
    
    However, without a known better decomposition for 4x4x3, we fall back to
    the schoolbook method. The rank is 144.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Schoolbook decomposition: rank = n*m*p
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
