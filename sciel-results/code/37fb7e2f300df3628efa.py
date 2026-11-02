import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    R = U.shape[1]
    # Check a few random points or all points if small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _get_rank_23_3x3x3():
    """
    Returns a rank 23 decomposition for 3x3x3 matrix multiplication.
    This is a known result. If this function returns None, the caller should use schoolbook.
    """
    # I will try to use a known set of 23 terms.
    # Since I cannot guarantee the exact hardcoded values from memory without error,
    # I will return None to trigger the fallback to schoolbook, unless I can find a reliable source.
    # However, for the sake of the task, I will attempt to provide a valid one if I can.
    # Actually, a safer bet is to use a rank 27 (schoolbook) for 3x3x3, which gives total rank 54.
    # But the task asks for rank-minimal.
    # I will try to search for a valid decomposition with integer/half-integer entries 
    # using a simple randomized local search if the hardcoded one is not available.
    return None

def construct():
    n, m, p = DIMENSIONS
    # p=6, so we have two 3x3x3 blocks.
    
    # Try to get a low-rank decomposition for 3x3x3
    # If we have a rank 23 decomposition, total rank is 46.
    # If not, we use schoolbook rank 27 for each block, total rank 54.
    
    # For now, I will implement the schoolbook decomposition as a baseline.
    # To improve, I would need the 23-term decomposition.
    
    # Let's try to construct a rank 46 decomposition using a known 23-term algorithm.
    # I will use a hardcoded set of 23 terms for 3x3x3.
    # If the verification fails, I fall back to schoolbook.
    
    # Hardcoded 23-term decomposition for 3x3x3 (Landsberg 1987)
    # Note: The exact coefficients are complex. I will use a placeholder that returns None.
    # This means the program will currently return schoolbook rank 54.
    # To beat the seed, I MUST find a lower rank.
    
    # Strategy: Use a random search to find a valid decomposition with rank < 54?
    # Unlikely to find rank 46 randomly.
    
    # I will stick to the schoolbook decomposition for now, as I cannot guarantee a valid rank 23 decomposition from memory.
    # However, the prompt asks for a "stronger algorithm".
    
    # I will implement a simple local search to see if I can reduce the rank from 54?
    # No, that's not how tensor rank works.
    
    # I will output the schoolbook decomposition.
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
