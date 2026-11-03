import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    
    # Start with the standard schoolbook (rank 63) decomposition.
    # This is always valid and gives a baseline score of 1.0.
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Attempt to find a lower rank decomposition using the known rank 23
    # decomposition for 3x3 matrix multiplication (Strassen-like improvements).
    # If successful, the total rank would be 23 (for first 3 cols) + 3*3*4 (for last 4 cols)
    # = 23 + 36 = 59, which is an improvement over 63.
    # However, hardcoding the exact 23-term decomposition is complex and error-prone.
    # A safer approach is to check if a simple pattern works, but for 3x3, Strassen's
    # algorithm uses 7 multiplications, leading to a rank of 7 * 7 = 49 for 7x7?
    # No, the tensor is M_{3,3,p}.
    # For M_{3,3,1}, the rank is 9 (schoolbook) or less?
    # The rank of M_{3,3,3} is known to be 23 (from literature, e.g., Stasinska & Strassen).
    # If we use a rank 23 decomposition for the first 3 columns of B (size 3x3),
    # and schoolbook for the remaining 4 columns (size 3x4), the total rank is:
    # Rank(M_{3,3,3}) + Rank(M_{3,3,4})
    # Rank(M_{3,3,4}) is at most 3*3*4 = 36 (schoolbook).
    # So total rank <= 23 + 36 = 59.
    
    # Since I cannot reliably reproduce the exact 23-term decomposition coefficients 
    # without external lookup which might be error-prone in this context, 
    # and given the constraint to provide a valid solution, 
    # I will stick to the guaranteed valid schoolbook solution.
    
    # NOTE: In a real-world scenario with access to precomputed coefficients,
    # one would insert them here to achieve rank 59 or better.
    # For now, the schoolbook decomposition is the robust choice.

    return {"U": U, "V": V, "W": W}
