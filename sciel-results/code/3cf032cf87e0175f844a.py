import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 41 decomposition
    # 23 terms for 3x3x3
    # 18 terms for 3x3x2 (schoolbook)
    rank = 41
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 3x3x3 decomposition (Rank 23)
    # Coefficients from Landsberg's "Tensors: Geometry and Applications" or similar
    # The terms are (u, v, w) where u, v, w are 3x3 matrices flattened.
    # I will use a known set of 23 terms.
    # Note: The specific coefficients must satisfy sum u_c v_c w_c^T = T_333
    
    # Since I don't have the exact 23 terms memorized perfectly, 
    # I will use a fallback: 
    # If I can't guarantee the 23 terms, I should stick to a valid decomposition.
    # However, 45 is the seed. 41 is the target.
    # I will hardcode the 23 terms from a trusted source if possible, 
    # otherwise I will use a simpler improvement.
    
    # Actually, there is a simpler improvement:
    # Use Strassen's 7-term for 2x2x2 blocks?
    # 3x3x5 can be decomposed into blocks.
    # This is complex.
    
    # Let's try to construct a rank 41 solution using the known 23-term 3x3x3.
    # I will insert the 23 terms here.
    
    # For the sake of the code, I will define the 23 terms.
    # (i,j,k) -> coefficient
    # The 23 terms for 3x3x3 are known to have integer coefficients.
    
    # If I cannot recall them, I will use the schoolbook for 3x3x3 (27) + schoolbook for 3x3x2 (18) = 45.
    # This is not an improvement.
    
    # Let's try to find a rank < 45 decomposition using a simple heuristic in the code?
    # No, must be exact.
    
    # I will use the 23-term decomposition.
    # Terms:
    # 1. (1,1,1) at (0,0,0) etc? No.
    
    # I will use a known Python implementation of the 3x3x3 rank 23 decomposition if I can generate it.
    # Since I cannot, I will use the following:
    # The rank of 3x3x3 is 23.
    # I will hardcode the 23 terms from the "Strassen" generalization or "Landsberg".
    
    # If I fail, I return 45.
    
    # Let's assume I can find the 23 terms.
    # I will define them as a list of (u, v, w) tuples.
    
    # Actually, I can use the fact that 3x3x5 is 3x3x3 + 3x3x2.
    # I will use the 23-term 3x3x3.
    
    # I will write the code to use the 23 terms.
    
    # The 23 terms are:
    # (u, v, w)
    # ...
    
    # Since I don't have them, I will use the schoolbook 45.
    # But I want to improve.
    
    # Alternative: Use rank 38?
    # I will stick to 45 to be safe, as I cannot guarantee the 23 terms.
    
    # Wait, I can use the 2x2x2 Strassen (rank 7) to build a 3x3x3?
    # No, 3x3x3 rank 23 is the best known.
    
    # I will return the schoolbook 45.
    
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
