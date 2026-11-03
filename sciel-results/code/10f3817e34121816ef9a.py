"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Decompose 4x4x7 matrix multiplication tensor.
    
    Strategy: Use Kronecker product structure.
    4x4x7 = (2x2) x (2x2) x 7
    rank(4x4x7) <= rank(2x2x7) * rank(2x2x1)
    
    rank(2x2x1) = 4 (schoolbook)
    rank(2x2x7) <= rank(2x2x4) + rank(2x2x3)
    
    For rank(2x2x4): use tensor product rank(2x2x2)^2 = 7*7 = 49? No, that's for 2x2x4 = (2x2)x(2x2)x(2x2)? 
    Actually 2x2x4: m=2, so 2x2x4. rank <= 2*2*4 = 16 schoolbook.
    
    Let me just use: rank(4x4x7) <= rank(2x2x7) * rank(2x2x1)
    where rank(2x2x7) is computed as schoolbook = 28, and rank(2x2x1) = 4.
    Total = 28 * 4 = 112. Same as schoolbook.
    
    Alternative: Direct schoolbook is rank 112. Let me verify the tensor product gives same.
    
    Actually for 4x4x7, let's try: split n=4 as 2*2, keep m=4, p=7.
    rank(4x4x7) <= rank(2x4x7) * rank(2x1x1) ... no, tensor product requires splitting all dims.
    
    The correct tensor product: if M_{n,m,p} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2} where n=n1*n2, etc.
    So 4x4x7 = (2x2x7) ⊗ (2x2x1)? No, 2*2=4, 2*2=4, 7*1=7. Yes!
    
    rank(4x4x7) <= rank(2x2x7) * rank(2x2x1)
    
    rank(2x2x1) = 4 (each of 2*2*1 = 4 terms)
    rank(2x2x7) = 2*2*7 = 28 (schoolbook)
    
    Total = 28 * 4 = 112. Same as direct schoolbook.
    
    To do better, we need a non-trivial decomposition of 2x2x7 or 2x2x1.
    2x2x1 is already minimal at 4.
    2x2x7: can we do better than 28? 2x2x7 with 7=4+3: rank <= rank(2x2x4)+rank(2x2x3) <= 16+12=28. Same.
    
    So the minimum via this decomposition is 112. Let's just implement schoolbook correctly.
    """
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
# EVOLVE-BLOCK-END
