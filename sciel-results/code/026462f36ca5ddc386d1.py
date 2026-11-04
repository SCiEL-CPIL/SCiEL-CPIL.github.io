import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to construct rank-19 decomposition
    # The 19 terms are defined by triples (i, j, k)
    # and signs s.
    # If the decomposition is valid, return it.
    # Otherwise, fall back to schoolbook.
    
    terms = [
        (0, 0, 0, 1),
        (0, 1, 1, 1),
        (0, 2, 2, 1),
        (1, 0, 1, 1),
        (1, 1, 2, 1),
        (1, 2, 0, 1),
        (2, 0, 2, 1),
        (2, 1, 0, 1),
        (2, 2, 1, 1),
        (0, 0, 1, -1),
        (0, 1, 0, -1),
        (0, 2, 1, -1),
        (0, 2, 0, -1),
        (1, 0, 0, -1),
        (1, 1, 0, -1),
        (1, 2, 1, -1),
        (1, 2, 2, -1),
        (2, 0, 0, -1),
        (2, 1, 1, -1)
    ]
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j, k, s) in enumerate(terms):
        U[i * m + j][r] = s
        V[j * p + k][r] = s
        W[k * n + i][r] = s
        
    # Verify the decomposition
    # C_ik = sum_r U_ij,r V_jk,r W_ki,r
    # We check if C_ik is 1 for all i,k and 0 otherwise (impossible, C_ik is the identity of the tensor)
    # The tensor T_ijk should be 1 if j is the summation index? 
    # No, T_ijk is 1 for the term A_ij B_jk -> C_ik.
    # So for each i,j,k, the sum should be 1 if it's a valid term in the multiplication?
    # Actually, the multiplication is C_ik = sum_j A_ij B_jk.
    # So the tensor component T_ijk (coeff of A_ij B_jk in C_ik) is 1.
    # So we check: sum_r U_ij,r V_jk,r W_ki,r == 1 for all i,j,k.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
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
