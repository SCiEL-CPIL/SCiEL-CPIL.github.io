import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = build_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}

def build_decomp(n, m, p):
    """
    Recursively builds the tensor decomposition for M_{n,m,p}.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank).
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W

    # Base case: One dimension is 1
    if n == 1 or m == 1 or p == 1:
        return build_direct(n, m, p)

    # Base case: 2x2x2 (Strassen)
    if n == 2 and m == 2 and p == 2:
        return build_strassen()

    # Recursive step: Partition the largest dimension
    # Determine which dimension to split
    # Heuristic: Split the largest dimension into 2 and the rest
    # If dimension is 3, split 2 and 1.
    # If dimension is 5, split 2 and 3 (which splits 2 and 1).
    
    # We choose the dimension with the largest value to split.
    # If tie, prefer m, then p, then n? 
    # In our analysis:
    # 3,5,5 -> split m (5) -> 3,2,5 and 3,3,5? No, 3,2,5 and 3,2,5 and 3,1,5?
    # My previous analysis split 5 into 2,2,1.
    # If I split 5 into 2 and 3:
    # M_{3,5,5} -> M_{3,2,5} + M_{3,3,5}
    # M_{3,3,5} -> split p (5) -> M_{3,3,2} + M_{3,3,3}
    # M_{3,3,3} -> split n (3) -> M_{2,3,3} + M_{1,3,3}
    # This path might yield different ranks.
    
    # To match the 71 rank derivation, I should split 5 into 2, 2, 1.
    # This implies splitting 5 into 2 and 3, then 3 into 2 and 1.
    # So splitting the largest dimension into 2 and (D-2) is consistent.
    
    max_dim = max(n, m, p)
    
    if m == max_dim:
        # Split m into m1, m2
        m1 = 2
        m2 = m - 2
        if m2 < 1:
            m1 = 1
            m2 = m - 1
            
        U1, V1, W1 = build_decomp(n, m1, p)
        U2, V2, W2 = build_decomp(n, m2, p)
        
        # Embed
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Fill part 1 (j from 0 to m1-1)
        for i in range(n):
            for j in range(m1):
                for r in range(len(U1[0])):
                    U[i * m + j][r] = U1[i * m1 + j][r]
        
        for j in range(m1):
            for k in range(p):
                for r in range(len(V1[0])):
                    V[j * p + k][r] = V1[j * p + k][r]
        
        for r in range(len(W1[0])):
            for idx in range(len(W1)):
                W[idx][r] = W1[idx][r]
        
        # Fill part 2 (j from m1 to m-1)
        offset_r = len(U1[0])
        for i in range(n):
            for j in range(m2):
                for r in range(len(U2[0])):
                    U[i * m + (m1 + j)][offset_r + r] = U2[i * m2 + j][r]
        
        for j in range(m2):
            for k in range(p):
                for r in range(len(V2[0])):
                    V[(m1 + j) * p + k][offset_r + r] = V2[j * p + k][r]
        
        for r in range(len(W2[0])):
            for idx in range(len(W2)):
                W[idx][offset_r + r] = W2[idx][r]
                
        return U, V, W

    elif p == max_dim:
        # Split p into p1, p2
        p1 = 2
        p2 = p - 2
        if p2 < 1:
            p1 = 1
            p2 = p - 1
            
        U1, V1, W1 = build_decomp(n, m, p1)
        U2, V2, W2 = build_decomp(n, m, p2)
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Fill part 1 (k from 0 to p1-1)
        for i in range(n):
            for j in range(m):
                for r in range(len(U1[0])):
                    U[i * m + j][r] = U1[i * m + j][r]
        
        for j in range(m):
            for k in range(p1):
                for r in range(len(V1[0])):
                    V[j * p + k][r] = V1[j * p1 + k][r]
        
        for k in range(p1):
            for i in range(n):
                for r in range(len(W1[0])):
                    W[k * n + i][r] = W1[k * n + i][r]
        
        # Fill part 2 (k from p1 to p-1)
        offset_r = len(U1[0])
        for i in range(n):
            for j in range(m):
                for r in range(len(U2[0])):
                    U[i * m + j][offset_r + r] = U2[i * m + j][r]
        
        for j in range(m):
            for k in range(p2):
                for r in range(len(V2[0])):
                    V[j * p + (p1 + k)][offset_r + r] = V2[j * p2 + k][r]
        
        for k in range(p2):
            for i in range(n):
                for r in range(len(W2[0])):
                    W[(p1 + k) * n + i][offset_r + r] = W2[k * n + i][r]
                    
        return U, V, W

    else: # n == max_dim
        # Split n into n1, n2
        n1 = 2
        n2 = n - 2
        if n2 < 1:
            n1 = 1
            n2 = n - 1
            
        U1, V1, W1 = build_decomp(n1, m, p)
        U2, V2, W2 = build_decomp(n2, m, p)
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Fill part 1 (i from 0 to n1-1)
        for i in range(n1):
            for j in range(m):
                for r in range(len(U1[0])):
                    U[i * m + j][r] = U1[i * m + j][r]
        
        for j in range(m):
            for k in range(p):
                for r in range(len(V1[0])):
                    V[j * p + k][r] = V1[j * p + k][r]
        
        for k in range(p):
            for i in range(n1):
                for r in range(len(W1[0])):
                    W[k * n + i][r] = W1[k * n1 + i][r]
        
        # Fill part 2 (i from n1 to n-1)
        offset_r = len(U1[0])
        for i in range(n2):
            for j in range(m):
                for r in range(len(U2[0])):
                    U[(n1 + i) * m + j][offset_r + r] = U2[i * m + j][r]
        
        for j in range(m):
            for k in range(p):
                for r in range(len(V2[0])):
                    V[j * p + k][offset_r + r] = V2[j * p + k][r]
        
        for k in range(p):
            for i in range(n2):
                for r in range(len(W2[0])):
                    W[k * n + (n1 + i)][offset_r + r] = W2[k * n2 + i][r]
                    
        return U, V, W

def build_direct(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def build_strassen():
    # n=2, m=2, p=2
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3 = A11(B12-B22)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1
    
    # M4 = A22(B21-B11)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[3][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
