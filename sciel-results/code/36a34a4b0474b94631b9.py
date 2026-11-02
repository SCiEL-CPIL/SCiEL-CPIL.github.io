import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists or numpy arrays.
    # The rank is the number of columns.
    
    # Helper to add a rank-1 term
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # We need to handle the offsets correctly.
    # The decomposition is recursive.
    
    # Let's use a class or just functions returning (U, V, W, rank)
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    def decompose(n, m, p):
        # Returns U, V, W as lists of lists (or sparse dicts if needed, but small so dense is fine)
        # n, m, p are dimensions of this sub-problem.
        
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        if n == 1:
            # Schoolbook for 1 x m x p
            rank = m * p
            U = [[0] * rank for _ in range(m)] # n*m = m
            V = [[0] * rank for _ in range(m * p)] # m*p
            W = [[0] * rank for _ in range(p)] # p*n = p
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j * p + k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W
            
        if m == 1:
            # Schoolbook for n x 1 x p
            rank = n * p
            U = [[0] * rank for _ in range(n)] # n*m = n
            V = [[0] * rank for _ in range(p)] # m*p = p
            W = [[0] * rank for _ in range(p * n)] # p*n
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i][col] = 1
                    V[k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            return U, V, W
            
        if p == 1:
            # Schoolbook for n x m x 1
            rank = n * m
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m)]
            W = [[0] * rank for _ in range(n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1
                    V[j][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W
        
        # Recursive case
        # Strategy: Split m into 2 and m-2 if m >= 2.
        # If m < 2, we would have hit base cases.
        
        if m >= 2:
            # Split m = 2 + (m-2)
            # A = [A1 (n x 2), A2 (n x m-2)]
            # B = [B1; B2] (2 x p, m-2 x p)
            # C = A1 B1 + A2 B2
            
            U1, V1, W1 = decompose(n, 2, p)
            U2, V2, W2 = decompose(n, m-2, p)
            
            # Combine
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill U1 (rows 0..n*2-1)
            for i in range(n):
                for j in range(2):
                    row_idx = i * m + j
                    for c in range(rank1):
                        U[row_idx][c] = U1[i * 2 + j][c]
                        
            # Fill U2 (rows n*2..n*m-1)
            for i in range(n):
                for j in range(m-2):
                    row_idx = i * m + (2 + j)
                    for c in range(rank2):
                        U[row_idx][rank1 + c] = U2[i * (m-2) + j][c]
                        
            # Fill V1 (rows 0..2*p-1)
            for j in range(2):
                for k in range(p):
                    row_idx = j * p + k
                    for c in range(rank1):
                        V[row_idx][c] = V1[j * p + k][c]
                        
            # Fill V2 (rows 2*p..m*p-1)
            for j in range(m-2):
                for k in range(p):
                    row_idx = (2 + j) * p + k
                    for c in range(rank2):
                        V[row_idx][rank1 + c] = V2[j * p + k][c]
                        
            # Fill W (same indices)
            for i in range(n):
                for k in range(p):
                    row_idx = k * n + i
                    for c in range(rank1):
                        W[row_idx][c] = W1[k * n + i][c]
                    for c in range(rank2):
                        W[row_idx][rank1 + c] = W2[k * n + i][c]
                        
            return U, V, W
        else:
            # m == 1 handled above, so this shouldn't happen if m>=2 logic is correct
            # But if m is odd and we split 2, m-2 can be 1.
            # If m-2 == 1, decompose(n, 1, p) is handled.
            pass

    # Special case: If n, m, p are all 2, use Strassen.
    # The recursive split above splits m into 2 and m-2.
    # If m=2, it calls decompose(n, 2, p) and decompose(n, 0, p)??
    # No, if m=2, m-2=0.
    # We need to handle m=2 specifically with Strassen if n=2 and p=2.
    # If n=2, p=2, m=2: Strassen.
    
    # Let's refine the recursion to check for 2x2x2 first.
    
    def decompose_opt(n, m, p):
        if n == 1 or m == 1 or p == 1:
            return decompose(n, m, p)
            
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
            
        # Split m into 2 and m-2
        U1, V1, W1 = decompose_opt(n, 2, p)
        U2, V2, W2 = decompose_opt(n, m-2, p)
        
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(2):
                row_idx = i * m + j
                for c in range(rank1):
                    U[row_idx][c] = U1[i * 2 + j][c]
                    
        for i in range(n):
            for j in range(m-2):
                row_idx = i * m + (2 + j)
                for c in range(rank2):
                    U[row_idx][rank1 + c] = U2[i * (m-2) + j][c]
                    
        for j in range(2):
            for k in range(p):
                row_idx = j * p + k
                for c in range(rank1):
                    V[row_idx][c] = V1[j * p + k][c]
                    
        for j in range(m-2):
            for k in range(p):
                row_idx = (2 + j) * p + k
                for c in range(rank2):
                    V[row_idx][rank1 + c] = V2[j * p + k][c]
                    
        for i in range(n):
            for k in range(p):
                row_idx = k * n + i
                for c in range(rank1):
                    W[row_idx][c] = W1[k * n + i][c]
                for c in range(rank2):
                    W[row_idx][rank1 + c] = W2[k * n + i][c]
                    
        return U, V, W

    def strassen_222():
        # U (4x7), V (4x7), W (4x7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        # a=U[0], d=U[3], e=V[0], h=V[3]
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Col 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[2][0] = -1; W[1][0] = 1 # W[k*n+i]
        # W indices:
        # C11 (i=0,k=0) -> W[0*2+0]=W[0]
        # C12 (i=0,k=1) -> W[1*2+0]=W[2]
        # C21 (i=1,k=0) -> W[0*2+1]=W[1]
        # C22 (i=1,k=1) -> W[1*2+1]=W[3]
        
        # Wait, W convention: W[k*n+i].
        # n=2.
        # i=0, k=0 -> W[0]
        # i=0, k=1 -> W[2]
        # i=1, k=0 -> W[1]
        # i=1, k=1 -> W[3]
        
        # M1 contributes to C11, C22.
        # C11: +1. C22: +1.
        # So W[0]=1, W[3]=1.
        W[0][0] = 1
        W[3][0] = 1
        
        # Col 1: M2 = (c+d)e
        # c=U[2], d=U[3], e=V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21: +1.
        W[1][1] = 1
        
        # Col 2: M3 = a(f-h)
        # a=U[0], f=V[1], h=V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12: +1.
        W[2][2] = 1
        
        # Col 3: M4 = d(g-c)
        # d=U[3], g=V[2], c=V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11: +1, C21: +1.
        W[0][3] = 1
        W[1][3] = 1
        
        # Col 4: M5 = (a+b)h
        # a=U[0], b=U[1], h=V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11: -1, C12: +1.
        W[0][4] = -1
        W[2][4] = 1
        
        # Col 5: M6 = (c-a)(e+f)
        # c=U[2], a=U[0], e=V[0], f=V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22: +1.
        W[3][5] = 1
        
        # Col 6: M7 = (b-d)(g+h)
        # b=U[1], d=U[3], g=V[2], h=V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11: +1.
        W[0][6] = 1
        
        return U, V, W

    U, V, W = decompose_opt(n, m, p)
    return {"U": U, "V": V, "W": W}
