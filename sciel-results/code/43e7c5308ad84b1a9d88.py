import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Recursive decomposition
    def get_decomposition(n, m, p):
        # Returns U, V, W as lists of lists
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2: (A10+A11)B00 -> C10, C11
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = 1
            
            # M3: A00(B10-B11) -> C01, C11
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[3][2] = 1
            
            # M4: A11(B01-B00) -> C00, C10
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5: (A00+A01)B11 -> -C00, C01
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6: (A10-A00)(B00+B01) -> C11
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7: (A01-A11)(B10+B11) -> C00
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1
            
            return U, V, W
            
        # Base case: 1xm x p or nx1 x p or nxm x 1
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return U, V, W
            
        # Recursive Split
        # Try splitting n, m, p
        # We want to minimize rank.
        # Heuristic: split the largest dimension, or balance the blocks.
        
        best_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomposition(n1, m, p)
                U2, V2, W2 = get_decomposition(n2, m, p)
                
                # Combine
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                total_rank = rank1 + rank2
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # Map U1 to top n1 rows
                    for i in range(n1):
                        for r in range(rank1):
                            U[i*m + (0 if False else 0)][r] = 0 # Placeholder
                    # Actually, U index is i*m + j.
                    # U1 rows are 0..n1*m-1.
                    for i in range(n1*m):
                        for r in range(rank1):
                            U[i][r] = U1[i][r]
                            
                    # U2 rows are n1*m .. n*m-1
                    for i in range(n2*m):
                        for r in range(rank2):
                            U[n1*m + i][rank1 + r] = U2[i][r]
                            
                    # V is same for both
                    for i in range(m*p):
                        for r in range(rank1):
                            V[i][r] = V1[i][r]
                        for r in range(rank2):
                            V[i][rank1 + r] = V2[i][r]
                            
                    # W index is k*n + i.
                    # W1 rows: k*n + i where i < n1.
                    # W2 rows: k*n + i where i >= n1.
                    for k in range(p):
                        for i in range(n1):
                            for r in range(rank1):
                                W[k*n + i][r] = W1[k*n1 + i][r]
                        for i in range(n2):
                            for r in range(rank2):
                                W[k*n + (n1+i)][rank1 + r] = W2[k*n2 + i][r]
                                
                    best_U, best_V, best_W = U, V, W

        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomposition(n, m1, p)
                U2, V2, W2 = get_decomposition(n, m2, p)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                total_rank = rank1 + rank2
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # U index i*m + j.
                    # U1 cols j < m1.
                    # U2 cols j >= m1.
                    for i in range(n):
                        for j in range(m1):
                            for r in range(rank1):
                                U[i*m + j][r] = U1[i*m1 + j][r]
                        for j in range(m2):
                            for r in range(rank2):
                                U[i*m + (m1+j)][rank1 + r] = U2[i*m2 + j][r]
                                
                    # V index j*p + k.
                    # V1 rows j < m1.
                    # V2 rows j >= m1.
                    for j in range(m1):
                        for k in range(p):
                            for r in range(rank1):
                                V[j*p + k][r] = V1[j*p + k][r]
                    for j in range(m2):
                        for k in range(p):
                            for r in range(rank2):
                                V[(m1+j)*p + k][rank1 + r] = V2[j*p + k][r]
                                
                    # W same
                    for i in range(p*n):
                        for r in range(rank1):
                            W[i][r] = W1[i][r]
                        for r in range(rank2):
                            W[i][rank1 + r] = W2[i][r]
                            
                    best_U, best_V, best_W = U, V, W

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomposition(n, m, p1)
                U2, V2, W2 = get_decomposition(n, m, p2)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                total_rank = rank1 + rank2
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # U same
                    for i in range(n*m):
                        for r in range(rank1):
                            U[i][r] = U1[i][r]
                        for r in range(rank2):
                            U[i][rank1 + r] = U2[i][r]
                            
                    # V index j*p + k.
                    # V1 cols k < p1.
                    # V2 cols k >= p1.
                    for j in range(m):
                        for k in range(p1):
                            for r in range(rank1):
                                V[j*p + k][r] = V1[j*p1 + k][r]
                        for k in range(p2):
                            for r in range(rank2):
                                V[j*p + (p1+k)][rank1 + r] = V2[j*p2 + k][r]
                                
                    # W index k*n + i.
                    # W1 rows k < p1.
                    # W2 rows k >= p1.
                    for k in range(p1):
                        for i in range(n):
                            for r in range(rank1):
                                W[k*n + i][r] = W1[k*n + i][r]
                    for k in range(p2):
                        for i in range(n):
                            for r in range(rank2):
                                W[(p1+k)*n + i][rank1 + r] = W2[k*n + i][r]
                                
                    best_U, best_V, best_W = U, V, W

        return best_U, best_V, best_W

    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
