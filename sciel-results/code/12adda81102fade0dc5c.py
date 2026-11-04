import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Memoization cache for (n, m, p) -> (U, V, W, rank)
    cache = {}
    
    def get_decomp(n, m, p):
        if (n, m, p) in cache:
            return cache[(n, m, p)]
        
        # Base case: one of the dimensions is 1
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            result = (U, V, W, rank)
            cache[(n, m, p)] = result
            return result
            
        best_rank = float('inf')
        best_decomp = None
        
        # Try splitting n
        if n > 1:
            # Split n-1, 1
            n1, n2 = n - 1, 1
            U1, V1, W1, r1 = get_decomp(n1, m, p)
            U2, V2, W2, r2 = get_decomp(n2, m, p)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                # Construct new U, V, W
                new_rank = rank
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                # Fill U
                for i in range(n1):
                    for j in range(m):
                        for r in range(r1):
                            new_U[i * m + j][r] = U1[i * m + j][r]
                for i in range(n2):
                    for j in range(m):
                        for r in range(r2):
                            new_U[(n1 + i) * m + j][r1 + r] = U2[i * m + j][r]
                            
                # Fill V
                for j in range(m):
                    for k in range(p):
                        for r in range(r1):
                            new_V[j * p + k][r] = V1[j * p + k][r]
                        for r in range(r2):
                            new_V[j * p + k][r1 + r] = V2[j * p + k][r]
                            
                # Fill W
                for k in range(p):
                    for i in range(n):
                        # W index is k * n + i
                        if i < n1:
                            for r in range(r1):
                                new_W[k * n + i][r] = W1[k * n1 + i][r]
                        else:
                            for r in range(r2):
                                new_W[k * n + i][r1 + r] = W2[k * n2 + (i - n1)][r1 + r - r1] # Wait, W2 index
                                # W2 has rows k * n2 + i2. Here i2 = i - n1.
                                # Col index in new_W is r1 + r.
                                new_W[k * n + i][r1 + r] = W2[k * n2 + (i - n1)][r]
                
                best_decomp = (new_U, new_V, new_W, rank)
                
            # Split n/2, n/2 if even
            if n % 2 == 0:
                n1, n2 = n // 2, n // 2
                U1, V1, W1, r1 = get_decomp(n1, m, p)
                U2, V2, W2, r2 = get_decomp(n2, m, p)
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    new_rank = rank
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    
                    for i in range(n1):
                        for j in range(m):
                            for r in range(r1):
                                new_U[i * m + j][r] = U1[i * m + j][r]
                    for i in range(n2):
                        for j in range(m):
                            for r in range(r2):
                                new_U[(n1 + i) * m + j][r1 + r] = U2[i * m + j][r]
                                
                    for j in range(m):
                        for k in range(p):
                            for r in range(r1):
                                new_V[j * p + k][r] = V1[j * p + k][r]
                            for r in range(r2):
                                new_V[j * p + k][r1 + r] = V2[j * p + k][r]
                                
                    for k in range(p):
                        for i in range(n):
                            if i < n1:
                                for r in range(r1):
                                    new_W[k * n + i][r] = W1[k * n1 + i][r]
                            else:
                                for r in range(r2):
                                    new_W[k * n + i][r1 + r] = W2[k * n2 + (i - n1)][r]
                                    
                    best_decomp = (new_U, new_V, new_W, rank)

        # Try splitting m
        if m > 1:
            # Split m-1, 1
            m1, m2 = m - 1, 1
            U1, V1, W1, r1 = get_decomp(n, m1, p)
            U2, V2, W2, r2 = get_decomp(n, m2, p)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                new_rank = rank
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m1):
                        for r in range(r1):
                            new_U[i * m + j][r] = U1[i * m1 + j][r]
                    for j in range(m2):
                        for r in range(r2):
                            new_U[i * m + (m1 + j)][r1 + r] = U2[i * m2 + j][r]
                            
                for j in range(m):
                    for k in range(p):
                        if j < m1:
                            for r in range(r1):
                                new_V[j * p + k][r] = V1[j * p + k][r]
                        else:
                            for r in range(r2):
                                new_V[j * p + k][r1 + r] = V2[(j - m1) * p + k][r]
                                
                for k in range(p):
                    for i in range(n):
                        for r in range(r1):
                            new_W[k * n + i][r] = W1[k * n + i][r]
                        for r in range(r2):
                            new_W[k * n + i][r1 + r] = W2[k * n + i][r]
                            
                best_decomp = (new_U, new_V, new_W, rank)
                
            # Split m/2, m/2 if even
            if m % 2 == 0:
                m1, m2 = m // 2, m // 2
                U1, V1, W1, r1 = get_decomp(n, m1, p)
                U2, V2, W2, r2 = get_decomp(n, m2, p)
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    new_rank = rank
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m1):
                            for r in range(r1):
                                new_U[i * m + j][r] = U1[i * m1 + j][r]
                        for j in range(m2):
                            for r in range(r2):
                                new_U[i * m + (m1 + j)][r1 + r] = U2[i * m2 + j][r]
                                
                    for j in range(m):
                        for k in range(p):
                            if j < m1:
                                for r in range(r1):
                                    new_V[j * p + k][r] = V1[j * p + k][r]
                            else:
                                for r in range(r2):
                                    new_V[j * p + k][r1 + r] = V2[(j - m1) * p + k][r]
                                    
                    for k in range(p):
                        for i in range(n):
                            for r in range(r1):
                                new_W[k * n + i][r] = W1[k * n + i][r]
                            for r in range(r2):
                                new_W[k * n + i][r1 + r] = W2[k * n + i][r]
                                
                    best_decomp = (new_U, new_V, new_W, rank)

        # Try splitting p
        if p > 1:
            # Split p-1, 1
            p1, p2 = p - 1, 1
            U1, V1, W1, r1 = get_decomp(n, m, p1)
            U2, V2, W2, r2 = get_decomp(n, m, p2)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                new_rank = rank
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for r in range(r1):
                            new_U[i * m + j][r] = U1[i * m + j][r]
                        for r in range(r2):
                            new_U[i * m + j][r1 + r] = U2[i * m + j][r]
                            
                for j in range(m):
                    for k in range(p1):
                        for r in range(r1):
                            new_V[j * p + k][r] = V1[j * p1 + k][r]
                    for k in range(p2):
                        for r in range(r2):
                            new_V[j * p + (p1 + k)][r1 + r] = V2[j * p2 + k][r]
                            
                for k in range(p):
                    for i in range(n):
                        if k < p1:
                            for r in range(r1):
                                new_W[k * n + i][r] = W1[k * n + i][r]
                        else:
                            for r in range(r2):
                                new_W[k * n + i][r1 + r] = W2[(k - p1) * n + i][r]
                                
                best_decomp = (new_U, new_V, new_W, rank)
                
            # Split p/2, p/2 if even
            if p % 2 == 0:
                p1, p2 = p // 2, p // 2
                U1, V1, W1, r1 = get_decomp(n, m, p1)
                U2, V2, W2, r2 = get_decomp(n, m, p2)
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    new_rank = rank
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m):
                            for r in range(r1):
                                new_U[i * m + j][r] = U1[i * m + j][r]
                            for r in range(r2):
                                new_U[i * m + j][r1 + r] = U2[i * m + j][r]
                                
                    for j in range(m):
                        for k in range(p1):
                            for r in range(r1):
                                new_V[j * p + k][r] = V1[j * p1 + k][r]
                        for k in range(p2):
                            for r in range(r2):
                                new_V[j * p + (p1 + k)][r1 + r] = V2[j * p2 + k][r]
                                
                    for k in range(p):
                        for i in range(n):
                            if k < p1:
                                for r in range(r1):
                                    new_W[k * n + i][r] = W1[k * n + i][r]
                            else:
                                for r in range(r2):
                                    new_W[k * n + i][r1 + r] = W2[(k - p1) * n + i][r]
                                    
                    best_decomp = (new_U, new_V, new_W, rank)

        if best_decomp is None:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            best_decomp = (U, V, W, rank)
            
        cache[(n, m, p)] = best_decomp
        return best_decomp

    U, V, W, rank = get_decomp(n, m, p)
    
    if time.time() > deadline:
        # Return schoolbook if time runs out
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
