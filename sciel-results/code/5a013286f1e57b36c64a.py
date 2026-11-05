import time
import sys

DIMENSIONS = [4, 5, 6]
_start_time = time.time()
_DEADLINE = 140  # seconds, leave 10s margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    memo = {}
    
    def get_decomp(n, m, p):
        if time.time() - _start_time > _DEADLINE:
            return schoolbook(n, m, p)
        
        key = (n, m, p)
        if key in memo:
            return memo[key]
        
        # Base cases
        if n == 2 and m == 2 and p == 2:
            result = strassen_2x2x2()
            memo[key] = result
            return result
        
        if n * m * p <= 8:
            result = schoolbook(n, m, p)
            memo[key] = result
            return result
        
        best = None
        best_rank = n * m * p  # schoolbook rank as upper bound
        
        # Try splitting m
        if m >= 2:
            for m1 in range(1, m):
                if time.time() - _start_time > _DEADLINE:
                    break
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                r = len(U1[0]) + len(U2[0])
                if r < best_rank:
                    U, V, W = combine_m_split(n, m, p, m1, U1, V1, W1, U2, V2, W2)
                    best = (U, V, W)
                    best_rank = r
        
        # Try splitting n
        if n >= 2:
            for n1 in range(1, n):
                if time.time() - _start_time > _DEADLINE:
                    break
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                r = len(U1[0]) + len(U2[0])
                if r < best_rank:
                    U, V, W = combine_n_split(n, m, p, n1, U1, V1, W1, U2, V2, W2)
                    best = (U, V, W)
                    best_rank = r
        
        # Try splitting p
        if p >= 2:
            for p1 in range(1, p):
                if time.time() - _start_time > _DEADLINE:
                    break
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                r = len(U1[0]) + len(U2[0])
                if r < best_rank:
                    U, V, W = combine_p_split(n, m, p, p1, U1, V1, W1, U2, V2, W2)
                    best = (U, V, W)
                    best_rank = r
        
        if best is not None:
            memo[key] = best
            return best
        
        result = schoolbook(n, m, p)
        memo[key] = result
        return result
    
    def combine_m_split(n, m, p, m1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        m2 = m - m1
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i*m + (m1+j)][r1+c] = U2[i*m2 + j][c]
        
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1+j)*p + k][r1+c] = V2[j*p + k][c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
                for c in range(r2):
                    W[k*n + i][r1+c] = W2[k*n + i][c]
        
        return U, V, W
    
    def combine_n_split(n, m, p, n1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        n2 = n - n1
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1+i)*m + j][r1+c] = U2[i*m + j][c]
        
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1+c] = V2[j*p + k][c]
        
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k*n + (n1+i)][r1+c] = W2[k*n2 + i][c]
        
        return U, V, W
    
    def combine_p_split(n, m, p, p1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        p2 = p - p1
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
                for c in range(r2):
                    U[i*m + j][r1+c] = U2[i*m + j][c]
        
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j*p + (p1+k)][r1+c] = V2[j*p2 + k][c]
        
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[(p1+k)*n + i][r1+c] = W2[k*n + i][c]
        
        return U, V, W
    
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        
        V[0][0] = 1; V[3][0] = 1
        V[0][1] = 1
        V[2][2] = 1; V[3][2] = -1
        V[1][3] = 1; V[0][3] = -1
        V[3][4] = 1
        V[0][5] = 1; V[1][5] = 1
        V[2][6] = 1; V[3][6] = 1
        
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        W[1][2] = 1; W[1][4] = 1
        W[2][1] = 1; W[2][3] = 1
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        
        return U, V, W
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W
    
    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
