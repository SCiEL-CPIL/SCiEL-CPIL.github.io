import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0

    # Memoization cache
    cache = {}

    def schoolbook(na, ma, pa):
        rank = na * ma * pa
        U = [[0]*rank for _ in range(na*ma)]
        V = [[0]*rank for _ in range(ma*pa)]
        W = [[0]*rank for _ in range(pa*na)]
        col = 0
        for i in range(na):
            for j in range(ma):
                for k in range(pa):
                    U[i*ma+j][col] = 1
                    V[j*pa+k][col] = 1
                    W[k*na+i][col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1
        
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1
        
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[1][3]=1; W[0][3]=-1
        
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=1; W[2][4]=1
        
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[2][5]=1
        W[3][5]=1
        
        U[1][6]=1; U[3][6]=-1
        V[1][6]=1; V[3][6]=1
        W[1][6]=-1; W[3][6]=1
        
        return U, V, W, 7

    def combine_split_n(n1, n2, ma, pa, U1, V1, W1, r1, U2, V2, W2, r2):
        total_rank = r1 + r2
        na = n1 + n2
        U = [[0]*total_rank for _ in range(na*ma)]
        V = [[0]*total_rank for _ in range(ma*pa)]
        W = [[0]*total_rank for _ in range(pa*na)]
        
        for i in range(n1):
            for j in range(ma):
                for r in range(r1):
                    U[i*ma+j][r] = U1[i*ma+j][r]
        for i in range(n2):
            for j in range(ma):
                for r in range(r2):
                    U[(n1+i)*ma+j][r1+r] = U2[i*ma+j][r]
        
        for j in range(ma):
            for k in range(pa):
                for r in range(r1):
                    V[j*pa+k][r] = V1[j*pa+k][r]
                for r in range(r2):
                    V[j*pa+k][r1+r] = V2[j*pa+k][r]
        
        for k in range(pa):
            for i in range(n1):
                for r in range(r1):
                    W[k*na+i][r] = W1[k*n1+i][r]
            for i in range(n2):
                for r in range(r2):
                    W[k*na+(n1+i)][r1+r] = W2[k*n2+i][r]
        
        return U, V, W, total_rank

    def combine_split_m(na, m1, m2, pa, U1, V1, W1, r1, U2, V2, W2, r2):
        total_rank = r1 + r2
        ma = m1 + m2
        U = [[0]*total_rank for _ in range(na*ma)]
        V = [[0]*total_rank for _ in range(ma*pa)]
        W = [[0]*total_rank for _ in range(pa*na)]
        
        for i in range(na):
            for j in range(m1):
                for r in range(r1):
                    U[i*ma+j][r] = U1[i*m1+j][r]
            for j in range(m2):
                for r in range(r2):
                    U[i*ma+(m1+j)][r1+r] = U2[i*m2+j][r]
        
        for j in range(m1):
            for k in range(pa):
                for r in range(r1):
                    V[j*pa+k][r] = V1[j*pa+k][r]
        for j in range(m2):
            for k in range(pa):
                for r in range(r2):
                    V[(m1+j)*pa+k][r1+r] = V2[j*pa+k][r]
        
        for k in range(pa):
            for i in range(na):
                for r in range(r1):
                    W[k*na+i][r] = W1[k*na+i][r]
                for r in range(r2):
                    W[k*na+i][r1+r] = W2[k*na+i][r]
        
        return U, V, W, total_rank

    def combine_split_p(na, ma, p1, p2, U1, V1, W1, r1, U2, V2, W2, r2):
        total_rank = r1 + r2
        pa = p1 + p2
        U = [[0]*total_rank for _ in range(na*ma)]
        V = [[0]*total_rank for _ in range(ma*pa)]
        W = [[0]*total_rank for _ in range(pa*na)]
        
        for i in range(na):
            for j in range(ma):
                for r in range(r1):
                    U[i*ma+j][r] = U1[i*ma+j][r]
                for r in range(r2):
                    U[i*ma+j][r1+r] = U2[i*ma+j][r]
        
        for j in range(ma):
            for k in range(p1):
                for r in range(r1):
                    V[j*pa+k][r] = V1[j*p1+k][r]
            for k in range(p2):
                for r in range(r2):
                    V[j*pa+(p1+k)][r1+r] = V2[j*p2+k][r]
        
        for k in range(p1):
            for i in range(na):
                for r in range(r1):
                    W[k*na+i][r] = W1[k*na+i][r]
        for k in range(p2):
            for i in range(na):
                for r in range(r2):
                    W[(p1+k)*na+i][r1+r] = W2[k*na+i][r]
        
        return U, V, W, total_rank

    def decompose(na, ma, pa):
        key = (na, ma, pa)
        if key in cache:
            return cache[key]
        
        # Base cases
        if na == 1 and ma == 1 and pa == 1:
            result = schoolbook(1, 1, 1)
            cache[key] = result
            return result
        
        if na == 2 and ma == 2 and pa == 2:
            result = strassen_2x2x2()
            cache[key] = result
            return result
        
        # If any dimension is 1, can't use Strassen, use schoolbook or split others
        best_rank = na * ma * pa
        best = None
        
        # Try splitting n
        for n1 in range(1, na):
            n2 = na - n1
            U1, V1, W1, r1 = decompose(n1, ma, pa)
            U2, V2, W2, r2 = decompose(n2, ma, pa)
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                U, V, W, _ = combine_split_n(n1, n2, ma, pa, U1, V1, W1, r1, U2, V2, W2, r2)
                best = (U, V, W, total_rank)
        
        # Try splitting m
        for m1 in range(1, ma):
            m2 = ma - m1
            U1, V1, W1, r1 = decompose(na, m1, pa)
            U2, V2, W2, r2 = decompose(na, m2, pa)
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                U, V, W, _ = combine_split_m(na, m1, m2, pa, U1, V1, W1, r1, U2, V2, W2, r2)
                best = (U, V, W, total_rank)
        
        # Try splitting p
        for p1 in range(1, pa):
            p2 = pa - p1
            U1, V1, W1, r1 = decompose(na, ma, p1)
            U2, V2, W2, r2 = decompose(na, ma, p2)
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                U, V, W, _ = combine_split_p(na, ma, p1, p2, U1, V1, W1, r1, U2, V2, W2, r2)
                best = (U, V, W, total_rank)
        
        if best is None:
            best = schoolbook(na, ma, pa)
        
        cache[key] = best
        return best

    # Get the decomposition
    U, V, W, rank = decompose(n, m, p)
    
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    expected = 1 if j == k else 0
                    if abs(s - expected) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
