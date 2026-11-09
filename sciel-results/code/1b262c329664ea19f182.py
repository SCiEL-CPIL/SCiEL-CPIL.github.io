import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
class TensorDecomposition:
    def __init__(self):
        self.cache = {}
        self.deadline = time.time() + 110

    def get_rank(self, n, m, p):
        if (n, m, p) in self.cache:
            return self.cache[(n, m, p)]
        
        best_rank = n * m * p
        
        # Try splitting n
        for i in range(1, n):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting m
        for j in range(1, m):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting p
        for k in range(1, p):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            best_rank = min(best_rank, r1 + r2)
            
        self.cache[(n, m, p)] = best_rank
        return best_rank

    def construct(self, n, m, p):
        if time.time() > self.deadline:
             rank = n * m * p
             U = [[0] * rank for _ in range(n * m)]
             V = [[0] * rank for _ in range(m * p)]
             W = [[0] * rank for _ in range(p * n)]
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j][col] = 1
                         V[j*p+k][col] = 1
                         W[k*n+i][col] = 1
                         col += 1
             return U, V, W, rank

        best_rank = n * m * p
        best_split = None
        
        for i in range(1, n):
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('n', i)
                
        for j in range(1, m):
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('m', j)
                
        for k in range(1, p):
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('p', k)

        if best_split is None:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W, rank
        else:
            dim, idx = best_split
            if dim == 'n':
                n1, n2 = idx, n - idx
                U1, V1, W1, r1 = self.construct(n1, m, p)
                U2, V2, W2, r2 = self.construct(n2, m, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(n1+i)*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                        for c in range(r2):
                            V[j*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n1):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n1+i][c]
                    for i in range(n2):
                        for c in range(r2):
                            W[k*n + (n1+i)][r1+c] = W2[k*n2+i][c]
                            
                return U, V, W, r1 + r2
            elif dim == 'm':
                m1, m2 = idx, m - idx
                U1, V1, W1, r1 = self.construct(n, m1, p)
                U2, V2, W2, r2 = self.construct(n, m2, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m1):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m1+j][c]
                    for j in range(m2):
                        for c in range(r2):
                            U[i*m + (m1+j)][r1+c] = U2[i*m2+j][c]
                            
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(m1+j)*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                        for c in range(r2):
                            W[k*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2
            else:
                p1, p2 = idx, p - idx
                U1, V1, W1, r1 = self.construct(n, m, p1)
                U2, V2, W2, r2 = self.construct(n, m, p2)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                        for c in range(r2):
                            U[i*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p1):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p1+k][c]
                    for k in range(p2):
                        for c in range(r2):
                            V[j*p + (p1+k)][r1+c] = V2[j*p2+k][c]
                            
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(p1+k)*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2

def construct():
    n, m, p = DIMENSIONS
    solver = TensorDecomposition()
    U, V, W, rank = solver.construct(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
