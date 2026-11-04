import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to combine decompositions
    # U1, V1, W1 for M_{n1, m1, p1}
    # U2, V2, W2 for M_{n2, m2, p2}
    # Returns U, V, W for M_{n, m, p} where n=n1+n2 (if splitting n), etc.
    
    def combine_n(U1, V1, W1, U2, V2, W2, n1, m, p):
        """
        Combine two decompositions for M_{n1, m, p} and M_{n2, m, p} into M_{n, m, p}
        n = n1 + n2
        U1: (n1*m, r1), V1: (m*p, r1), W1: (p*n1, r1)
        U2: (n2*m, r2), V2: (m*p, r2), W2: (p*n2, r2)
        Result U: (n*m, r1+r2)
        Result V: (m*p, r1+r2)
        Result W: (p*n, r1+r2)
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Fill U
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1+i)*m + j][r1+c] = U2[i*m + j][c]
                    
        # Fill V
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1+c] = V2[j*p + k][c]
                    
        # Fill W
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k*n + (n1+i)][r1+c] = W2[k*n2 + i][c]
                    
        return U, V, W

    def combine_m(U1, V1, W1, U2, V2, W2, n, m1, p):
        """
        Combine two decompositions for M_{n, m1, p} and M_{n, m2, p} into M_{n, m, p}
        m = m1 + m2
        U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
        U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
        Result U: (n*m, r1+r2)
        Result V: (m*p, r1+r2)
        Result W: (p*n, r1+r2)
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        m2 = m - m1
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Fill U
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i*m + (m1+j)][r1+c] = U2[i*m2 + j][c]
                    
        # Fill V
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c] if False else V1[j*p + k][c] # V1 is m1*p
            # Correction: V1 index is j*p + k where j < m1
            # V2 index is j*p + k where j < m2, global j is m1+j
            
        # Re-implement V filling carefully
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1+j)*p + k][r1+c] = V2[j*p + k][c]
                    
        # Fill W
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
                for c in range(r2):
                    W[k*n + i][r1+c] = W2[k*n + i][c]
                    
        return U, V, W

    def combine_p(U1, V1, W1, U2, V2, W2, n, m, p1):
        """
        Combine two decompositions for M_{n, m, p1} and M_{n, m, p2} into M_{n, m, p}
        p = p1 + p2
        U1: (n*m, r1), V1: (m*p1, r1), W1: (p1*n, r1)
        U2: (n*m, r2), V2: (m*p2, r2), W2: (p2*n, r2)
        Result U: (n*m, r1+r2)
        Result V: (m*p, r1+r2)
        Result W: (p*n, r1+r2)
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        p2 = p - p1
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Fill U
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
                for c in range(r2):
                    U[i*m + j][r1+c] = U2[i*m + j][c]
                    
        # Fill V
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j*p + (p1+k)][r1+c] = V2[j*p2 + k][c]
                    
        # Fill W
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
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2 = (A10+A11)B00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=-1
        
        # M3 = A00(B10-B11)
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1
        
        # M4 = A11(B01-B00)
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        
        # M5 = (A00+A01)B11
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        return U, V, W

    def schoolbook(n, m, p):
        rank = n*m*p
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

    def get_decomp(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return schoolbook(1,1,1)
        if n == 1:
            return schoolbook(1, m, p)
        if m == 1:
            return schoolbook(n, 1, p)
        if p == 1:
            return schoolbook(n, m, 1)
        
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
            
        # Recursive splitting
        # Heuristic: Split the largest dimension, or split to create 2s.
        # For 4,5,6:
        # Split n=4 into 2,2
        # Split p=6 into 3,3
        # Split m=5 into 2,3
        # Split p=3 into 2,1
        # Split m=3 into 2,1
        
        if n >= 2:
            n1 = 2
            n2 = n - 2
            if n2 < 1: n1 = n//2; n2 = n-n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            return combine_n(U1, V1, W1, U2, V2, W2, n1, m, p)
            
        if m >= 2:
            m1 = 2
            m2 = m - 2
            if m2 < 1: m1 = m//2; m2 = m-m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            return combine_m(U1, V1, W1, U2, V2, W2, n, m1, p)
            
        if p >= 2:
            p1 = 2
            p2 = p - 2
            if p2 < 1: p1 = p//2; p2 = p-p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            return combine_p(U1, V1, W1, U2, V2, W2, n, m, p1)
            
        return schoolbook(n, m, p)

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
