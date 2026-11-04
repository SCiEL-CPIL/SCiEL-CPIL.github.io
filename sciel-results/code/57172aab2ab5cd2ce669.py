import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p):
        """
        Returns U, V, W for M_{n,m,p}.
        U: list of lists, shape (n*m, rank)
        V: list of lists, shape (m*p, rank)
        W: list of lists, shape (p*n, rank)
        """
        # Base Cases
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        if n == 2 and m == 2 and p == 1:
            return schoolbook(2, 2, 1)
        if n == 2 and m == 1 and p == 3:
            return schoolbook(2, 1, 3)
            
        # Recursive Splits
        # Strategy:
        # 1. If n > 2, split n into 2 and n-2 (or balanced)
        # 2. If m > 2, split m into 2 and m-2 (or 1 and m-1)
        # 3. If p > 2, split p into 2 and p-2 (or 3 and p-3)
        
        # We want to hit the specific efficient sub-blocks.
        
        if n == 4 and m == 2 and p == 6:
            # Split n into 2, 2
            U1, V1, W1 = get_decomp(2, 2, 6)
            U2, V2, W2 = get_decomp(2, 2, 6)
            return combine_n(U1, V1, W1, U2, V2, W2, 2, 2, 6, 2)
            
        if n == 4 and m == 3 and p == 6:
            # Split n into 2, 2
            U1, V1, W1 = get_decomp(2, 3, 6)
            U2, V2, W2 = get_decomp(2, 3, 6)
            return combine_n(U1, V1, W1, U2, V2, W2, 2, 3, 6, 2)
            
        if n == 4 and m == 5 and p == 6:
            # Split m into 2, 3
            U1, V1, W1 = get_decomp(4, 2, 6)
            U2, V2, W2 = get_decomp(4, 3, 6)
            return combine_m(U1, V1, W1, U2, V2, W2, 4, 2, 6, 3)
            
        if n == 2 and m == 2 and p == 6:
            # Split p into 2, 2, 2
            U1, V1, W1 = get_decomp(2, 2, 2)
            U2, V2, W2 = get_decomp(2, 2, 2)
            U12, V12, W12 = combine_p(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2)
            U3, V3, W3 = get_decomp(2, 2, 2)
            return combine_p(U12, V12, W12, U3, V3, W3, 2, 2, 4, 2)
            
        if n == 2 and m == 3 and p == 6:
            # Split m into 1, 2
            U1, V1, W1 = get_decomp(2, 1, 6)
            U2, V2, W2 = get_decomp(2, 2, 6)
            return combine_m(U1, V1, W1, U2, V2, W2, 2, 1, 6, 2)
            
        if n == 2 and m == 1 and p == 6:
            # Split p into 3, 3
            U1, V1, W1 = get_decomp(2, 1, 3)
            U2, V2, W2 = get_decomp(2, 1, 3)
            return combine_p(U1, V1, W1, U2, V2, W2, 2, 1, 3, 3)

        # Fallback: Schoolbook
        return schoolbook(n, m, p)

    def strassen_2x2x2():
        # Rank 7
        # U, V, W are 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # A: 00(0), 11(3). B: 00(0), 11(3). C: 00(0), 11(3).
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00 -> C10 += 1
        # A: 10(2), 11(3). B: 00(0). C: 10(2).
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3: A00(B01-B11) -> C01 += 1
        # A: 00(0). B: 01(1), 11(3) with -1. C: 01(1).
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4: A11(B10-B00) -> C00 += 1, C10 += 1
        # A: 11(3). B: 10(2), 00(0) with -1. C: 00(0), 10(2).
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        # A: 00(0), 01(1). B: 11(3). C: 00(0) with -1, 01(1) with 1.
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (A11-A00)(B00+B01) -> C11 += 1
        # A: 11(3), 00(0) with -1. B: 00(0), 01(1). C: 11(3).
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        # A: 01(1), 11(3) with -1. B: 10(2), 11(3). C: 00(0).
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
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

    def combine_n(U1, V1, W1, U2, V2, W2, n1, m, p, n2):
        """
        Combine M_{n1,m,p} and M_{n2,m,p} into M_{n1+n2,m,p}.
        """
        R1 = len(U1[0]) if U1 else 0
        R2 = len(U2[0]) if U2 else 0
        R = R1 + R2
        n = n1 + n2
        
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        
        # Fill U
        for i in range(n1):
            for j in range(m):
                for c in range(R1):
                    U[i*m + j][c] = U1[i*m1 + j][c] if False else U1[i*m + j][c] # Note: U1 index is i*m + j because m is same
            # Correction: U1 has shape (n1*m, R1). Index is i*m + j.
            # Global U index is i*m + j.
            # So direct copy works for rows 0..n1*m-1.
            
        for i in range(n2):
            for j in range(m):
                for c in range(R2):
                    U[(n1+i)*m + j][R1+c] = U2[i*m + j][c]
                    
        # Fill V
        for j in range(m):
            for k in range(p):
                for c in range(R1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(R2):
                    V[j*p + k][R1+c] = V2[j*p + k][c]
                    
        # Fill W
        # W1 shape (p*n1, R1). Index k*n1 + i1.
        # W2 shape (p*n2, R2). Index k*n2 + i2.
        # Global W shape (p*n, R). Index k*n + i.
        
        for k in range(p):
            for i1 in range(n1):
                for c in range(R1):
                    W[k*n + i1][c] = W1[k*n1 + i1][c]
            for i2 in range(n2):
                for c in range(R2):
                    W[k*n + n1 + i2][R1+c] = W2[k*n2 + i2][c]
                    
        return U, V, W

    def combine_m(U1, V1, W1, U2, V2, W2, n, m1, p, m2):
        """
        Combine M_{n,m1,p} and M_{n,m2,p} into M_{n,m1+m2,p}.
        """
        R1 = len(U1[0]) if U1 else 0
        R2 = len(U2[0]) if U2 else 0
        R = R1 + R2
        m = m1 + m2
        
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        
        # Fill U
        # U1 shape (n*m1, R1). Index i*m1 + j1.
        # U2 shape (n*m2, R2). Index i*m2 + j2.
        # Global U shape (n*m, R). Index i*m + j.
        
        for i in range(n):
            for j1 in range(m1):
                for c in range(R1):
                    U[i*m + j1][c] = U1[i*m1 + j1][c]
            for j2 in range(m2):
                for c in range(R2):
                    U[i*m + m1 + j2][R1+c] = U2[i*m2 + j2][c]
                    
        # Fill V
        # V1 shape (m1*p, R1). Index j1*p + k.
        # V2 shape (m2*p, R2). Index j2*p + k.
        # Global V shape (m*p, R). Index j*p + k.
        
        for j1 in range(m1):
            for k in range(p):
                for c in range(R1):
                    V[j1*p + k][c] = V1[j1*p + k][c]
        for j2 in range(m2):
            for k in range(p):
                for c in range(R2):
                    V[(m1+j2)*p + k][R1+c] = V2[j2*p + k][c]
                    
        # Fill W
        # W1 shape (p*n, R1). Index k*n + i.
        # W2 shape (p*n, R2). Index k*n + i.
        # Global W shape (p*n, R). Index k*n + i.
        
        for k in range(p):
            for i in range(n):
                for c in range(R1):
                    W[k*n + i][c] = W1[k*n + i][c]
                for c in range(R2):
                    W[k*n + i][R1+c] = W2[k*n + i][c]
                    
        return U, V, W

    def combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}.
        """
        R1 = len(U1[0]) if U1 else 0
        R2 = len(U2[0]) if U2 else 0
        R = R1 + R2
        p = p1 + p2
        
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        
        # Fill U
        # U1 shape (n*m, R1).
        # U2 shape (n*m, R2).
        # Global U shape (n*m, R).
        
        for i in range(n):
            for j in range(m):
                for c in range(R1):
                    U[i*m + j][c] = U1[i*m + j][c]
                for c in range(R2):
                    U[i*m + j][R1+c] = U2[i*m + j][c]
                    
        # Fill V
        # V1 shape (m*p1, R1). Index j*p1 + k1.
        # V2 shape (m*p2, R2). Index j*p2 + k2.
        # Global V shape (m*p, R). Index j*p + k.
        
        for j in range(m):
            for k1 in range(p1):
                for c in range(R1):
                    V[j*p + k1][c] = V1[j*p1 + k1][c]
            for k2 in range(p2):
                for c in range(R2):
                    V[j*p + p1 + k2][R1+c] = V2[j*p2 + k2][c]
                    
        # Fill W
        # W1 shape (p1*n, R1). Index k1*n + i.
        # W2 shape (p2*n, R2). Index k2*n + i.
        # Global W shape (p*n, R). Index k*n + i.
        
        for k1 in range(p1):
            for i in range(n):
                for c in range(R1):
                    W[k1*n + i][c] = W1[k1*n + i][c]
        for k2 in range(p2):
            for i in range(n):
                for c in range(R2):
                    W[(p1+k2)*n + i][R1+c] = W2[k2*n + i][c]
                    
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
