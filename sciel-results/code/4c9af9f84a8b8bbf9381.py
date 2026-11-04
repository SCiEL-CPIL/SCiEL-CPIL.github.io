import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook(n, m, p):
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
        return U, V, W

    def get_strassen_2x2x2():
        # Rank 7 decomposition for 2x2x2
        # Returns U, V, W of shape 4x7, 4x7, 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        # Standard Strassen coefficients
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # U vectors (A coefficients)
        # V vectors (B coefficients)
        # W vectors (C coefficients)
        
        # We define 7 terms.
        # Term 1:
        # A: A11, A22 -> 1
        # B: B11, B22 -> 1
        # C: C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        # Wait, C11 = M1 - M4 + M5 - M6 + M7
        # C12 = M3 + M6 - M7
        # C21 = M2 - M5 + M7
        # C22 = M1 - M3 + M4 - M6
        
        # Let's list (U, V, W) for each term.
        # U is 4x1, V is 4x1, W is 4x1.
        
        terms = []
        
        # T1: (A11+A22)(B11+B22)
        # U: [1, 0, 0, 1] (A11, A12, A21, A22)
        # V: [1, 0, 0, 1] (B11, B12, B21, B22)
        # W: [1, 0, 0, 1] (C11, C12, C21, C22) -> C11 += 1, C22 += 1
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # T2: (A21+A22)B11
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0] -> C21 += 1
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        
        # T3: A11(B12-B22)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 0] -> C12 += 1
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        
        # T4: A22(B21-B11)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1]
        # W: [0, 0, 0, 1] -> C22 += 1
        terms.append(([0,0,0,1], [0,0,1,-1], [0,0,0,1]))
        
        # T5: (A11+A12)B22
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [0, 0, -1, 0] -> C21 -= 1
        terms.append(([1,1,0,0], [0,0,0,1], [0,0,-1,0]))
        
        # T6: (A21-A11)(B11+B12)
        # U: [0, 0, 1, -1]
        # V: [1, 1, 0, 0]
        # W: [0, 1, 0, -1] -> C12 += 1, C22 -= 1
        terms.append(([0,0,1,-1], [1,1,0,0], [0,1,0,-1]))
        
        # T7: (A12-A22)(B21+B22)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [-1, -1, 1, 1] -> C11 -= 1, C12 -= 1, C21 += 1, C22 += 1
        terms.append(([0,1,0,-1], [0,0,1,1], [-1,-1,1,1]))
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        for i, (u, v, w) in enumerate(terms):
            U[:, i] = u
            V[:, i] = v
            W[:, i] = w
        return U.tolist(), V.tolist(), W.tolist()

    def combine(U1, V1, W1, U2, V2, W2, dim):
        # U1, V1, W1 are lists of lists
        # dim: 0 for n, 1 for m, 2 for p
        
        n1, m1, p1 = len(U1), len(V1)//p1 if p1 else 0, len(W1)//n1 if n1 else 0
        # This is tricky to infer dimensions. Better to pass dimensions.
        
    # I will implement the builder with explicit dimensions.
    
    def build(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return get_strassen_2x2x2()
        
        # Determine split
        # Heuristic: split the largest dimension, or specific logic for known small cases
        if n == 3 and m == 5 and p == 6:
            # Split m into 3, 2
            n1, m1, p1 = 3, 3, 6
            n2, m2, p2 = 3, 2, 6
            U1, V1, W1 = build(n1, m1, p1)
            U2, V2, W2 = build(n2, m2, p2)
            return combine(U1, V1, W1, U2, V2, W2, 1, n, m, p)
            
        if n == 3 and m == 3 and p == 6:
            # Split p into 3, 3
            n1, m1, p1 = 3, 3, 3
            n2, m2, p2 = 3, 3, 3
            U1, V1, W1 = build(n1, m1, p1)
            U2, V2, W2 = build(n2, m2, p2)
            return combine(U1, V1, W1, U2, V2, W2, 2, n, m, p)
            
        if n == 3 and m == 2 and p == 6:
            # Split p into 3, 3
            n1, m1, p1 = 3, 2, 3
            n2, m2, p2 = 3, 2, 3
            U1, V1, W1 = build(n1, m1, p1)
            U2, V2, W2 = build(n2, m2, p2)
            return combine(U1, V1, W1, U2, V2, W2, 2, n, m, p)
            
        if n == 3 and m == 2 and p == 3:
            # Split p into 2, 1
            n1, m1, p1 = 3, 2, 2
            n2, m2, p2 = 3, 2, 1
            U1, V1, W1 = build(n1, m1, p1)
            U2, V2, W2 = build(n2, m2, p2)
            return combine(U1, V1, W1, U2, V2, W2, 2, n, m, p)
            
        if n == 3 and m == 2 and p == 2:
            # Split n into 2, 1
            n1, m1, p1 = 2, 2, 2
            n2, m2, p2 = 1, 2, 2
            U1, V1, W1 = build(n1, m1, p1)
            U2, V2, W2 = build(n2, m2, p2)
            return combine(U1, V1, W1, U2, V2, W2, 0, n, m, p)
            
        if n == 1 and m == 2 and p == 2:
            return get_schoolbook(n, m, p)
            
        if n == 3 and m == 2 and p == 1:
            return get_schoolbook(n, m, p)
            
        if n == 3 and m == 3 and p == 3:
            # Use schoolbook for simplicity, rank 27
            return get_schoolbook(n, m, p)
            
        # Fallback
        return get_schoolbook(n, m, p)

    def combine(U1, V1, W1, U2, V2, W2, dim, n, m, p):
        # U1: n1*m1 x R1
        # V1: m1*p1 x R1
        # W1: p1*n1 x R1
        
        # U2: n2*m2 x R2
        # V2: m2*p2 x R2
        # W2: p2*n2 x R2
        
        R1 = len(U1[0])
        R2 = len(U2[0])
        R = R1 + R2
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        if dim == 0: # Split n
            # U: Vstack
            for i in range(len(U1)):
                for r in range(R1):
                    U[i][r] = U1[i][r]
            for i in range(len(U2)):
                for r in range(R2):
                    U[len(U1) + i][R1 + r] = U2[i][r]
            # V: Hstack
            for i in range(len(V1)):
                for r in range(R1):
                    V[i][r] = V1[i][r]
            for i in range(len(V2)):
                for r in range(R2):
                    V[i][R1 + r] = V2[i][r]
            # W: Vstack
            for i in range(len(W1)):
                for r in range(R1):
                    W[i][r] = W1[i][r]
            for i in range(len(W2)):
                for r in range(R2):
                    W[len(W1) + i][R1 + r] = W2[i][r]
                    
        elif dim == 1: # Split m
            # U: Vstack
            for i in range(len(U1)):
                for r in range(R1):
                    U[i][r] = U1[i][r]
            for i in range(len(U2)):
                for r in range(R2):
                    U[len(U1) + i][R1 + r] = U2[i][r]
            # V: Vstack
            for i in range(len(V1)):
                for r in range(R1):
                    V[i][r] = V1[i][r]
            for i in range(len(V2)):
                for r in range(R2):
                    V[len(V1) + i][R1 + r] = V2[i][r]
            # W: Hstack
            for i in range(len(W1)):
                for r in range(R1):
                    W[i][r] = W1[i][r]
            for i in range(len(W2)):
                for r in range(R2):
                    W[i][R1 + r] = W2[i][r]
                    
        elif dim == 2: # Split p
            # U: Hstack
            for i in range(len(U1)):
                for r in range(R1):
                    U[i][r] = U1[i][r]
            for i in range(len(U2)):
                for r in range(R2):
                    U[i][R1 + r] = U2[i][r]
            # V: Vstack
            for i in range(len(V1)):
                for r in range(R1):
                    V[i][r] = V1[i][r]
            for i in range(len(V2)):
                for r in range(R2):
                    V[len(V1) + i][R1 + r] = V2[i][r]
            # W: Vstack
            for i in range(len(W1)):
                for r in range(R1):
                    W[i][r] = W1[i][r]
            for i in range(len(W2)):
                for r in range(R2):
                    W[len(W1) + i][R1 + r] = W2[i][r]
                    
        return U, V, W

    U, V, W = build(n, m, p)
    return {"U": U, "V": V, "W": W}
