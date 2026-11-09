import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Helper to create a schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        U = []
        V = []
        W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(n * p)
                    
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    
                    U.append(u)
                    V.append(v)
                    W.append(w)
        
        if not U:
            return np.zeros((n * m, 0)), np.zeros((m * p, 0)), np.zeros((n * p, 0))
            
        return np.column_stack(U), np.column_stack(V), np.column_stack(W)

    # Helper to create Strassen decomposition for 2 x 2 x 2
    def strassen_222():
        U = []
        V = []
        W = []
        
        # M1 = (A11 + A22)(B11 + B22)
        # u1: 1 at (1,1), (2,2)
        # v1: 1 at (1,1), (2,2)
        # w1: 1 at (1,1), (2,2)
        u = np.zeros(4); u[0] = 1; u[3] = 1
        v = np.zeros(4); v[0] = 1; v[3] = 1
        w = np.zeros(4); w[0] = 1; w[3] = 1
        U.append(u); V.append(v); W.append(w)
        
        # M2 = (A21 + A22)B11
        # u2: 1 at (2,1), (2,2)
        # v2: 1 at (1,1)
        # w2: 1 at (1,1), (2,1)  <-- Wait, C21 = M2 + M4. C11 = ... - M5 + M7.
        # Let's use the standard Strassen factors carefully.
        
        # Indices:
        # A: 00, 01, 10, 11 (row, col)
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # M1 = (A00+A11)(B00+B11)
        # u1: 00, 11. v1: 00, 11. w1: 00, 11.
        u = np.zeros(4); u[0]=1; u[3]=1
        v = np.zeros(4); v[0]=1; v[3]=1
        w = np.zeros(4); w[0]=1; w[3]=1
        U.append(u); V.append(v); W.append(w)
        
        # M2 = (A10+A11)B00
        # u2: 10, 11. v2: 00. w2: 00, 10. (C00 gets M2? No, C11 gets M2? No.)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map M2 to C21 and C11?
        # M2 contributes to C21 (coefficient 1) and C11 (coefficient 0? No, C11 has no M2).
        # Wait, C11 = M1 + M4 - M5 + M7.
        # C21 = M2 + M4.
        # So M2 contributes to C21 with 1.
        # u2: 10, 11. v2: 00.
        # w2 should be 1 at (1,0) which is index 2?
        # w index: k*n + i.
        # C21 is i=1, k=0. Index 0*2 + 1 = 1? No.
        # C is n x p. i is row (0..n-1), k is col (0..p-1).
        # Index in W: k*n + i.
        # C21: i=1, k=0. Index 0*2 + 1 = 1.
        # So w2[1] = 1.
        
        u = np.zeros(4); u[2]=1; u[3]=1
        v = np.zeros(4); v[0]=1
        w = np.zeros(4); w[1]=1
        U.append(u); V.append(v); W.append(w)
        
        # M3 = A00(B01-B11)
        # Contributes to C12 (1) and C22 (1).
        # C12: i=0, k=1. Index 1*2+0=2.
        # C22: i=1, k=1. Index 1*2+1=3.
        # u3: 00. v3: 01, 11.
        # w3: 1 at 2 and 3.
        
        u = np.zeros(4); u[0]=1
        v = np.zeros(4); v[1]=1; v[3]=-1
        w = np.zeros(4); w[2]=1; w[3]=1
        U.append(u); V.append(v); W.append(w)
        
        # M4 = A11(B10-B00)
        # Contributes to C11 (1) and C21 (1).
        # C11: i=0, k=0. Index 0.
        # C21: i=1, k=0. Index 1.
        # u4: 11. v4: 10, 00.
        # w4: 1 at 0 and 1.
        
        u = np.zeros(4); u[3]=1
        v = np.zeros(4); v[2]=1; v[0]=-1
        w = np.zeros(4); w[0]=1; w[1]=1
        U.append(u); V.append(v); W.append(w)
        
        # M5 = (A00+A01)B11
        # Contributes to C12 (1) and C11 (-1).
        # C12: index 2. C11: index 0.
        # u5: 00, 01. v5: 11.
        # w5: 1 at 2, -1 at 0.
        
        u = np.zeros(4); u[0]=1; u[1]=1
        v = np.zeros(4); v[3]=1
        w = np.zeros(4); w[2]=1; w[0]=-1
        U.append(u); V.append(v); W.append(w)
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to C22 (1).
        # C22: index 3.
        # u6: 10, 00. v6: 00, 01.
        # w6: 1 at 3.
        
        u = np.zeros(4); u[2]=1; u[0]=-1
        v = np.zeros(4); v[0]=1; v[1]=1
        w = np.zeros(4); w[3]=1
        U.append(u); V.append(v); W.append(w)
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to C11 (1).
        # C11: index 0.
        # u7: 01, 11. v7: 10, 11.
        # w7: 1 at 0.
        
        u = np.zeros(4); u[1]=1; u[3]=-1
        v = np.zeros(4); v[2]=1; v[3]=1
        w = np.zeros(4); w[0]=1
        U.append(u); V.append(v); W.append(w)
        
        return np.column_stack(U), np.column_stack(V), np.column_stack(W)

    def build(n, m, p):
        if n == 0 or m == 0 or p == 0:
            return np.zeros((n*m, 0)), np.zeros((m*p, 0)), np.zeros((n*p, 0))
            
        if n == 1 and m == 1 and p == 1:
            return schoolbook(1, 1, 1)
            
        if n == 1:
            return schoolbook(1, m, p)
        if m == 1:
            return schoolbook(n, 1, p)
        if p == 1:
            return schoolbook(n, m, 1)
            
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
            
        # Recursive split
        # Choose split dimension
        # Heuristic: split the largest dimension
        max_dim = max(n, m, p)
        
        if max_dim == n:
            # Split n into n1, n2
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = build(n1, m, p)
            U2, V2, W2 = build(n2, m, p)
            
            # Concatenate
            U = np.hstack([U1, U2]) if U1.shape[1] > 0 and U2.shape[1] > 0 else (U1 if U1.shape[1] > 0 else U2)
            V = np.hstack([V1, V2]) if V1.shape[1] > 0 and V2.shape[1] > 0 else (V1 if V1.shape[1] > 0 else V2)
            W = np.hstack([W1, W2]) if W1.shape[1] > 0 and W2.shape[1] > 0 else (W1 if W1.shape[1] > 0 else W2)
            
            return U, V, W
            
        elif max_dim == m:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = build(n, m1, p)
            U2, V2, W2 = build(n, m2, p)
            
            U = np.hstack([U1, U2]) if U1.shape[1] > 0 and U2.shape[1] > 0 else (U1 if U1.shape[1] > 0 else U2)
            V = np.hstack([V1, V2]) if V1.shape[1] > 0 and V2.shape[1] > 0 else (V1 if V1.shape[1] > 0 else V2)
            W = np.hstack([W1, W2]) if W1.shape[1] > 0 and W2.shape[1] > 0 else (W1 if W1.shape[1] > 0 else W2)
            
            return U, V, W
            
        else:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = build(n, m, p1)
            U2, V2, W2 = build(n, m, p2)
            
            U = np.hstack([U1, U2]) if U1.shape[1] > 0 and U2.shape[1] > 0 else (U1 if U1.shape[1] > 0 else U2)
            V = np.hstack([V1, V2]) if V1.shape[1] > 0 and V2.shape[1] > 0 else (V1 if V1.shape[1] > 0 else V2)
            W = np.hstack([W1, W2]) if W1.shape[1] > 0 and W2.shape[1] > 0 else (W1 if W1.shape[1] > 0 else W2)
            
            return U, V, W

    U, V, W = build(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
