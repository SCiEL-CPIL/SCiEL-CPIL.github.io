import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    # Base decompositions
    # 2x2x2 Strassen
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_222 = np.array([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, -1, 1, 0]
    ])
    
    # 2x1x2
    U_212 = np.array([
        [1, 1],
        [0, 1] # Wait, my manual derivation:
        # Col 0: U0=1, U1=0. V0=1, V1=0. W0=1, W2=0, W1=0, W3=0.
        # Col 1: U0=1, U1=1. V0=1, V1=1. W0=0, W1=1, W2=1, W3=0.
    ])
    # Correcting U_212
    U_212 = np.array([
        [1, 1],
        [0, 1]
    ])
    V_212 = np.array([
        [1, 1],
        [0, 1]
    ])
    W_212 = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 0]
    ])
    
    # 2x1x3
    U_213 = np.array([
        [1, 0],
        [0, 1]
    ])
    V_213 = np.array([
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    W_213 = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    
    # 2x2x1
    U_221 = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    V_221 = np.array([
        [1, 0],
        [0, 1]
    ])
    W_221 = np.array([
        [1, 1],
        [1, 1]
    ])

    # Combine functions
    def combine_split_m(d1, d2, n, p):
        # d1 is (n, m1, p), d2 is (n, m2, p)
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        m1 = U1.shape[0] // n
        m2 = U2.shape[0] // n
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        U = np.zeros((n * (m1 + m2), r1 + r2))
        V = np.zeros(((m1 + m2) * p, r1 + r2))
        W = np.zeros((p * n, r1 + r2))
        
        U[:n*m1, :r1] = U1
        U[n*m1:, r1:] = U2
        
        V[:m1*p, :r1] = V1
        V[m1*p:, r1:] = V2
        
        W[:, :r1] = W1
        W[:, r1:] = W2
        
        return (U, V, W)

    def combine_split_p(d1, d2, n, m):
        # d1 is (n, m, p1), d2 is (n, m, p2)
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        p1 = W1.shape[0] // n
        p2 = W2.shape[0] // n
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        U = np.zeros((n * m, r1 + r2))
        V = np.zeros((m * (p1 + p2), r1 + r2))
        W = np.zeros(((p1 + p2) * n, r1 + r2))
        
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        V[:m*p1, :r1] = V1
        V[m*p1:, r1:] = V2
        
        W[:p1*n, :r1] = W1
        W[p1*n:, r1:] = W2
        
        return (U, V, W)

    # Build 2x2x3
    d_223 = combine_split_p((U_222, V_222, W_222), (U_221, V_221, W_221), 2, 2)
    
    # Build 2x3x2
    d_232 = combine_split_m((U_222, V_222, W_222), (U_212, V_212, W_212), 2, 2)
    
    # Build 2x3x3
    d_233 = combine_split_m(d_223, (U_213, V_213, W_213), 2, 3)
    
    # Build 2x3x7
    # Split p=7 into 3+2+2
    d_235 = combine_split_p(d_233, d_232, 2, 3)
    d_237 = combine_split_p(d_235, d_232, 2, 3)
    
    U, V, W = d_237
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
