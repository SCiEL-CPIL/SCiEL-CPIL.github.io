import numpy as np

def _tensor_product(U1, V1, W1, U2, V2, W2):
    """
    Compute the tensor product of two decompositions.
    U1: (n1, r1), V1: (m1, r1), W1: (p1, r1)
    U2: (n2, r2), V2: (m2, r2), W2: (p2, r2)
    Returns U, V, W for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    """
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    return U, V, W

def _direct_sum(U1, V1, W1, U2, V2, W2):
    """
    Compute the direct sum of two decompositions with disjoint inputs/outputs.
    U1: (n1, r1), V1: (m1, r1), W1: (p1, r1)
    U2: (n2, r2), V2: (m2, r2), W2: (p2, r2)
    Returns U, V, W for (n1+n2, m1+m2, p1+p2) with rank r1+r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    n1, m1, p1 = U1.shape[0], V1.shape[0], W1.shape[0]
    n2, m2, p2 = U2.shape[0], V2.shape[0], W2.shape[0]
    
    U = np.zeros((n1 + n2, r1 + r2))
    V = np.zeros((m1 + m2, r1 + r2))
    W = np.zeros((p1 + p2, r1 + r2))
    
    U[:n1, :r1] = U1
    V[:m1, :r1] = V1
    W[:p1, :r1] = W1
    
    U[n1:, r1:] = U2
    V[m1:, r1:] = V2
    W[p1:, r1:] = W2
    
    return U, V, W

def _sum_n_split(U1, V1, W1, U2, V2, W2):
    """
    Sum of two decompositions sharing the middle dimension B.
    U1: (n1, r1), V1: (m, r1), W1: (p, r1)
    U2: (n2, r2), V2: (m, r2), W2: (p, r2)
    Returns U, V, W for (n1+n2, m, p) with rank r1+r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    n1, m, p = U1.shape[0], V1.shape[0], W1.shape[0]
    n2 = U2.shape[0]
    
    U = np.zeros((n1 + n2, r1 + r2))
    V = np.zeros((m, r1 + r2))
    W = np.zeros((p, r1 + r2))
    
    U[:n1, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    
    U[n1:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    
    return U, V, W

def _decomp_111():
    return np.array([[1]]), np.array([[1]]), np.array([[1]])

def _decomp_11p(p):
    U = np.ones((1, p))
    V = np.eye(1, p) # No, V is (1, p)? No, m=1. V is (1, p).
    # M_{1,1,p}: A is 1x1, B is 1xp, C is 1xp.
    # C_j = A_1 B_j.
    # U[0, j] = 1, V[0, j] = 1, W[j, j] = 1?
    # W is (p, p).
    U = np.ones((1, p))
    V = np.ones((1, p))
    W = np.eye(p)
    return U, V, W

def _decomp_1mp(m, p):
    # M_{1,m,p}
    # A is 1xm, B is mxp, C is 1xp.
    # C_j = sum_k A_k B_kj.
    # Rank mp.
    U = np.zeros((1, m * p))
    V = np.zeros((m, m * p))
    W = np.zeros((p, m * p))
    
    col = 0
    for k in range(m):
        for j in range(p):
            U[0, col] = 1
            V[k, col] = 1
            W[j, col] = 1
            col += 1
    return U, V, W

def _decomp_222():
    # Strassen
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = -1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    # 1. Get M_{1,1,3}
    U113, V113, W113 = _decomp_11p(3)
    
    # 2. Get M_{2,2,2}
    U222, V222, W222 = _decomp_222()
    
    # 3. Tensor to get M_{2,2,6}
    U226, V226, W226 = _tensor_product(U222, V222, W222, U113, V113, W113)
    
    # 4. Get M_{1,2,6}
    U126, V126, W126 = _decomp_1mp(2, 6)
    
    # 5. Sum n-split to get M_{3,2,6}
    U326, V326, W326 = _sum_n_split(U226, V226, W226, U126, V126, W126)
    
    # 6. Direct sum to get M_{3,4,6}
    U346, V346, W346 = _direct_sum(U326, V326, W326, U326, V326, W326)
    
    return {
        "U": U346.tolist(),
        "V": V346.tolist(),
        "W": W346.tolist()
    }
