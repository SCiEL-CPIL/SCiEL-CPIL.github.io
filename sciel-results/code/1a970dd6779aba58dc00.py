import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = build_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def build_decomposition(n, m, p):
    # Base case: any dimension is 1
    if n == 1 or m == 1 or p == 1:
        return build_base_case(n, m, p)
    
    # Base case: 2x2x2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return build_strassen_2x2x2()
        
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = build_decomposition(n1, m, p)
        U2, V2, W2 = build_decomposition(n2, m, p)
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            U, V, W = combine_split_n(n, m, p, n1, n2, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_U, best_V, best_W = U, V, W
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = build_decomposition(n, m1, p)
        U2, V2, W2 = build_decomposition(n, m2, p)
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            U, V, W = combine_split_m(n, m, p, m1, m2, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_U, best_V, best_W = U, V, W
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = build_decomposition(n, m, p1)
        U2, V2, W2 = build_decomposition(n, m, p2)
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            U, V, W = combine_split_p(n, m, p, p1, p2, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_U, best_V, best_W = U, V, W
            
    return best_U, best_V, best_W

def build_base_case(n, m, p):
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i*m+j, 0] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j, 0] = 1
            
    # V[j*p+k, 0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k, 0] = 1
            
    # W[k*n+i, 0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, 0] = 1
            
    return U, V, W

def build_strassen_2x2x2():
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    rank = 7
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: i, j, k in 0..1
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # M1
    U[0*2+0, 0] = 1 # A00
    U[1*2+1, 0] = 1 # A11
    V[0*2+0, 0] = 1 # B00
    V[1*2+1, 0] = 1 # B11
    W[0*2+0, 0] = 1 # C00
    W[1*2+1, 0] = 1 # C11
    
    # M2
    U[1*2+0, 1] = 1 # A10
    U[1*2+1, 1] = 1 # A11
    V[0*2+0, 1] = 1 # B00
    W[0*2+1, 1] = 1 # C10
    W[1*2+1, 1] = -1 # C11
    
    # M3
    U[0*2+0, 2] = 1 # A00
    V[0*2+1, 2] = 1 # B01
    V[1*2+1, 2] = -1 # B11
    W[0*2+0, 2] = 1 # C01? No, C01 = M3+M4.
    # Wait, C01 is index k=1, i=0?
    # C is n x p. C[i, k].
    # W[k*n+i].
    # C01 means i=0, k=1. W[1*2+0] = W[2].
    W[1*2+0, 2] = 1
    W[0*2+1, 2] = 0 # C11? No.
    
    # Let's re-verify C indices.
    # C00: i=0, k=0 -> W[0*2+0]
    # C01: i=0, k=1 -> W[1*2+0]
    # C10: i=1, k=0 -> W[0*2+1]
    # C11: i=1, k=1 -> W[1*2+1]
    
    # M3 = A00(B01-B11)
    # C01 = M3 + M4
    # So W[1*2+0, 2] = 1
    # W[0*2+1, 2] = 0 (C10 doesn't use M3)
    # W[1*2+1, 2] = 0 (C11 doesn't use M3? C11 = M1-M2+M3+M6. Yes it does.)
    # Wait, C11 uses M3.
    W[1*2+1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[1*2+1, 3] = 1 # A11
    V[1*2+0, 3] = 1 # B10
    V[0*2+0, 3] = -1 # B00
    # C00 = M1 - M4 ... -> W[0, 3] = -1
    W[0*2+0, 3] = -1
    # C01 = M3 + M4 ... -> W[2, 3] = 1
    W[1*2+0, 3] = 1
    # C11 = ... M3 + M6 (M4 not in C11)
    
    # M5 = (A00+A01)B11
    U[0*2+0, 4] = 1 # A00
    U[0*2+1, 4] = 1 # A01
    V[1*2+1, 4] = 1 # B11
    # C00 = M1 - M4 + M5 ... -> W[0, 4] = 1
    W[0*2+0, 4] = 1
    # C10 = M2 + M5 ... -> W[1, 4] = 1
    W[0*2+1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1*2+0, 5] = 1 # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1 # B00
    V[0*2+1, 5] = 1 # B01
    # C11 = ... + M6 -> W[3, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0*2+1, 6] = 1 # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1 # B10
    V[1*2+1, 6] = 1 # B11
    # C00 = ... - M7 -> W[0, 6] = -1
    W[0*2+0, 6] = -1
    
    return U, V, W

def combine_split_n(n, m, p, n1, n2, U1, V1, W1, U2, V2, W2):
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U1 corresponds to i in 0..n1-1
    for i in range(n1):
        for j in range(m):
            U[i * m + j, :rank1] = U1[i * m + j, :]
            
    # U2 corresponds to i in n1..n-1 (mapped to 0..n2-1 in U2)
    for i in range(n2):
        for j in range(m):
            U[(i + n1) * m + j, rank1:] = U2[i * m + j, :]
            
    # V is same for both
    V[:, :rank1] = V1
    V[:, rank1:] = V2
    
    # W1 corresponds to i in 0..n1-1
    for k in range(p):
        for i in range(n1):
            W[k * n + i, :rank1] = W1[k * n1 + i, :]
            
    # W2 corresponds to i in n1..n-1
    for k in range(p):
        for i in range(n2):
            W[k * n + (i + n1), rank1:] = W2[k * n2 + i, :]
            
    return U, V, W

def combine_split_m(n, m, p, m1, m2, U1, V1, W1, U2, V2, W2):
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U1 corresponds to j in 0..m1-1
    for i in range(n):
        for j in range(m1):
            U[i * m + j, :rank1] = U1[i * m1 + j, :]
            
    # U2 corresponds to j in m1..m-1
    for i in range(n):
        for j in range(m2):
            U[i * m + (j + m1), rank1:] = U2[i * m2 + j, :]
            
    # V1 corresponds to j in 0..m1-1
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :rank1] = V1[j * p + k, :]
            
    # V2 corresponds to j in m1..m-1
    for j in range(m2):
        for k in range(p):
            V[(j + m1) * p + k, rank1:] = V2[j * p + k, :]
            
    # W is same
    W[:, :rank1] = W1
    W[:, rank1:] = W2
            
    return U, V, W

def combine_split_p(n, m, p, p1, p2, U1, V1, W1, U2, V2, W2):
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U is same
    U[:, :rank1] = U1
    U[:, rank1:] = U2
    
    # V1 corresponds to k in 0..p1-1
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :rank1] = V1[j * p1 + k, :]
            
    # V2 corresponds to k in p1..p-1
    for j in range(m):
        for k in range(p2):
            V[j * p + (k + p1), rank1:] = V2[j * p2 + k, :]
            
    # W1 corresponds to k in 0..p1-1
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :rank1] = W1[k * n + i, :]
            
    # W2 corresponds to k in p1..p-1
    for k in range(p2):
        for i in range(n):
            W[(k + p1) * n + i, rank1:] = W2[k * n + i, :]
            
    return U, V, W
