import numpy as np
import time

def _trivial_decomposition(n, m, p):
    """
    Returns U, V, W for n x m x p where at least one dimension is 1.
    Rank is n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2_2_2():
    """
    Returns U, V, W for 2 x 2 x 2 matrix multiplication.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # U: i*m+j. 0:00, 1:01, 2:10, 3:11
    # V: j*p+k. 0:00, 1:01, 2:10, 3:11
    # W: k*n+i. 0:00, 1:10, 2:01, 3:11 (k=0,i=0->0; k=1,i=0->2; k=0,i=1->1; k=1,i=1->3)
    # Wait, W index is k*n+i.
    # n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: 0, 3. V: 0, 3.
    # Contributes to C00 (W0), C11 (W3).
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # U: 2, 3. V: 0.
    # Contributes to C10 (W1), C11 (W3).
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # Term 3: M3 = A00(B01-B11)
    # U: 0. V: 1, -3.
    # Contributes to C01 (W2), C11 (W3).
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B10-A00)
    # U: 3. V: 2, -0.
    # Contributes to C10 (W1), C11 (W3).
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # U: 0, 1. V: 3.
    # Contributes to C01 (W2), C11 (W3).
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: 2, -0. V: 0, 1.
    # Contributes to C11 (W3).
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: 1, -3. V: 2, 3.
    # Contributes to C00 (W0), C11 (W3).
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, dim, n, m, p, split):
    """
    Combines two decompositions.
    dim: 0 for n, 1 for m, 2 for p.
    split: the size of the first part.
    """
    if dim == 0: # Split n
        # U: stack vertically
        U = np.vstack([U1, U2])
        # V: stack vertically
        V = np.vstack([V1, V2])
        # W: concat horizontally
        W = np.hstack([W1, W2])
    elif dim == 1: # Split m
        # U: stack vertically
        U = np.vstack([U1, U2])
        # V: stack vertically
        V = np.vstack([V1, V2])
        # W: concat horizontally
        W = np.hstack([W1, W2])
    else: # Split p
        # U: concat horizontally
        U = np.hstack([U1, U2])
        # V: concat horizontally
        V = np.hstack([V1, V2])
        # W: stack vertically
        W = np.vstack([W1, W2])
        
    return U, V, W

def _decompose(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return _trivial_decomposition(n, m, p)
    if n == 1 or m == 1 or p == 1:
        return _trivial_decomposition(n, m, p)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2_2_2()
        
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try splitting n
    if n > 1:
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = _decompose(n1, m, p)
            U2, V2, W2 = _decompose(n2, m, p)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = _combine_decompositions(U1, V1, W1, U2, V2, W2, 0, n, m, p, n1)
                
    # Try splitting m
    if m > 1:
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = _decompose(n, m1, p)
            U2, V2, W2 = _decompose(n, m2, p)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = _combine_decompositions(U1, V1, W1, U2, V2, W2, 1, n, m, p, m1)
                
    # Try splitting p
    if p > 1:
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = _decompose(n, m, p1)
            U2, V2, W2 = _decompose(n, m, p2)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = _combine_decompositions(U1, V1, W1, U2, V2, W2, 2, n, m, p, p1)
                
    return best_U, best_V, best_W

def construct():
    n, m, p = 3, 5, 5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
