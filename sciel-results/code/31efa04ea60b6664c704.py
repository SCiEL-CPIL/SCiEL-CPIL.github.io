"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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

@lru_cache(maxsize=None)
def _get_decomposition(n, m, p):
    """
    Recursively find the minimal rank decomposition for n x m x p.
    Returns (U, V, W, rank).
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W, 1

    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7

    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _get_decomposition(n1, m, p)
        U2, V2, W2, r2 = _get_decomposition(n2, m, p)
        
        if r1 + r2 < best_rank:
            # Combine
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Stack rows
            U[:n1*m, :r1] = U1
            U[n1*m:, r1:] = U2
            
            # V: Concat columns
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            # W: Scatter rows based on i index
            # W1 corresponds to i in [0, n1)
            # W2 corresponds to i in [n1, n)
            # Global W index: k*n + i
            for k in range(p):
                for i1 in range(n1):
                    W[k*n + i1, :r1] = W1[k*n1 + i1, :]
                for i2 in range(n2):
                    W[k*n + (n1 + i2), r1:] = W2[k*n2 + i2, :]
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = _get_decomposition(n, m1, p)
        U2, V2, W2, r2 = _get_decomposition(n, m2, p)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Scatter columns based on j index? No, U is indexed by i*m + j.
            # U1 corresponds to j in [0, m1)
            # U2 corresponds to j in [m1, m)
            for i in range(n):
                for j1 in range(m1):
                    U[i*m + j1, :r1] = U1[i*m1 + j1, :]
                for j2 in range(m2):
                    U[i*m + (m1 + j2), r1:] = U2[i*m2 + j2, :]
            
            # V: Stack rows
            # V is indexed by j*p + k
            # V1 corresponds to j in [0, m1)
            # V2 corresponds to j in [m1, m)
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            
            # W: Concat columns
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _get_decomposition(n, m, p1)
        U2, V2, W2, r2 = _get_decomposition(n, m, p2)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Concat columns
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # V: Scatter columns based on k index? No, V is indexed by j*p + k.
            # V1 corresponds to k in [0, p1)
            # V2 corresponds to k in [p1, p)
            for j in range(m):
                for k1 in range(p1):
                    V[j*p + k1, :r1] = V1[j*p1 + k1, :]
                for k2 in range(p2):
                    V[j*p + (p1 + k2), r1:] = V2[j*p2 + k2, :]
            
            # W: Stack rows
            # W is indexed by k*n + i
            # W1 corresponds to k in [0, p1)
            # W2 corresponds to k in [p1, p)
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    return best_U, best_V, best_W, best_rank

def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _get_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
