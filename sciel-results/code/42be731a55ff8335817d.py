"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2*2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # The 7 terms of Strassen:
    # Column 0: P1 = (a00+a11)*(b00+b11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # Column 1: P2 = (a10+a11)*b00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 1]
    
    # Column 2: P3 = a00*(b01-b11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, -1]
    
    # Column 3: P4 = a11*(b10-b00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [-1, 0, 0, 1]
    
    # Column 4: P5 = (a00+a01)*b11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [1, 0, 0, 1]
    
    # Column 5: P6 = (a10-a00)*(b00+b01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 1, 1]
    
    # Column 6: P7 = (a01-a11)*(b10+b11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [0, 1, 0, -1]
    
    return U, V, W

def _identity_block(n, m, p):
    """Return trivial rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

@lru_cache(maxsize=None)
def _decompose_nm_p(n, m, p):
    """Decompose n x m * m x p matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2 blocks.
    Returns U, V, W as numpy arrays.
    """
    # Base case: if any dimension is 0, return empty
    if n == 0 or m == 0 or p == 0:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    # Base case: 2x2*2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Base case: if m == 1, it's just outer product scaled
    if m == 1:
        rank = n * p
        U = np.zeros((n*1, rank))
        V = np.zeros((1*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[0, col] = 1
                W[k*n + i, col] = 1
                col += 1
        return U, V, W
    
    # If n == 1
    if n == 1:
        rank = m * p
        U = np.zeros((1*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*1, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j*p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    # If p == 1
    if p == 1:
        rank = n * m
        U = np.zeros((n*m, rank))
        V = np.zeros((m*1, rank))
        W = np.zeros((1*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try splitting m into m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    # Try splitting n into n1 + n2
    for n1 in range(1, n):
        n2 = n - n1
        if n2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n1, m, p)
        U2, V2, W2 = _decompose_nm_p(n2, m, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for i in range(n1):
                for j in range(m):
                    u1_row = i * m + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            for i in range(n2):
                for j in range(m):
                    u2_row = i * m + j
                    full_row = (n1 + i) * m + j
                    U[full_row, r1:] = U2[u2_row, :]
            
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            for k in range(p):
                for i in range(n1):
                    w1_row = k * n1 + i
                    full_row = k * n + i
                    W[full_row, :r1] = W1[w1_row, :]
            
            for k in range(p):
                for i in range(n2):
                    w2_row = k * n2 + i
                    full_row = k * n + (n1 + i)
                    W[full_row, r1:] = W2[w2_row, :]
            
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    # Try splitting p into p1 + p2
    for p1 in range(1, p):
        p2 = p - p1
        if p2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n, m, p1)
        U2, V2, W2 = _decompose_nm_p(n, m, p2)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            for j in range(m):
                for k in range(p1):
                    v1_row = j * p1 + k
                    full_row = j * p + k
                    V[full_row, :r1] = V1[v1_row, :]
            
            for j in range(m):
                for k in range(p2):
                    v2_row = j * p2 + k
                    full_row = j * p + (p1 + k)
                    V[full_row, r1:] = V2[v2_row, :]
            
            for k in range(p1):
                for i in range(n):
                    w1_row = k * n + i
                    full_row = k * n + i
                    W[full_row, :r1] = W1[w1_row, :]
            
            for k in range(p2):
                for i in range(n):
                    w2_row = k * n + i
                    full_row = (p1 + k) * n + i
                    W[full_row, r1:] = W2[w2_row, :]
            
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_nm_p(n, m, p)
    rank = U.shape[1]
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in U.tolist()]
    V_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in V.tolist()]
    W_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
