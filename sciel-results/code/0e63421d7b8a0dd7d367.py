"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _strassen_222():
    """
    Return the Strassen decomposition for 2x2*2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 tensor.
    Tensor convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    where i in [0,2), j in [0,2), k in [0,2).
    
    Strassen's 7 products:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: (i,j) pairs -> coefficients for A[i,j]
    # V: (j,k) pairs -> coefficients for B[j,k]
    # W: (k,i) pairs -> coefficients for C[i,k] (note: W index is k*2+i per convention)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)*(B11+B22)
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W index: k*2+i
    
    # M1: A11+A22 = U[0]+U[3], B11+B22 = V[0]+V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0*2+0, 0] = 1  # C11, k=0,i=0
    W[1*2+1, 0] = 1  # C22, k=1,i=1
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[0*2+1, 1] = 1  # C21, k=0,i=1
    W[1*2+1, 1] = -1  # C22, k=1,i=1
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1*2+0, 2] = 1  # C12, k=1,i=0
    W[1*2+1, 2] = 1  # C22, k=1,i=1
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0*2+0, 3] = 1  # C11, k=0,i=0
    W[0*2+1, 3] = 1  # C21, k=0,i=1
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0*2+0, 4] = -1  # C11, k=0,i=0
    W[1*2+0, 4] = 1   # C12, k=1,i=0
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[1*2+1, 5] = 1  # C22, k=1,i=1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0*2+0, 6] = 1  # C11, k=0,i=0
    
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """
    Return schoolbook decomposition for n_b x m_b * m_b x p_b multiplication.
    Returns (U, V, W) each of shape (n_b*m_b, n_b*m_b*p_b), (m_b*p_b, n_b*m_b*p_b), (p_b*n_b, n_b*m_b*p_b).
    """
    rank = n_b * m_b * p_b
    U = np.zeros((n_b * m_b, rank), dtype=int)
    V = np.zeros((m_b * p_b, rank), dtype=int)
    W = np.zeros((p_b * n_b, rank), dtype=int)
    
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i * m_b + j, col] = 1
                V[j * p_b + k, col] = 1
                W[k * n_b + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition:
    - n=4 split into [2, 2]
    - m=5 split into [2, 2, 1]
    - p=6 split into [2, 2, 2]
    
    (2,2,2) blocks use Strassen (rank 7), (2,1,2) blocks use schoolbook (rank 4).
    Total: 12 * 7 + 6 * 4 = 84 + 24 = 108.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Block partitions
    i_splits = [2, 2]       # n=4
    j_splits = [2, 2, 1]    # m=5
    k_splits = [2, 2, 2]    # p=6
    
    # Compute offsets
    i_offsets = []
    off = 0
    for s in i_splits:
        i_offsets.append(off)
        off += s
    
    j_offsets = []
    off = 0
    for s in j_splits:
        j_offsets.append(off)
        off += s
    
    k_offsets = []
    off = 0
    for s in k_splits:
        k_offsets.append(off)
        off += s
    
    # Get Strassen decomposition for 2x2x2
    S_U, S_V, S_W = _strassen_222()
    
    # Count total rank
    total_rank = 0
    for nb in i_splits:
        for mb in j_splits:
            for pb in k_splits:
                if nb == 2 and mb == 2 and pb == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += nb * mb * pb  # Schoolbook
    
    # Allocate factor matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    for i_idx in range(len(i_splits)):
        i_0 = i_offsets[i_idx]
        n_b = i_splits[i_idx]
        for j_idx in range(len(j_splits)):
            j_0 = j_offsets[j_idx]
            m_b = j_splits[j_idx]
            for k_idx in range(len(k_splits)):
                k_0 = k_offsets[k_idx]
                p_b = k_splits[k_idx]
                
                # Check deadline
                if time.time() - start_time > DEADLINE:
                    # Return what we have (partial but valid if we complete current block)
                    pass
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen
                    for r in range(7):
                        for i in range(2):
                            for j in range(2):
                                if S_U[i * 2 + j, r] != 0:
                                    U[(i_0 + i) * m + (j_0 + j), col + r] += S_U[i * 2 + j, r]
                        for j in range(2):
                            for k in range(2):
                                if S_V[j * 2 + k, r] != 0:
                                    V[(j_0 + j) * p + (k_0 + k), col + r] += S_V[j * 2 + k, r]
                        for k in range(2):
                            for i in range(2):
                                if S_W[k * 2 + i, r] != 0:
                                    W[(k_0 + k) * n + (i_0 + i), col + r] += S_W[k * 2 + i, r]
                    col += 7
                else:
                    # Schoolbook
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0 + u) * m + (j_0 + v), col] = 1
                                V[(j_0 + v) * p + (k_0 + w), col] = 1
                                W[(k_0 + w) * n + (i_0 + u), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
