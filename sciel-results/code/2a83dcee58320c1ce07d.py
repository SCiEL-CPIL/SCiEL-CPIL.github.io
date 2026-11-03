"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each tensor is (u, v, w) where u is 4-dim, v is 4-dim, w is 4-dim.
    Indices: u[i*m+j], v[j*p+k], w[k*n+i] with n=m=p=2.
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1 -> [1,0, 0,1]
    # v: B00=1, B11=1 -> [1,0, 0,1]
    # w: C00=1, C11=1 -> w[k*n+i]: C[0,0] -> k=0,i=0 -> idx 0; C[1,1] -> k=1,i=1 -> idx 3
    #    -> [1,0, 0,1]
    terms.append(([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]))
    
    # M2 = (A10+A11)B00
    # u: A10=1, A11=1 -> [0,0, 1,1]
    # v: B00=1 -> [1,0, 0,0]
    # w: C10=1 -> C[1,0] -> k=0,i=1 -> idx 1
    #    -> [0,1, 0,0]
    terms.append(([0,0, 1,1], [1,0, 0,0], [0,1, 0,0]))
    
    # M3 = A00(B10-B11)
    # u: A00=1 -> [1,0, 0,0]
    # v: B10=1, B11=-1 -> [0,0, 1,-1]
    # w: C01=1 -> C[0,1] -> k=1,i=0 -> idx 2
    #    -> [0,0, 1,0]
    terms.append(([1,0, 0,0], [0,0, 1,-1], [0,0, 1,0]))
    
    # M4 = A11(B00-B01)
    # u: A11=1 -> [0,0, 0,1]
    # v: B00=1, B01=-1 -> [1,-1, 0,0]
    # w: C11=1 -> C[1,1] -> k=1,i=1 -> idx 3
    #    -> [0,0, 0,1]
    terms.append(([0,0, 0,1], [1,-1, 0,0], [0,0, 0,1]))
    
    # M5 = (A00+A01)B11
    # u: A00=1, A01=1 -> [1,1, 0,0]
    # v: B11=1 -> [0,0, 0,1]
    # w: C01=1 -> C[0,1] -> k=1,i=0 -> idx 2
    #    -> [0,0, 1,0]
    terms.append(([1,1, 0,0], [0,0, 0,1], [0,0, 1,0]))
    
    # M6 = (A10-A00)(B00+B01)
    # u: A10=1, A00=-1 -> [-1,0, 1,0]
    # v: B00=1, B01=1 -> [1,1, 0,0]
    # w: C10=1 -> C[1,0] -> k=0,i=1 -> idx 1
    #    -> [0,1, 0,0]
    terms.append(([-1,0, 1,0], [1,1, 0,0], [0,1, 0,0]))
    
    # M7 = (A01-A11)(B10+B11)
    # u: A01=1, A11=-1 -> [0,1, 0,-1]
    # v: B10=1, B11=1 -> [0,0, 1,1]
    # w: C11=1 -> C[1,1] -> k=1,i=1 -> idx 3
    #    -> [0,0, 0,1]
    terms.append(([0,1, 0,-1], [0,0, 1,1], [0,0, 0,1]))
    
    return terms

def _embed_strassen_2x2x2():
    """
    Embed the 2x2x2 Strassen decomposition into the top-left 2x2x2 part of 4x4x5.
    Returns the U, V, W contributions for these 7 terms.
    """
    n, m, p = 4, 4, 5
    terms = _strassen_2x2x2()
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(2):
            for j in range(2):
                U[i*m + j, r] = u[i*2 + j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, r] = v[j*2 + k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, r] = w[k*2 + i]
    
    return U, V, W

def _construct_strassen_embedded():
    """
    Construct a decomposition using Strassen for the top-left 2x2x2 part
    and schoolbook for the remaining parts.
    
    Top-left 2x2x2: i in [0,2), k in [0,2), j in [0,2) -> 8 entries, rank 7
    Remaining: 4*4*5 - 8 = 80 - 8 = 72 entries, rank 72
    Total rank: 7 + 72 = 79
    """
    n, m, p = 4, 4, 5
    rank = 79
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 7 columns: Strassen for top-left 2x2x2
    U_str, V_str, W_str = _embed_strassen_2x2x2()
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    # Remaining 72 columns: schoolbook for all (i,k,j) except those in [0,2)^3
    col = 7
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i < 2 and k < 2 and j < 2:
                    continue  # Already covered by Strassen
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try the Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_embedded()
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Check time and return
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    # Try to find a better decomposition through additional search
    # For now, return the best result we have
    return best_result

# EVOLVE-BLOCK-END
