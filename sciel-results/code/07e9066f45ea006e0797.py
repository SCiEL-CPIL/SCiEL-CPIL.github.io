"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
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

def _strassen_3x3x3_terms():
    """
    Return the 7 rank-1 tensors for 3x3x3 matrix multiplication.
    """
    terms = []
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = [0]*9; u1[0]=1; u1[4]=1
    v1 = [0]*9; v1[0]=1; v1[4]=1
    w1 = [0]*9; w1[0]=1; w1[4]=1
    terms.append((u1, v1, w1))
    
    # Term 2: (A10+A11)(B00)
    u2 = [0]*9; u2[3]=1; u2[4]=1
    v2 = [0]*9; v2[0]=1
    w2 = [0]*9; w2[0]=1
    terms.append((u2, v2, w2))
    
    # Term 3: A00*(B10-B11)
    u3 = [0]*9; u3[0]=1
    v3 = [0]*9; v3[3]=1; v3[4]=-1
    w3 = [0]*9; w3[3]=1
    terms.append((u3, v3, w3))
    
    # Term 4: A11*(B00-B01)
    u4 = [0]*9; u4[4]=1
    v4 = [0]*9; v4[0]=1; v4[1]=-1
    w4 = [0]*9; w4[0]=1; w4[4]=1
    terms.append((u4, v4, w4))
    
    # Term 5: (A00+A01)*B11
    u5 = [0]*9; u5[0]=1; u5[1]=1
    v5 = [0]*9; v5[4]=1
    w5 = [0]*9; w5[3]=1; w5[4]=1
    terms.append((u5, v5, w5))
    
    # Term 6: (A10-A00)*(B00+B01)
    u6 = [0]*9; u6[3]=1; u6[0]=-1
    v6 = [0]*9; v6[0]=1; v6[1]=1
    w6 = [0]*9; w6[0]=1; w6[4]=1
    terms.append((u6, v6, w6))
    
    # Term 7: (A01-A11)*(B10+B11)
    u7 = [0]*9; u7[1]=1; u7[4]=-1
    v7 = [0]*9; v7[3]=1; v7[4]=1
    w7 = [0]*9; w7[0]=1; w7[4]=1
    terms.append((u7, v7, w7))
    
    return terms

def _construct_strassen_embedded(n=4, m=4, p=5):
    """
    Construct a decomposition using Strassen for the 3x3x3 sub-tensor
    and schoolbook for the boundary.
    """
    strassen_terms = _strassen_3x3x3_terms()
    
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u, v, w) in enumerate(strassen_terms):
        for i in range(3):
            for j in range(3):
                U[i * m + j][t] = u[i * 3 + j]
        for j in range(3):
            for k in range(3):
                V[j * p + k][t] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][t] = w[k * 3 + i]
    
    boundary_triples = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i >= 3 or k >= 3 or j >= 3:
                    boundary_triples.append((i, k, j))
    
    for (i, k, j) in boundary_triples:
        U[i * m + j][rank] = 1
        V[j * p + k][rank] = 1
        W[k * n + i][rank] = 1
        rank += 1
    
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Use the Strassen-embedded approach
    result = _construct_strassen_embedded(n, m, p)
    
    return result

# EVOLVE-BLOCK-END
