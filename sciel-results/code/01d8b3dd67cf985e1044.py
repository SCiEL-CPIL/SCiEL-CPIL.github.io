import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    
    # Try to find a better decomposition using a structured approach
    # based on the block structure of the problem.
    
    # For 3x3x7, we can write:
    # C = A * B
    # where A is 3x3, B is 3x7, C is 3x7.
    
    # Let's try to use a decomposition that exploits the fact that
    # the last 4 columns of B can be handled with rank 12 (schoolbook),
    # and the first 3 columns can be handled with a lower-rank decomposition.
    
    # For the 3x3x3 core, let's try to use a rank-20 decomposition.
    # I'll try to construct one using algebraic identities.
    
    U_core, V_core, W_core = _decompose_3x3x3()
    if U_core is not None:
        r_core = U_core.shape[1]
        
        # For the 3x3x4 extension, use schoolbook
        U_ext, V_ext, W_ext = _decompose_3x3x4()
        r_ext = U_ext.shape[1]
        
        # Combine the decompositions
        # U has shape (9, r_core + r_ext)
        # V has shape (21, r_core + r_ext)
        # W has shape (21, r_core + r_ext)
        
        U = np.hstack([U_core, U_ext])
        
        # V: for the core, V_core has shape (9, r_core) corresponding to B[:, 0:3]
        #    for the extension, V_ext has shape (12, r_ext) corresponding to B[:, 3:7]
        #    V has shape (21, r_core + r_ext)
        #    V[j*7 + k, r] for k in [0,3) should come from V_core
        #    V[j*7 + k, r] for k in [3,7) should come from V_ext
        
        V = np.zeros((21, r_core + r_ext))
        for j in range(3):
            for k in range(3):
                V[j*7 + k, :r_core] = V_core[j*3 + k, :]
            for k in range(4):
                V[j*7 + 3 + k, r_core:] = V_ext[j*4 + k, :]
        
        # W: for the core, W_core has shape (9, r_core) corresponding to C[:, 0:3]
        #    for the extension, W_ext has shape (12, r_ext) corresponding to C[:, 3:7]
        #    W has shape (21, r_core + r_ext)
        #    W[k*3 + i, r] for k in [0,3) should come from W_core
        #    W[k*3 + i, r] for k in [3,7) should come from W_ext
        
        W = np.zeros((21, r_core + r_ext))
        for k in range(3):
            for i in range(3):
                W[k*3 + i, :r_core] = W_core[k*3 + i, :]
            for k2 in range(4):
                W[3 + k2, :r_core] = 0  # This is wrong, let me fix it
        
        # Actually, the indexing is:
        # W[k*n + i, r] where k in [0,7), i in [0,3)
        # For k in [0,3): W[k*3 + i, r] should come from W_core[k*3 + i, r]
        # For k in [3,7): W[k*3 + i, r] should come from W_ext[(k-3)*3 + i, r]
        
        W = np.zeros((21, r_core + r_ext))
        for k in range(3):
            for i in range(3):
                W[k*3 + i, :r_core] = W_core[k*3 + i, :]
        for k in range(4):
            for i in range(3):
                W[(3+k)*3 + i, r_core:] = W_ext[k*3 + i, :]
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _decompose_3x3x3():
    """Try to find a rank-20 decomposition for 3x3x3. Returns None if not found."""
    # This is a placeholder. I need to find or derive a rank-20 decomposition.
    # For now, return None.
    return None, None, None

def _decompose_3x3x4():
    """Schoolbook decomposition for 3x3x4."""
    n, m, p = 3, 3, 4
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

def _schoolbook(n, m, p):
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
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-9)
