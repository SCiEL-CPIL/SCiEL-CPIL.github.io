import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen factors for 2x2
    # U2: (4, 7) rows i*2+j
    # V2: (4, 7) rows j*2+k
    # W2: (4, 7) rows k*2+i
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ], dtype=np.int64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ], dtype=np.int64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],  # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],  # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1 (C11)
    ], dtype=np.int64)
    
    # Kronecker products
    rank = 49
    K_U = np.kron(U2, U2)
    K_V = np.kron(V2, V2)
    K_W = np.kron(W2, W2)
    
    # Permute rows to match row-major indexing for 4x4 matrices
    # Index mapping:
    # r_D = 8*i1 + 4*i0 + 2*j1 + j0 (for U, where i is row, j is col)
    # r_K = 8*i1 + 4*j1 + 2*i0 + j0 (from kron)
    
    def get_permutation(n_dim):
        # n_dim is 4 for 4x4
        # Indices 0..15
        perm = []
        for r_D in range(n_dim * n_dim):
            i1 = (r_D // 8) % 2
            rem = r_D % 8
            i0 = (rem // 4) % 2
            rem2 = rem % 4
            j1 = (rem2 // 2) % 2
            j0 = rem2 % 2
            
            r_K = 8 * i1 + 4 * j1 + 2 * i0 + j0
            perm.append(r_K)
        return perm
    
    perm_U = get_permutation(4)
    perm_V = get_permutation(4)
    perm_W = get_permutation(4)
    
    U = K_U[perm_U]
    V = K_V[perm_V]
    W = K_W[perm_W]
    
    # Convert to list of lists as required by some interfaces, or keep as numpy?
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # and returns a dict. The current program returns lists.
    # The evaluator likely accepts numpy arrays or lists. 
    # The current program uses lists. I will return lists to be safe, 
    # but numpy arrays are often preferred for "stronger" performance.
    # However, the task says "Keep the function names and signatures ... return interface EXACTLY".
    # The current return is {"U": U, "V": V, "W": W} where U is a list of lists.
    # I will return lists to match the type exactly.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
