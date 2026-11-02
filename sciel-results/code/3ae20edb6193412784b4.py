import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Verify by computing actual multiplication for random matrices
    np.random.seed(12345)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U_arr[:, c]
            vc = V_arr[:, c]
            wc = W_arr[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            A_scaled = A * A_c
            B_scaled = B * B_c
            C_computed += wc.reshape(p, n).T * (A_scaled @ B_scaled)
        
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_strassen_444():
    """Build the Strassen decomposition for 4x4x4 multiplication.
    Returns U, V, W matrices of shapes (16, 49), (16, 49), (16, 49).
    """
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, -1, -1, 1, 0, 1]
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 1, 0]
    ], dtype=float)
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    block_indices = {
        0: [0, 1, 4, 5],   # 11
        1: [2, 3, 6, 7],   # 12
        2: [8, 9, 12, 13], # 21
        3: [10, 11, 14, 15] # 22
    }
    
    for s in range(7):
        A_blocks = {}
        B_blocks = {}
        
        if s == 0:
            A_blocks = {0: 1, 3: 1}
            B_blocks = {0: 1, 3: 1}
        elif s == 1:
            A_blocks = {2: 1, 3: 1}
            B_blocks = {0: 1}
        elif s == 2:
            A_blocks = {0: 1}
            B_blocks = {1: 1, 3: -1}
        elif s == 3:
            A_blocks = {3: 1}
            B_blocks = {2: 1, 0: -1}
        elif s == 4:
            A_blocks = {0: 1, 1: 1}
            B_blocks = {3: 1}
        elif s == 5:
            A_blocks = {2: 1, 0: -1}
            B_blocks = {0: 1, 1: 1}
        elif s == 6:
            A_blocks = {1: 1, 3: -1}
            B_blocks = {2: 1, 3: 1}
            
        C_blocks = {}
        if s == 0:
            C_blocks = {0: 1, 3: 1}
        elif s == 1:
            C_blocks = {2: 1}
        elif s == 2:
            C_blocks = {1: 1}
        elif s == 3:
            C_blocks = {0: 1}
        elif s == 4:
            C_blocks = {1: 1, 3: -1}
        elif s == 5:
            C_blocks = {3: 1}
        elif s == 6:
            C_blocks = {0: 1}
            
        for t in range(7):
            col = s * 7 + t
            
            for block_id, coeff in A_blocks.items():
                indices = block_indices[block_id]
                for k, idx in enumerate(indices):
                    U[idx, col] += coeff * U_2x2[k, t]
            
            for block_id, coeff in B_blocks.items():
                indices = block_indices[block_id]
                for k, idx in enumerate(indices):
                    V[idx, col] += coeff * V_2x2[k, t]
            
            for block_id, coeff in C_blocks.items():
                indices = block_indices[block_id]
                for k, idx in enumerate(indices):
                    W[idx, col] += coeff * W_2x2[k, t]
                    
    return U, V, W

def _build_schoolbook_443():
    """Build the schoolbook decomposition for 4x4x3 multiplication.
    Returns U, V, W matrices of shapes (16, 48), (12, 48), (12, 48).
    """
    n, m, p = 4, 4, 3
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

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # 1. Strassen for 4x4x4 (columns 0-3)
    U_str, V_str, W_str = _build_strassen_444()
    
    # 2. Schoolbook for 4x4x3 (columns 4-6)
    U_sch, V_sch, W_sch = _build_schoolbook_443()
    
    # Combine them
    # U: (16, 49) and (16, 48) -> (16, 97)
    U = np.hstack([U_str, U_sch])
    
    # V: need to properly place V_sch
    # V_str: (16, 49) for B columns 0-3, layout V[j*4+k] for j=0..3, k=0..3
    # V_sch: (12, 48) for B columns 4-6, layout V[j*3+k'] for j=0..3, k'=0..2
    # Full V: (28, 97), layout V[j*7+k] for j=0..3, k=0..6
    V = np.zeros((28, 97))
    V[0:16, 0:49] = V_str
    # Place V_sch correctly: V_sch[j*3+k'] -> V[j*7+(4+k')]
    for j in range(4):
        for k_prime in range(3):
            V[j*7 + (4+k_prime), 49:97] = V_sch[j*3 + k_prime, :]
    
    # W: (16, 49) and (12, 48)
    # W layout is W[k*n+i] which is consistent
    W = np.zeros((28, 97))
    W[0:16, 0:49] = W_str
    W[16:28, 49:97] = W_sch
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify with deadline check
    if time.time() < deadline - 5:
        if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
            # Fallback to schoolbook
            U_fb = np.zeros((16, 112))
            V_fb = np.zeros((28, 112))
            W_fb = np.zeros((28, 112))
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(7):
                        U_fb[i*4+j, col] = 1
                        V_fb[j*7+k, col] = 1
                        W_fb[k*4+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
