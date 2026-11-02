import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for m_idx in range(2):
            for k in range(6):
                a_idx = i * 2 + m_idx
                b_idx = m_idx * 6 + k
                c_idx = i * 6 + k
                T[a_idx, b_idx, c_idx] = 1.0

    target_rank = 20
    
    best_result = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Strassen coefficients for 2x2x2
    # From the current program's fallback
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Indices for B and C splits
    # B: 2x6. B1: cols 0-3, B2: cols 4-5
    # B indices: m*6+k
    # B1 indices: 0,1,2,3 (m=0), 6,7,8,9 (m=1)
    # B2 indices: 4,5 (m=0), 10,11 (m=1)
    B1_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    B2_indices = [4, 5, 10, 11]
    
    # C: 2x6. C1: cols 0-3, C2: cols 4-5
    # C indices: i*6+k
    # C1 indices: 0,1,2,3 (i=0), 6,7,8,9 (i=1)
    # C2 indices: 4,5 (i=0), 10,11 (i=1)
    C1_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    C2_indices = [4, 5, 10, 11]
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        U = np.zeros((4, target_rank))
        V = np.zeros((12, target_rank))
        W = np.zeros((12, target_rank))
        
        # Initialize last 7 columns with Strassen for B2, C2
        # Map V_local (4x7) to V (12x7)
        # V_local rows 0,1 correspond to B row 0. Rows 2,3 correspond to B row 1.
        # In V (12x7), rows 4,5 are B row 0, cols 4,5. Rows 10,11 are B row 1, cols 4,5.
        V[4, 13:] = V_local[0, :]
        V[5, 13:] = V_local[1, :]
        V[10, 13:] = V_local[2, :]
        V[11, 13:] = V_local[3, :]
        
        # Map W_local (4x7) to W (12x7)
        # W_local rows 0,1 correspond to C row 0. Rows 2,3 correspond to C row 1.
        # In W (12x7), rows 4,5 are C row 0, cols 4,5. Rows 10,11 are C row 1, cols 4,5.
        W[4, 13:] = W_local[0, :]
        W[5, 13:] = W_local[1, :]
        W[10, 13:] = W_local[2, :]
        W[11, 13:] = W_local[3, :]
        
        U[:, 13:] = U_local[:, :]
        
        # Initialize first 13 columns randomly, but restricted to B1, C1
        # U[:, :13] random
        U[:, :13] = np.random.randn(4, 13) * 0.5
        # V[:, :13] random only at B1 indices
        V[B1_indices, :13] = np.random.randn(8, 13) * 0.5
        # W[:, :13] random only at C1 indices
        W[C1_indices, :13] = np.random.randn(8, 13) * 0.5
        
        # ALS Optimization
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Optimize U
            M = np.zeros((144, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            for i in range(4):
                T_flat = T[i].flatten()
                try:
                    U[i] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Optimize V
            M = np.zeros((48, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            for j in range(12):
                T_flat = T[:, j, :].flatten()
                try:
                    V[j] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Optimize W
            M = np.zeros((48, target_rank))
            for r in range(target_rank):
                M[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            for k in range(12):
                T_flat = T[:, :, k].flatten()
                try:
                    W[k] = np.linalg.lstsq(M, T_flat, rcond=None)[0]
                except:
                    pass
            
            # Calculate error
            T_approx = np.zeros((4, 12, 12))
            for r in range(target_rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T - T_approx)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-5:
                break

    if best_result is not None and best_err < 1e-4:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 21
    n, m, p = DIMENSIONS
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        for i in range(4):
            for r in range(7):
                U[i, base + r] = U_local[i][r]
        for r in range(7):
            V[2*k, base + r] = V_local[0][r]
            V[2*k+1, base + r] = V_local[1][r]
            V[6+2*k, base + r] = V_local[2][r]
            V[7+2*k, base + r] = V_local[3][r]
        for r in range(7):
            W[4*k, base + r] = W_local[0][r]
            W[4*k+2, base + r] = W_local[1][r]
            W[4*k+1, base + r] = W_local[2][r]
            W[4*k+3, base + r] = W_local[3][r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
