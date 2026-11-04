import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook_338():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_rank23_333():
        # Search for rank 23 solution for 3x3x3
        n3, m3, p3 = 3, 3, 3
        target_rank = 23
        max_sweeps = 50
        n_seeds = 20
        
        for seed in range(n_seeds):
            np.random.seed(seed)
            U = np.random.choice([-1, 0, 1], size=(n3*m3, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m3*p3, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p3*n3, target_rank))
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for sweep in range(max_sweeps):
                # Solve for W
                for k in range(p3):
                    for i in range(n3):
                        A = np.zeros((m3, target_rank))
                        for j in range(m3):
                            A[j, :] = U[i*m3+j, :] * V[j*p3+k, :]
                        try:
                            W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m3), rcond=None)
                            W[k*n3+i, :] = W_col
                        except:
                            W[k*n3+i, :] = 0
                            
                # Solve for U
                for i in range(n3):
                    for j in range(m3):
                        A = np.zeros((p3, target_rank))
                        for k in range(p3):
                            A[k, :] = V[j*p3+k, :] * W[k*n3+i, :]
                        try:
                            U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p3), rcond=None)
                            U[i*m3+j, :] = U_col
                        except:
                            U[i*m3+j, :] = 0
                            
                # Solve for V
                for j in range(m3):
                    for k in range(p3):
                        A = np.zeros((n3, target_rank))
                        for i in range(n3):
                            A[i, :] = U[i*m3+j, :] * W[k*n3+i, :]
                        try:
                            V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n3), rcond=None)
                            V[j*p3+k, :] = V_col
                        except:
                            V[j*p3+k, :] = 0
                            
                # Calculate loss
                loss = 0.0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            val = np.dot(U[i*m3+j], V[j*p3+k] * W[k*n3+i])
                            loss += (val - 1.0)**2
                            
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if loss < 1e-12:
                    break
                    
                if time.time() - start_time > timeout - 30:
                    return None
            
            # Round and verify
            U_r = np.round(best_U)
            V_r = np.round(best_V)
            W_r = np.round(best_W)
            
            if verify(U_r, V_r, W_r, n3, m3, p3):
                return U_r, V_r, W_r, target_rank
        
        return None

    # Try to get rank 23 for 3x3x3
    res_333 = get_rank23_333()
    
    if res_333 is not None:
        U333, V333, W333, r333 = res_333
        # Construct 3x3x4
        r331 = 9
        r334 = r333 + r331
        
        U334 = np.zeros((9, r334))
        V334 = np.zeros((12, r334))
        W334 = np.zeros((12, r334))
        
        # Fill 333 part
        U334[:, :r333] = U333
        V334[:9, :r333] = V333
        W334[:9, :r333] = W333
        
        # Fill 331 part (Schoolbook for 3x3x1)
        # U331 is Identity 9x9
        # V331: V[j, i*3+j] = 1
        # W331: W[i, i*3+j] = 1
        for i in range(3):
            for j in range(3):
                idx = i*3 + j
                U334[idx, r333 + idx] = 1
                V334[j*4 + 3, r333 + idx] = 1 # j is row in V (0..2), k=3 is col. Row index j*4+3? 
                # V shape is (m*p, rank) = (3*4, rank). Index is j*p + k = j*4 + 3.
                W334[3*3 + i, r333 + idx] = 1 # W shape is (p*n, rank) = (4*3, rank). Index is k*n + i = 3*3 + i.
                
        # Verify 334
        if verify(U334, V334, W334, 3, 3, 4):
            # Construct 3x3x8 by doubling 3x3x4
            r338 = 2 * r334
            U338 = np.zeros((9, r338))
            V338 = np.zeros((24, r338))
            W338 = np.zeros((24, r338))
            
            # First 4 cols
            U338[:, :r334] = U334
            V338[:12, :r334] = V334
            W338[:12, :r334] = W334
            
            # Last 4 cols
            U338[:, r334:] = U334
            # V rows 12..23 correspond to k=4..7.
            # V334 rows 0..8 correspond to k=0..2. Rows 9..11 correspond to k=3.
            # We need to map V334 to V338 rows 12..23.
            # V338[j*8 + k, r] = V334[j*4 + (k-4), r-r334] for k=4..7.
            for j in range(3):
                for k in range(4, 8):
                    V338[j*8 + k, r334:] = V334[j*4 + (k-4), :]
            
            # W rows 12..23 correspond to k=4..7.
            for k in range(4, 8):
                for i in range(3):
                    W338[k*3 + i, r334:] = W334[(k-4)*3 + i, :]
            
            if verify(U338, V338, W338, 3, 3, 8):
                return {
                    "U": [[float(x) for x in row] for row in U338],
                    "V": [[float(x) for x in row] for row in V338],
                    "W": [[float(x) for x in row] for row in W338]
                }
    
    # Fallback to schoolbook
    return schoolbook_338()
