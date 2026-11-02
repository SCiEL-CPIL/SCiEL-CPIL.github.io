import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110
    
    # Fallback Rank 53
    def get_fallback():
        rank = 53
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        U_coeffs = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        
        def get_V_coeffs(block_idx):
            c0 = 2 * block_idx
            c1 = 2 * block_idx + 1
            return [
                [(0*p + c0, 1), (1*p + c1, 1)],
                [(0*p + c0, 1)],
                [(0*p + c1, 1), (1*p + c1, -1)],
                [(1*p + c0, 1), (0*p + c0, -1)],
                [(1*p + c1, 1)],
                [(0*p + c0, 1), (0*p + c1, 1)],
                [(1*p + c0, 1), (1*p + c1, 1)]
            ]
            
        def get_W_coeffs(block_idx):
            c0 = 2 * block_idx
            c1 = 2 * block_idx + 1
            return [
                [(c0*n + 0, 1), (c1*n + 1, 1)],
                [(c0*n + 1, 1), (c1*n + 1, -1)],
                [(c1*n + 0, 1), (c1*n + 1, 1)],
                [(c0*n + 0, 1), (c0*n + 1, 1)],
                [(c0*n + 0, -1), (c1*n + 0, 1)],
                [(c1*n + 1, 1)],
                [(c0*n + 0, 1)]
            ]
            
        for b in range(7):
            V_sparse = get_V_coeffs(b)
            W_sparse = get_W_coeffs(b)
            for k in range(7):
                for idx, val in enumerate(U_coeffs[k]):
                    if val != 0:
                        U[idx][col] = val
                for idx, val in V_sparse[k]:
                    V[idx][col] = val
                for idx, val in W_sparse[k]:
                    W[idx][col] = val
                col += 1
                
        # 1 block 2x1
        U[0][col] = 1; V[0*p + 14][col] = 1; W[14*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + 14][col] = 1; W[14*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + 14][col] = 1; W[14*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + 14][col] = 1; W[14*n + 1][col] = 1; col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try Rank 50
    try:
        T = np.zeros((4, 6, 6))
        T[0, 0, 0] = 1; T[1, 3, 0] = 1
        T[0, 1, 1] = 1; T[1, 4, 1] = 1
        T[0, 2, 2] = 1; T[1, 5, 2] = 1
        T[2, 0, 3] = 1; T[3, 3, 3] = 1
        T[2, 1, 4] = 1; T[3, 4, 4] = 1
        T[2, 2, 5] = 1; T[3, 5, 5] = 1
        
        rank = 10
        found = False
        best_err = 1e9
        
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
            np.random.seed(restart)
            U = np.random.randn(4, rank)
            V = np.random.randn(6, rank)
            W = np.random.randn(6, rank)
            
            lr = 0.05
            for i in range(2000):
                if time.time() - start_time > timeout:
                    break
                R = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = R - T
                if np.linalg.norm(err) < 1e-6:
                    break
                grad_U = np.einsum('ijk,jr,kr->ir', err, V, W)
                grad_V = np.einsum('ijk,ir,kr->jr', err, U, W)
                grad_W = np.einsum('ijk,ir,jr->kr', err, U, V)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            R_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            err_snap = np.linalg.norm(R_snap - T)
            if err_snap < 1e-6:
                found = True
                break
                
        if found:
            U_full = np.zeros((4, rank * 5))
            V_full = np.zeros((6, rank * 5)) # Wait, V is 30xRank for 2x15
            V_full = np.zeros((30, rank * 5))
            W_full = np.zeros((30, rank * 5))
            
            for b in range(5):
                offset = b * rank
                U_full[:, offset:offset+rank] = U_snap
                V_full[3*b:3*b+3, offset:offset+rank] = V_snap[0:3, :]
                V_full[15+3*b:15+3*b+3, offset:offset+rank] = V_snap[3:6, :]
                W_full[3*b:3*b+3, offset:offset+rank] = W_snap[0:3, :]
                W_full[15+3*b:15+3*b+3, offset:offset+rank] = W_snap[3:6, :]
                
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
    except Exception:
        pass
        
    return get_fallback()
