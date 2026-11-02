import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    def solve_2x3x1(rank_target=4, n_restarts=50, max_iters=100):
        best_err = float('inf')
        best = None
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            U = np.random.randn(6, rank_target) * 0.5
            V = np.random.randn(3, rank_target) * 0.5
            W = np.random.randn(2, rank_target) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                # Update U
                for a in range(6):
                    i = a // 3
                    j = a % 3
                    vec = V[j] * W[i]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-9:
                        U[a] = vec / norm_sq
                    else:
                        U[a] = 0
                        
                # Update V
                for b in range(3):
                    j = b
                    A = np.array([U[i*3+j] * W[i] for i in range(2)])
                    bvec = np.ones(2)
                    try:
                        V[j] = np.linalg.solve(A, bvec)
                    except np.linalg.LinAlgError:
                        V[j] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                        
                # Update W
                for c in range(2):
                    i = c
                    A = np.array([U[i*3+j] * V[j] for j in range(3)])
                    bvec = np.ones(3)
                    try:
                        W[i] = np.linalg.solve(A, bvec)
                    except np.linalg.LinAlgError:
                        W[i] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                
                # Compute error
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        val = np.dot(U[i*3+j], V[j] * W[i])
                        err += (val - 1.0)**2
                        for c in range(2):
                            if c != i:
                                val = np.dot(U[i*3+j], V[j] * W[c])
                                err += val**2
                            
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                if err < 1e-12:
                    break
        
        if best and best_err < 1e-9:
            return best
        return None

    res = solve_2x3x1(4, 100, 200)
    
    if res:
        U_block, V_block, W_block = res
        rank = 36
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        
        for k in range(9):
            offset = k * 4
            U[:, offset:offset+4] = U_block
            for j in range(3):
                V[j*9+k, offset:offset+4] = V_block[j]
            for i in range(2):
                W[k*2+i, offset:offset+4] = W_block[i]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 54
    rank = 54
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(27)]
    W = [[0]*rank for _ in range(18)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(9):
                U[i*3+j][col] = 1
                V[j*9+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
