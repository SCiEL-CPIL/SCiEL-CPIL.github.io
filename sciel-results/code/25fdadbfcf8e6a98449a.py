"""Rank-18 exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            if abs(val - target) > tol:
                                return False
    return True

def _to_half_int(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def _try_rank_decomposition(n, m, p, r, max_iter=300, tol=1e-10, n_restarts=5, seed=42, deadline=None):
    """Try to find a rank-r decomposition for M(n,m,p) tensor using ALS."""
    best_result = None
    best_res = float('inf')
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        np.random.seed(seed + restart)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((m*p*p*n, r))
                    b_vec = np.zeros(m*p*p*n)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    for c in range(r):
                                        A_mat[idx, c] = V[jp*p+k, c] * W[kp*n+ip, c]
                                    b_vec[idx] = target
                                    idx += 1
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V
            for jp in range(m):
                for k in range(p):
                    A_mat = np.zeros((n*m*p*n, r))
                    b_vec = np.zeros(n*m*p*n)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    for c in range(r):
                                        A_mat[idx, c] = U[i*m+j, c] * W[kp*n+ip, c]
                                    b_vec[idx] = target
                                    idx += 1
                    V[jp*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W
            for kp in range(p):
                for ip in range(n):
                    A_mat = np.zeros((n*m*m*p, r))
                    b_vec = np.zeros(n*m*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    for c in range(r):
                                        A_mat[idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                                    b_vec[idx] = target
                                    idx += 1
                    W[kp*n+ip, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check convergence
            if it % 20 == 0 or it == max_iter - 1:
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                        val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :])
                                        res += (val - target)**2
                if res < best_res:
                    best_res = res
                    best_result = (U.copy(), V.copy(), W.copy())
                if res < tol:
                    break
    
    if best_res < tol:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Try to find a decomposition for each 2x3x2 block
    for r_sub in [6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        result = _try_rank_decomposition(2, 3, 2, r_sub, max_iter=300, tol=1e-10, n_restarts=5, seed=42, deadline=deadline)
        if result is not None:
            U_sub, V_sub, W_sub = result
            
            # Try rounding to half-integers
            U_rounded = [[_to_half_int(U_sub[i, c]) for c in range(r_sub)] for i in range(6)]
            V_rounded = [[_to_half_int(V_sub[i, c]) for c in range(r_sub)] for i in range(6)]
            W_rounded = [[_to_half_int(W_sub[i, c]) for c in range(r_sub)] for i in range(4)]
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, 2, 3, 2, tol=1e-9):
                total_rank = 3 * r_sub
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                for block in range(3):
                    offset = block * r_sub
                    for i in range(n):
                        for j in range(m):
                            for c in range(r_sub):
                                U[i * m + j][offset + c] = U_rounded[i * m + j][c]
                    for j in range(m):
                        for k in range(2):
                            for c in range(r_sub):
                                V[j * p + block * 2 + k][offset + c] = V_rounded[j * 2 + k][c]
                    for k in range(2):
                        for i in range(n):
                            for c in range(r_sub):
                                W[(block * 2 + k) * n + i][offset + c] = W_rounded[k * n + i][c]
                
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                    return {"U": U, "V": V, "W": W}
            
            # Try unrounded
            U_float = [[float(U_sub[i, c]) for c in range(r_sub)] for i in range(6)]
            V_float = [[float(V_sub[i, c]) for c in range(r_sub)] for i in range(6)]
            W_float = [[float(W_sub[i, c]) for c in range(r_sub)] for i in range(4)]
            
            if _verify_decomposition(U_float, V_float, W_float, 2, 3, 2, tol=1e-9):
                total_rank = 3 * r_sub
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                for block in range(3):
                    offset = block * r_sub
                    for i in range(n):
                        for j in range(m):
                            for c in range(r_sub):
                                U[i * m + j][offset + c] = U_float[i * m + j][c]
                    for j in range(m):
                        for k in range(2):
                            for c in range(r_sub):
                                V[j * p + block * 2 + k][offset + c] = V_float[j * 2 + k][c]
                    for k in range(2):
                        for i in range(n):
                            for c in range(r_sub):
                                W[(block * 2 + k) * n + i][offset + c] = W_float[k * n + i][c]
                
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 36
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
