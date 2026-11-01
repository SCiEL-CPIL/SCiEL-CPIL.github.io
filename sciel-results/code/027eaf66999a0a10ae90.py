import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                triples.append((A, B, C))
                
    T_target = np.zeros((n*m, m*p, p*n))
    for A, B, C in triples:
        T_target[A, B, C] = 1.0

    def verify(U, V, W):
        T_recon = np.einsum('ar, br, cr -> abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=1e-6)

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j * p + k
                    C = k * n + i
                    U[A, col] = 1
                    V[B, col] = 1
                    W[C, col] = 1
                    col += 1
        return U, V, W

    def search_rank(rank, time_limit=10):
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        for attempt in range(30):
            if time.time() - start_time > time_limit:
                break
            
            U = rng.integers(-2, 3, size=(n*m, rank)).astype(float)
            V = rng.integers(-2, 3, size=(m*p, rank)).astype(float)
            W = rng.integers(-2, 3, size=(p*n, rank)).astype(float)
            
            lr = 0.05
            for step in range(300):
                if time.time() - start_time > time_limit:
                    break
                T_recon = np.einsum('ar, br, cr -> abc', U, V, W)
                error = T_recon - T_target
                err_norm = np.linalg.norm(error, 'fro')
                if err_norm < 1e-5:
                    # Try snapping to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if verify(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    if verify(U, V, W):
                        return U, V, W
                    break
                
                grad_U = np.einsum('abc, br, cr -> ar', error, V, W)
                grad_V = np.einsum('abc, ar, cr -> br', error, U, W)
                grad_W = np.einsum('abc, ar, br -> cr', error, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
        return None

    start_total = time.time()
    # Try ranks 15, 14, 13, 12
    for rank in [15, 14, 13, 12]:
        if time.time() - start_total > 110:
            break
        result = search_rank(rank, time_limit=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
