import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s budget, 10s safety margin
    
    # Construct target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    nm, mp, pn = n * m, m * p, p * n
    
    def solve_for_U(V, W, rank):
        B = np.zeros((rank, mp * pn))
        for r in range(rank):
            for b in range(mp):
                for c in range(pn):
                    B[r, b * pn + c] = V[b, r] * W[c, r]
        B_pinv = np.linalg.pinv(B)
        U = np.zeros((nm, rank))
        for a in range(nm):
            U[a, :] = T[a].ravel() @ B_pinv
        return U
    
    def solve_for_V(U, W, rank):
        B = np.zeros((rank, nm * pn))
        for r in range(rank):
            for a in range(nm):
                for c in range(pn):
                    B[r, a * pn + c] = U[a, r] * W[c, r]
        B_pinv = np.linalg.pinv(B)
        V = np.zeros((mp, rank))
        for b in range(mp):
            V[b, :] = T[:, b, :].ravel() @ B_pinv
        return V
    
    def solve_for_W(U, V, rank):
        B = np.zeros((rank, nm * mp))
        for r in range(rank):
            for a in range(nm):
                for b in range(mp):
                    B[r, a * mp + b] = U[a, r] * V[b, r]
        B_pinv = np.linalg.pinv(B)
        W = np.zeros((pn, rank))
        for c in range(pn):
            W[c, :] = T[:, :, c].ravel() @ B_pinv
        return W
    
    def check_decomposition(U, V, W, tol=1e-6):
        rank = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_rec - T)) < tol
    
    def try_rank(rank, max_iters=50, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        for it in range(max_iters):
            U = solve_for_U(V, W, rank)
            V = solve_for_V(U, W, rank)
            W = solve_for_W(U, V, rank)
            if check_decomposition(U, V, W):
                return U, V, W
        return None
    
    best_result = None
    for rank in range(20, 49):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_rank(rank, max_iters=30, seed=seed)
            if result is not None:
                U, V, W = result
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                    best_result = (rank, U_r, V_r, W_r)
                else:
                    # Try integers
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                        best_result = (rank, U_r, V_r, W_r)
                    else:
                        best_result = (rank, U, V, W)
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(nm)]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(mp)]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
