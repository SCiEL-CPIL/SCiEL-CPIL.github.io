"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute indices for the tensor
    ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    U_idx = ii * m + jj
    V_idx = jj * p + kk
    W_idx = kk * n + ii
    
    def verify(U, V, W):
        U_sel = U[U_idx]
        V_sel = V[V_idx]
        W_sel = W[W_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)
    
    def try_rank_with_linear_W(rank, max_time=10):
        """Fix U,V randomly with half-integer entries, solve for W linearly."""
        deadline_local = min(deadline, time.time() + max_time)
        np.random.seed(42)
        candidates = np.array([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5])
        
        for attempt in range(500):
            if time.time() > deadline_local:
                break
            U = np.random.choice(candidates, size=(n*m, rank))
            V = np.random.choice(candidates, size=(m*p, rank))
            
            A = U[U_idx] * V[V_idx]  # (84, rank)
            b = np.ones(84)
            
            M = np.zeros((84, p*n*rank))
            for t in range(84):
                row = W_idx[t]
                for c in range(rank):
                    M[t, row*rank + c] = A[t, c]
            
            try:
                W_flat, residuals, rank_svd, singular_vals = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            
            W = W_flat.reshape(p*n, rank)
            W_round = np.round(W * 2) / 2
            W_err = np.max(np.abs(W - W_round))
            
            if W_err < 1e-9 and verify(U, V, W_round):
                return U.tolist(), V.tolist(), W_round.tolist()
        
        return None
    
    best_result = None
    for rank in [28, 30, 35, 40, 50, 60, 84]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = try_rank_with_linear_W(rank, max_time=min(15, remaining - 5))
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
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
# EVOLVE-BLOCK-END
