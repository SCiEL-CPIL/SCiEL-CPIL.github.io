import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = n * m * p
    
    I = np.repeat(np.arange(n), m * p)
    J = np.tile(np.repeat(np.arange(m), p), n)
    K = np.tile(np.arange(p), n * m)
    
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I
    
    def check_decomp(U, V, W):
        rank = U.shape[1]
        total = np.zeros(nmp)
        for c in range(rank):
            total += U[U_idx, c] * V[V_idx, c] * W[W_idx, c]
        return np.max(np.abs(total - 1.0))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    for rank in range(40, best_rank):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            seed = attempt * 104729 + rank * 7919
            rng = np.random.RandomState(seed)
            
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 20:
                    break
                
                total = np.zeros(nmp)
                for c in range(rank):
                    total += U[U_idx, c] * V[V_idx, c] * W[W_idx, c]
                err = total - 1.0
                max_err = np.max(np.abs(err))
                
                if max_err < 1e-10:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(rank):
                    uc = U[U_idx, c]
                    vc = V[V_idx, c]
                    wc = W[W_idx, c]
                    
                    np.add.at(dU, (U_idx, c), err * vc * wc)
                    np.add.at(dV, (V_idx, c), err * uc * wc)
                    np.add.at(dW, (W_idx, c), err * uc * vc)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 500 == 499:
                    lr *= 0.5
            
            final_err = check_decomp(U, V, W)
            if final_err < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomp(U_r, V_r, W_r) < 1e-6:
                    best_result = (U_r, V_r, W_r)
                else:
                    best_result = (U, V, W)
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    U, V, W = best_result
    
    U_list = [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
