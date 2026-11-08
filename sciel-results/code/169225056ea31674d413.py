import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nA, nB, nC = n * m, m * p, n * p
    
    # Build tensor for verification
    T = np.zeros((nA, nB, nC), dtype=np.float64)
    for a in range(nA):
        ai, aj = divmod(a, m)
        for b in range(nB):
            bi, bj = divmod(b, p)
            if aj == bi:
                c = ai * p + bj
                T[a, b, c] = 1.0
    
    # Try rank 21 with a quick search
    if time.time() < deadline - 40:
        result = _search_rank(T, 21, nA, nB, nC)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 22
    rank = 22
    U = np.zeros((nA, rank))
    V = np.zeros((nB, rank))
    W = np.zeros((nC, rank))
    
    def add_term(u_idx, v_idx, w_idx, col):
        for idx, val in u_idx:
            U[idx, col] += val
        for idx, val in v_idx:
            V[idx, col] += val
        for idx, val in w_idx:
            W[idx, col] += val
    
    # Block 1
    add_term([(0,1),(4,1)], [(0,1),(5,1)], [(0,1),(3,1)], 0)
    add_term([(3,1),(4,1)], [(0,1)], [(1,1),(3,-1)], 1)
    add_term([(0,1)], [(1,1),(5,-1)], [(2,1),(3,1)], 2)
    add_term([(4,1)], [(4,1),(0,-1)], [(0,1),(1,1)], 3)
    add_term([(0,1),(1,1)], [(5,1)], [(0,-1),(2,1)], 4)
    add_term([(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)], 5)
    add_term([(1,1),(4,-1)], [(4,1),(5,1)], [(0,1)], 6)
    
    # Block 2
    add_term([(0,1),(4,1)], [(2,1),(7,1)], [(4,1),(7,1)], 7)
    add_term([(3,1),(4,1)], [(2,1)], [(5,1),(7,-1)], 8)
    add_term([(0,1)], [(3,1),(7,-1)], [(6,1),(7,1)], 9)
    add_term([(4,1)], [(6,1),(2,-1)], [(4,1),(5,1)], 10)
    add_term([(0,1),(1,1)], [(7,1)], [(4,-1),(6,1)], 11)
    add_term([(3,1),(0,-1)], [(2,1),(3,1)], [(7,1)], 12)
    add_term([(1,1),(4,-1)], [(6,1),(7,1)], [(4,1)], 13)
    
    # Block 3
    add_term([(2,1)], [(8,1)], [(0,1)], 14)
    add_term([(2,1)], [(9,1)], [(2,1)], 15)
    add_term([(2,1)], [(10,1)], [(4,1)], 16)
    add_term([(2,1)], [(11,1)], [(6,1)], 17)
    add_term([(5,1)], [(8,1)], [(1,1)], 18)
    add_term([(5,1)], [(9,1)], [(3,1)], 19)
    add_term([(5,1)], [(10,1)], [(5,1)], 20)
    add_term([(5,1)], [(11,1)], [(7,1)], 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank(T, rank, nA, nB, nC):
    """Quick search for rank-`rank` decomposition."""
    rng = np.random.default_rng(999)
    
    for seed in range(30):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((nA, rank))
        V = rng.standard_normal((nB, rank))
        W = rng.standard_normal((nC, rank))
        
        for it in range(80):
            VW = V[:, :, None] * W[None, :, :]
            VW_mat = VW.transpose(2, 0, 1).reshape(rank, nB * nC)
            T_mat = T.reshape(nA, nB * nC)
            try:
                U = T_mat @ np.linalg.pinv(VW_mat.T)
            except:
                break
            
            UW = U[:, :, None] * W[None, :, :]
            UW_mat = UW.transpose(2, 0, 1).reshape(rank, nA * nC)
            T_mat_v = T.transpose(1, 0, 2).reshape(nB, nA * nC)
            try:
                V = T_mat_v @ np.linalg.pinv(UW_mat.T)
            except:
                break
            
            UV = U[:, :, None] * V[None, :, :]
            UV_mat = UV.transpose(2, 0, 1).reshape(rank, nA * nB)
            T_mat_w = T.transpose(2, 0, 1).reshape(nC, nA * nB)
            try:
                W = T_mat_w @ np.linalg.pinv(UV_mat.T)
            except:
                break
        
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(T_recon, T, atol=1e-8):
            for scale in [1, 2, 4]:
                Ur = np.round(U * scale) / scale
                Vr = np.round(V * scale) / scale
                Wr = np.round(W * scale) / scale
                if np.allclose(np.einsum('ak,bk,ck->abc', Ur, Vr, Wr), T, atol=1e-6):
                    return (Ur, Vr, Wr)
    
    return None
