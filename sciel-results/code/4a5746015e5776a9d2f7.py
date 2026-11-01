import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_full_decomp(partition, decomps):
        R_total = sum(d[0].shape[1] for d in decomps)
        U_full = np.zeros((4, R_total))
        V_full = np.zeros((18, R_total))
        W_full = np.zeros((18, R_total))
        
        r_offset = 0
        col_offset = 0
        for b, size in enumerate(partition):
            U_b, V_b, W_b = decomps[b]
            R_b = U_b.shape[1]
            r_end = r_offset + R_b
            
            U_full[:, r_offset:r_end] = U_b
            
            for k in range(2):
                for j_local in range(size):
                    j_global = col_offset + j_local
                    v_idx_block = k*size + j_local
                    v_idx_full = k*9 + j_global
                    V_full[v_idx_full, r_offset:r_end] = V_b[v_idx_block, :]
            
            for j_local in range(size):
                j_global = col_offset + j_local
                for i in range(2):
                    w_idx_block = j_local*2 + i
                    w_idx_full = j_global*2 + i
                    W_full[w_idx_full, r_offset:r_end] = W_b[w_idx_block, :]
            
            r_offset = r_end
            col_offset += size
        
        return U_full, V_full, W_full

    def search_rank_22p(p_size, R, max_restarts=8):
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+2*p_size*R].reshape(2*p_size, R)
            W = params[4*R+2*p_size*R:4*R+4*p_size*R].reshape(2*p_size, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, p_size, R)
            W_r = W.reshape(p_size, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return T.flatten() - 1.0
        
        best_sol = None
        best_cost = float('inf')
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            x0 = np.random.randn(8*p_size*R) * 0.5
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=2000)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if res.cost < 1e-10:
                        break
            except:
                continue
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+2*p_size*R].reshape(2*p_size, R)
            W = best_sol[4*R+2*p_size*R:4*R+4*p_size*R].reshape(2*p_size, R)
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, p_size):
                return U_snapped, V_snapped, W_snapped
        return None

    def get_block_2x1():
        # Rank 4 decomposition for M_{2,2,1}
        U = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        V = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        W = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        return U, V, W

    def get_block_2x2_strassen():
        # Rank 7 Strassen for M_{2,2,2}
        U = np.array([
            [1, 1, 1, 0, 1, -1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        V = np.array([
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [1, 0, -1, 1, 0, 0, 0],
            [0, 0, 1, -1, 1, 0, 1]
        ], dtype=float)
        W = np.array([
            [1, -1, 0, 1, 0, 0, -1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [1, 1, -1, 0, 0, 1, 0]
        ], dtype=float)
        return U, V, W

    # Strategy: partition 9 columns as 4+4+1
    # Try to find rank-12 for M_{2,2,4}, giving total rank 2*12+4 = 28
    # Fallback: rank-14 (two Strassen) for M_{2,2,4}, giving total rank 2*14+4 = 32

    best_partition = None
    best_decomps = None
    best_rank = 32

    # Try rank 12 for M_{2,2,4}
    if time.time() < deadline:
        decomp_4 = search_rank_22p(4, 12, max_restarts=10)
        if decomp_4 is not None:
            decomp_1 = get_block_2x1()
            decomps = [decomp_4, decomp_4, decomp_1]
            partition = [4, 4, 1]
            rank = 2*12 + 4
            if rank < best_rank:
                best_rank = rank
                best_partition = partition
                best_decomps = decomps

    # Try rank 13 for M_{2,2,4} as fallback
    if best_rank > 29 and time.time() < deadline:
        decomp_4 = search_rank_22p(4, 13, max_restarts=5)
        if decomp_4 is not None:
            decomp_1 = get_block_2x1()
            decomps = [decomp_4, decomp_4, decomp_1]
            partition = [4, 4, 1]
            rank = 2*13 + 4
            if rank < best_rank:
                best_rank = rank
                best_partition = partition
                best_decomps = decomps

    # Fallback: use Strassen blocks (rank 14 for M_{2,2,4})
    if best_rank > 32 and time.time() < deadline:
        U4, V4, W4 = get_block_2x2_strassen()
        # Build rank-14 for M_{2,2,4} by combining two Strassen blocks
        # Columns 0,1 use first Strassen, columns 2,3 use second Strassen
        U14 = np.zeros((4, 14))
        V14 = np.zeros((8, 14))
        W14 = np.zeros((8, 14))
        
        # First Strassen for cols 0,1
        U14[:, 0:7] = U4
        for k in range(2):
            for j in range(2):
                V14[k*4+j, 0:7] = V4[k*2+j, :]
        for j in range(2):
            for i in range(2):
                W14[j*2+i, 0:7] = W4[j*2+i, :]
        
        # Second Strassen for cols 2,3
        U14[:, 7:14] = U4
        for k in range(2):
            for j in range(2):
                V14[k*4+(2+j), 7:14] = V4[k*2+j, :]
        for j in range(2):
            for i in range(2):
                W14[(2+j)*2+i, 7:14] = W4[j*2+i, :]
        
        decomp_1 = get_block_2x1()
        decomps = [U14, V14, W14, decomp_1]
        # Wait, decomps should be list of (U,V,W) tuples
        decomps = [(U14, V14, W14), (U14, V14, W14), decomp_1]
        partition = [4, 4, 1]
        rank = 2*14 + 4
        if rank < best_rank:
            best_rank = rank
            best_partition = partition
            best_decomps = decomps

    if best_partition is not None:
        U_full, V_full, W_full = build_full_decomp(best_partition, best_decomps)
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Ultimate fallback: Rank 32 hardcoded
    terms = []
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
        terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
        terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
        terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
        terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
    
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
