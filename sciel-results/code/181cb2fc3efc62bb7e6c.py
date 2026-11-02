import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_block_decomp(partition, decomps, n=2, m=2, p_total=9):
        R_total = sum(d[0].shape[1] for d in decomps)
        U_full = np.zeros((n*m, R_total))
        V_full = np.zeros((m*p_total, R_total))
        W_full = np.zeros((p_total*n, R_total))
        
        r_offset = 0
        col_offset = 0
        for b, size in enumerate(partition):
            U_b, V_b, W_b = decomps[b]
            R_b = U_b.shape[1]
            r_end = r_offset + R_b
            
            U_full[:, r_offset:r_end] = U_b
            
            for k in range(m):
                for j_local in range(size):
                    j_global = col_offset + j_local
                    v_idx_block = k*size + j_local
                    v_idx_full = k*p_total + j_global
                    V_full[v_idx_full, r_offset:r_end] = V_b[v_idx_block, :]
            
            for j_local in range(size):
                j_global = col_offset + j_local
                for i in range(n):
                    w_idx_block = j_local*n + i
                    w_idx_full = j_global*n + i
                    W_full[w_idx_full, r_offset:r_end] = W_b[w_idx_block, :]
            
            r_offset = r_end
            col_offset += size
        
        return U_full, V_full, W_full

    def search_rank_22p(p_size, R, max_restarts=20, max_nfev=3000):
        """Search for rank-R decomposition of 2x2xp_size matrix multiplication."""
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+2*p_size*R].reshape(2*p_size, R)
            W = params[4*R+2*p_size*R:4*R+4*p_size*R].reshape(2*p_size, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, p_size, R)
            W_r = W.reshape(p_size, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return (T - 1.0).flatten()
        
        def residuals_sq(params):
            r = residuals(params)
            return np.sum(r*r)
        
        best_sol = None
        best_cost = float('inf')
        
        # Try multiple initializations
        for attempt in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            if attempt == 0:
                # Structured initialization: start from schoolbook and compress
                x0 = np.zeros(8*p_size*R)
                # Initialize with a pattern that might work
                for r in range(R):
                    # Distribute mass across entries
                    for idx in range(4):
                        x0[idx*R + r] = 1.0 / np.sqrt(R)
                    for idx in range(2*p_size):
                        x0[4*R + idx*R + r] = 1.0 / np.sqrt(R)
                    for idx in range(2*p_size):
                        x0[4*R + 2*p_size*R + idx*R + r] = 1.0 / np.sqrt(R)
            elif attempt == 1:
                # Random with moderate scale
                x0 = np.random.randn(8*p_size*R) * 0.3
            elif attempt == 2:
                # Sparse-like initialization
                x0 = np.zeros(8*p_size*R)
                for r in range(R):
                    # Pick random entries to be nonzero
                    u_idx = np.random.choice(4, size=min(2, 4), replace=False)
                    for idx in u_idx:
                        x0[idx*R + r] = np.random.choice([-1, 1])
                    v_idx = np.random.choice(2*p_size, size=min(2, 2*p_size), replace=False)
                    for idx in v_idx:
                        x0[4*R + idx*R + r] = np.random.choice([-1, 1])
                    w_idx = np.random.choice(2*p_size, size=min(2, 2*p_size), replace=False)
                    for idx in w_idx:
                        x0[4*R + 2*p_size*R + idx*R + r] = np.random.choice([-1, 1])
            else:
                x0 = np.random.randn(8*p_size*R) * (0.2 + 0.1 * attempt)
            
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev, 
                                    ftol=1e-12, xtol=1e-12, gtol=1e-12)
                cost = res.cost
                if cost < best_cost:
                    best_cost = cost
                    best_sol = res.x.copy()
                    if cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_sol is not None and best_cost < 1e-6:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+2*p_size*R].reshape(2*p_size, R)
            W = best_sol[4*R+2*p_size*R:4*R+4*p_size*R].reshape(2*p_size, R)
            
            # Try snapping to integers first, then half-integers
            for snap_factor in [1, 2, 4]:
                U_snap = np.round(U * snap_factor) / snap_factor
                V_snap = np.round(V * snap_factor) / snap_factor
                W_snap = np.round(W * snap_factor) / snap_factor
                if verify_decomp(U_snap, V_snap, W_snap, 2, 2, p_size):
                    return U_snap, V_snap, W_snap
            
            # If snapping fails, return the float solution (should still be valid)
            if verify_decomp(U, V, W, 2, 2, p_size):
                return U, V, W
        
        return None

    def get_strassen_222():
        """Standard Strassen rank-7 decomposition for 2x2x2."""
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

    def get_schoolbook_221():
        """Rank-4 schoolbook for 2x2x1."""
        U = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        V = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        W = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        return U, V, W

    best_result = None
    best_rank = 999

    # Strategy 1: Try 3 blocks of 3, target rank 8 each → total 24
    if time.time() < deadline - 40:
        decomp_3 = search_rank_22p(3, 8, max_restarts=25, max_nfev=4000)
        if decomp_3 is not None:
            decomps = [decomp_3, decomp_3, decomp_3]
            partition = [3, 3, 3]
            rank = 24
            if rank < best_rank:
                best_rank = rank
                U_full, V_full, W_full = build_block_decomp(partition, decomps, n, m, p)
                if verify_decomp(U_full, V_full, W_full, n, m, p):
                    best_result = (U_full, V_full, W_full)

    # Strategy 2: Try 2 blocks of 4 + 1 block of 1, target rank 11 each for 4-blocks → total 26
    if best_rank > 26 and time.time() < deadline - 25:
        decomp_4 = search_rank_22p(4, 11, max_restarts=20, max_nfev=3000)
        if decomp_4 is not None:
            decomp_1 = get_schoolbook_221()
            decomps = [decomp_4, decomp_4, decomp_1]
            partition = [4, 4, 1]
            rank = 2*11 + 4
            if rank < best_rank:
                best_rank = rank
                U_full, V_full, W_full = build_block_decomp(partition, decomps, n, m, p)
                if verify_decomp(U_full, V_full, W_full, n, m, p):
                    best_result = (U_full, V_full, W_full)

    # Strategy 3: Fallback to known rank-32 Strassen-based construction
    if best_result is None:
        U4, V4, W4 = get_strassen_222()
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
        
        decomp_1 = get_schoolbook_221()
        decomps = [(U14, V14, W14), (U14, V14, W14), decomp_1]
        partition = [4, 4, 1]
        rank = 2*14 + 4
        U_full, V_full, W_full = build_block_decomp(partition, decomps, n, m, p)
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            best_result = (U_full, V_full, W_full)
            best_rank = rank

    # Ultimate fallback: hardcoded rank-32
    if best_result is None:
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

    U_full, V_full, W_full = best_result
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
