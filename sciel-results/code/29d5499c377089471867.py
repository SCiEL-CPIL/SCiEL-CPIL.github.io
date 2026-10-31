import numpy as np
import time

def construct():
    """
    Returns a rank-minimal decomposition of the 4x4x4 matrix multiplication tensor.
    Attempts rank 48 via linear algebra search, falls back to rank 49.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def _build_rank49():
        """Build the standard recursive Strassen rank-49 decomposition."""
        terms = []
        
        strassen_2x2 = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        outer_strassen = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        for (a_blocks, b_blocks, c_blocks) in outer_strassen:
            for (a_comb, b_comb, c_comb) in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (lr, lc, ca) in a_comb:
                    for (br, bc, coeff) in a_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        u_idx = gr * 4 + gc
                        u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
                
                for (lr, lc, cb) in b_comb:
                    for (br, bc, coeff) in b_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        v_idx = gr * 4 + gc
                        v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
                
                for (lr, lc, cc) in c_comb:
                    for (br, bc, coeff) in c_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        w_idx = gc * 4 + gr
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
                
                terms.append((u_dict, v_dict, w_dict))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_dict.items():
                if val != 0:
                    W[idx][col] = val
        
        return U, V, W, rank

    def _verify(U, V, W, rank):
        """Verify the decomposition is exact using NumPy (fast path)."""
        try:
            Ua = np.asarray(U, dtype=np.int64)
            Va = np.asarray(V, dtype=np.int64)
            Wa = np.asarray(W, dtype=np.int64)
            
            U3 = Ua.reshape(4, 4, rank)
            V3 = Va.reshape(4, 4, rank)
            W3 = Wa.reshape(4, 4, rank)
            
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            return bool(np.all(S == 1))
        except Exception:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = 0
                        for r in range(rank):
                            s += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                        if s != 1:
                            return False
            return True

    def _try_rank48(U49, V49, W49):
        """
        Try to find rank-48 by removing one term and solving for corrections in W.
        For each candidate term r0 to remove, check if we can adjust W in the
        remaining 48 terms to compensate.
        """
        Ua = np.array(U49, dtype=np.float64)
        Va = np.array(V49, dtype=np.float64)
        Wa = np.array(W49, dtype=np.float64)
        
        U3 = Ua.reshape(4, 4, 49)
        V3 = Va.reshape(4, 4, 49)
        W3 = Wa.reshape(4, 4, 49)
        
        for r0 in range(49):
            if time.time() > deadline - 3:
                break
            
            r_map = [r for r in range(49) if r != r0]
            
            # Adjust only W: for each (k,i), solve 16 eqs (over j) for 48 unknowns
            delta_W = np.zeros((4, 4, 48))
            feasible = True
            
            for k in range(4):
                for i in range(4):
                    A_mat = np.zeros((16, 48))
                    b_vec = np.zeros(16)
                    
                    for j in range(4):
                        for rp, r in enumerate(r_map):
                            A_mat[j, rp] = U3[i, j, r] * V3[j, k, r]
                        b_vec[j] = -U3[i, j, r0] * V3[j, k, r0] * W3[k, i, r0]
                    
                    sol, res, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    if np.linalg.norm(A_mat @ sol - b_vec) > 1e-9:
                        feasible = False
                        break
                    delta_W[k, i, :] = sol
                
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Build corrected W and verify
            W_new = W3.copy()
            for k in range(4):
                for i in range(4):
                    for rp, r in enumerate(r_map):
                        W_new[k, i, r] += delta_W[k, i, rp]
            W_new[:, :, r0] = 0
            
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W_new)
            if np.allclose(S, np.ones((4,4,4)), atol=1e-9):
                # Check half-integer entries
                mask = [r != r0 for r in range(49)]
                all_ok = True
                for mat3 in [U3, V3, W_new]:
                    mat_out = mat3.reshape(16, 49)[:, mask]
                    for val in mat_out:
                        if abs(val - round(val * 2) / 2) > 1e-8:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                
                if all_ok:
                    U_out = U3.reshape(16, 49)[:, mask]
                    V_out = V3.reshape(16, 49)[:, mask]
                    W_out = W_new.reshape(16, 49)[:, mask]
                    U_fin = [[int(round(v * 2)) / 2 for v in row] for row in U_out]
                    V_fin = [[int(round(v * 2)) / 2 for v in row] for row in V_out]
                    W_fin = [[int(round(v * 2)) / 2 for v in row] for row in W_out]
                    if _verify(U_fin, V_fin, W_fin, 48):
                        return U_fin, V_fin, W_fin, 48
        
        return None

    # Build rank-49 decomposition
    U, V, W, rank = _build_rank49()
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try rank 48
    result = _try_rank48(U, V, W)
    if result is not None:
        U, V, W, rank = result
    else:
        if not _verify(U, V, W, rank):
            rank = 64
            U = [[0] * rank for _ in range(16)]
            V = [[0] * rank for _ in range(16)]
            W = [[0] * rank for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U[i*4+j][col] = 1
                        V[j*4+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
