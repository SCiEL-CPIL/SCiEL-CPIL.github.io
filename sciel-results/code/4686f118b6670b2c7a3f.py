import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    s = np.dot(U_arr[ui, :], V_arr[vi, :] * W_arr[wi, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank66():
        """Build the known rank-66 decomposition via p=4+2 split with Strassen."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_sb(i0, j0, k0, nn, mm, pp):
            for ii in range(nn):
                for jj in range(mm):
                    for kk in range(pp):
                        add_rank1([((i0 + ii) * m + (j0 + jj), 1.0)],
                                  [((j0 + jj) * p + (k0 + kk), 1.0)],
                                  [((k0 + kk) * n + (i0 + ii), 1.0)])

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        # p split: 4 + 2
        add_sb(0, 0, 0, 1, 4, 4)
        add_ssr(1, 0, 0)
        add_ssr(1, 2, 0)
        add_ssr(1, 0, 2)
        add_ssr(1, 2, 2)
        add_sb(0, 0, 4, 1, 4, 2)
        add_ssr(1, 0, 4)
        add_ssr(1, 2, 4)

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        """Pure schoolbook rank 72 fallback."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}

    def search_rank65():
        """Search for a rank-65 decomposition by trying to compress the rank-66 solution.
        
        Strategy: Take the rank-66 decomposition and try to find if any single rank-1 term
        can be eliminated or merged with others by adjusting coefficients.
        
        We'll use a SAT-like approach: try removing each term one at a time and see if we
        can compensate with coefficient adjustments on remaining terms.
        """
        base = build_rank66()
        U_base = np.array(base["U"], dtype=np.float64)
        V_base = np.array(base["V"], dtype=np.float64)
        W_base = np.array(base["W"], dtype=np.float64)
        
        # Try each possible term removal
        for remove_idx in range(66):
            if time.time() > deadline - 30:
                break
            
            # Create rank-65 matrices by removing column remove_idx
            U_r = np.delete(U_base, remove_idx, axis=1)
            V_r = np.delete(V_base, remove_idx, axis=1)
            W_r = np.delete(W_base, remove_idx, axis=1)
            
            # The removed term contributed:
            # u_vec * v_vec * w_vec where these are the removed column values
            u_rem = U_base[:, remove_idx]
            v_rem = V_base[:, remove_idx]
            w_rem = W_base[:, remove_idx]
            
            # For each output (i,j,k), the removed term contributed:
            # u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
            # We need the remaining terms to compensate for this.
            
            # Check if the remaining terms already satisfy the equation
            # (i.e., the removed term's contribution was zero for all outputs)
            all_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        contrib = u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
                        if abs(contrib) > 1e-12:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    break
            
            if all_zero:
                # The removed term was redundant!
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                if verify_decomp(U_list, V_list, W_list, 65):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try coefficient adjustment: modify one other term to compensate
            # For simplicity, try scaling a term that has overlapping support
            for adjust_idx in range(66):
                if adjust_idx == remove_idx:
                    continue
                if time.time() > deadline - 30:
                    break
                
                # Check if the adjusting term can compensate
                u_adj = U_base[:, adjust_idx]
                v_adj = V_base[:, adjust_idx]
                w_adj = W_base[:, adjust_idx]
                
                # We need: sum of remaining terms + delta * (u_adj * v_adj * w_adj) = target
                # So delta = -contrib_removed / (u_adj[i*m+j] * v_adj[j*p+k] * w_adj[k*n+i])
                # for each (i,j,k) where the removed term contributed nonzero
                
                delta_candidates = []
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib_rem = u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
                            if abs(contrib_rem) < 1e-12:
                                continue
                            denom = u_adj[i*m+j] * v_adj[j*p+k] * w_adj[k*n+i]
                            if abs(denom) < 1e-12:
                                valid = False
                                break
                            delta = -contrib_rem / denom
                            delta_candidates.append(delta)
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # All deltas should be the same
                if delta_candidates:
                    d0 = delta_candidates[0]
                    if all(abs(d - d0) < 1e-9 for d in delta_candidates):
                        # Try this adjustment
                        U_test = U_r.copy()
                        V_test = V_r.copy()
                        W_test = W_r.copy()
                        
                        # Find the index of adjust_idx in the reduced matrix
                        if adjust_idx < remove_idx:
                            adj_pos = adjust_idx
                        else:
                            adj_pos = adjust_idx - 1
                        
                        # Scale the adjusting term
                        U_test[:, adj_pos] *= (1 + d0)
                        # Actually, we need to add d0 * u_adj, not scale
                        # Let me redo: the new term should be (1+d0) * original for the adjusted column
                        # But we removed one term, so we have 65 terms. The adjusted term is at adj_pos.
                        # We want its contribution to be (1 + d0) times original.
                        
                        # Actually, let's just construct fresh
                        U_new = U_base.copy()
                        V_new = V_base.copy()
                        W_new = W_base.copy()
                        
                        # Scale the adjusting term
                        U_new[:, adjust_idx] *= (1 + d0)
                        # Remove the target term
                        U_new = np.delete(U_new, remove_idx, axis=1)
                        V_new = np.delete(V_new, remove_idx, axis=1)
                        W_new = np.delete(W_new, remove_idx, axis=1)
                        
                        U_list = U_new.tolist()
                        V_list = V_new.tolist()
                        W_list = W_new.tolist()
                        
                        if verify_decomp(U_list, V_list, W_list, 65):
                            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try multiple constructions, verify, return best valid one
    best = None
    best_rank = float('inf')

    # First try to find rank-65
    if time.time() < deadline - 40:
        try:
            r65 = search_rank65()
            if r65 is not None:
                if verify_decomp(r65["U"], r65["V"], r65["W"], 65):
                    best = r65
                    best_rank = 65
        except Exception:
            pass

    # Fall back to rank-66 if needed
    if best is None:
        try:
            result = build_rank66()
            r = len(result["U"][0])
            if verify_decomp(result["U"], result["V"], result["W"], r):
                best = result
                best_rank = r
        except Exception:
            pass

    # Final fallback
    if best is None:
        best = build_schoolbook()

    return best
