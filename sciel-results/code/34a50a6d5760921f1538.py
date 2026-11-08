"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, rank):
        """Verify using numpy for speed."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape to tensors
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        # Compute the tensor product sum
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        # Expected tensor is 1 everywhere
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        # Define groups
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_strassen_improved():
        """
        Try to find a better decomposition using Strassen-based approach.
        """
        if time.time() > deadline:
            return None
        
        result = build_strassen_decomposition()
        rank = result.pop("rank")
        
        # Verify
        if verify_numpy(result["U"], result["V"], result["W"], rank):
            return result
        
        return None

    def try_randomized_als(target_rank, max_iter=200, tol=1e-12):
        """
        Use alternating least squares to find a rank-target_rank decomposition.
        Returns the factors if successful, None otherwise.
        """
        if time.time() > deadline - 5:
            return None
        
        T = np.ones((n, m, p), dtype=np.float64)
        rng = np.random.default_rng(42)
        
        # Try multiple initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            r = target_rank
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            W = rng.standard_normal((p * n, r))
            
            # Normalize
            U /= np.sqrt(r)
            V /= np.sqrt(r) 
            W /= np.sqrt(r)
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(r)] for k in range(p)])
                        b = T[i, j, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            U[i*m+j, :] = 0.0
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(r)] for i in range(n)])
                        b = T[:, j, k]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            V[j*p+k, :] = 0.0
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                        b = T[i, :, k]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            W[k*n+i, :] = 0.0
                
                # Compute error
                T_recon = np.einsum('ijc,jkc,kic->ijk', 
                                     U.reshape(n, m, r), 
                                     V.reshape(m, p, r), 
                                     W.reshape(p, n, r))
                err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            # Check if we found an exact solution
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify_numpy(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try Strassen-based decomposition
    improved = try_strassen_improved()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    # Try randomized ALS with decreasing ranks
    if time.time() < deadline - 10:
        # Try ranks from 30 to 89
        for r in range(30, min(90, best_rank)):
            if time.time() > deadline - 10:
                break
            
            result = try_randomized_als(r, max_iter=100)
            if result is not None:
                if r < best_rank:
                    best_result = result
                    best_rank = r
                break

    return best_result
# EVOLVE-BLOCK-END
