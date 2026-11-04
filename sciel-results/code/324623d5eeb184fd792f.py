import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # ------------------------------------------------------------------
    # LADDER / PEELING CONSTRUCTION
    #
    # 2x3x7 = (j=2 row: 2x1x7, rank 14)
    #       + (k=6 col, j in {0,1}: 2x2x1, rank 4)
    #       + (core: j in {0,1}, k in {0..5}: 2x2x6)
    #
    # The 2x2x6 core is a direct sum of three 2x2x2 Strassen blocks
    # (k-pairs {0,1},{2,3},{4,5}), each of exact rank 7  ->  rank 21.
    #
    # Total rank = 14 + 4 + 21 = 39.  Verified exactly below.
    # ------------------------------------------------------------------
    def build_peeled_decomposition():
        rank = 14 + 4 + 21
        U = [[0] * rank for _ in range(n * m)]   # index i*m+j
        V = [[0] * rank for _ in range(m * p)]   # index j*p+k
        W = [[0] * rank for _ in range(p * n)]   # index k*n+i
        col = 0

        def add_term(u_map, v_map, w_map):
            """u_map/v_map/w_map: dict {index: coeff}. Adds one rank-1 term."""
            nonlocal col
            for idx, c in u_map.items():
                U[idx][col] += c
            for idx, c in v_map.items():
                V[idx][col] += c
            for idx, c in w_map.items():
                W[idx][col] += c
            col += 1

        def add_strassen_block(k0):
            """2x2x2 Strassen block on A[0..1,0..1] x B[0..1, k0..k0+1] -> C[k0..k0+1, 0..1]."""
            # A indices (i*m+j): A00=0, A01=1, A10=3, A11=4
            # B indices (j*p+k): b00=0*7+k0, b01=0*7+k0+1, b10=1*7+k0, b11=1*7+k0+1
            b00 = 0 * 7 + k0
            b01 = 0 * 7 + k0 + 1
            b10 = 1 * 7 + k0
            b11 = 1 * 7 + k0 + 1
            # C indices (k*n+i): c00=k0*2+0, c01=(k0+1)*2+0, c10=k0*2+1, c11=(k0+1)*2+1
            c00 = k0 * 2 + 0
            c01 = (k0 + 1) * 2 + 0
            c10 = k0 * 2 + 1
            c11 = (k0 + 1) * 2 + 1

            # M1: (A00+A11)(B00+B11) -> C00 + C11
            add_term({0: 1, 4: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
            # M2: (A10+A11)B00 -> C10 - C11
            add_term({3: 1, 4: 1}, {b00: 1}, {c10: 1, c11: -1})
            # M3: A00(B01-B11) -> C01 + C11
            add_term({0: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
            # M4: A11(B10-B00) -> C00 + C10
            add_term({4: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
            # M5: (A00+A01)B11 -> -C00 + C01
            add_term({0: 1, 1: 1}, {b11: 1}, {c00: -1, c01: 1})
            # M6: (A10-A00)(B00+B01) -> C11
            add_term({3: 1, 0: -1}, {b00: 1, b01: 1}, {c11: 1})
            # M7: (A01-A11)(B10+B11) -> C00
            add_term({1: 1, 4: -1}, {b10: 1, b11: 1}, {c00: 1})

        # --- Core: three Strassen blocks ---
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        # --- k=6 column, j in {0,1}: 2x2x1 schoolbook (rank 4) ---
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        add_term({0: 1}, {b06: 1}, {c06: 1})   # A00 B06 -> C06
        add_term({1: 1}, {b16: 1}, {c06: 1})   # A01 B16 -> C06
        add_term({3: 1}, {b06: 1}, {c16: 1})   # A10 B06 -> C16
        add_term({4: 1}, {b16: 1}, {c16: 1})   # A11 B16 -> C16

        # --- j=2 row: 2x1x7 schoolbook (rank 14) ---
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term({2: 1}, {b2k: 1}, {c0k: 1})   # A02 B2k -> C0k
            add_term({5: 1}, {b2k: 1}, {c1k: 1})   # A12 B2k -> C1k

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # ------------------------------------------------------------------
    # Build the verified base (rank 39) first.
    # ------------------------------------------------------------------
    base_solution = build_peeled_decomposition()
    base_rank = len(base_solution["U"][0])

    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
        # Fallback: pure schoolbook (rank 42) -- guaranteed correct.
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

    # ------------------------------------------------------------------
    # Full-tensor ALS refinement: try to find a decomposition with
    # rank < 39 by initializing from the base solution (with random
    # perturbations) and running alternating least squares on the
    # full 2x3x7 multiplication tensor.
    # ------------------------------------------------------------------
    def full_tensor_als(target_rank, max_iter=300, tol=1e-10, seed=0):
        """ALS on the full 2x3x7 multiplication tensor.
        Returns (U, V, W) numpy arrays of shapes (6, R), (21, R), (14, R)
        or None if no solution found within tolerance."""
        if time.time() > deadline - 10:
            return None
        
        # Build target tensor: T[i,j,k] = 1 for all (i,j,k)
        # Flatten indices:
        #   U index: a = i*m + j, range [0, 6)
        #   V index: b = j*p + k, range [0, 21)
        #   W index: c = k*n + i, range [0, 14)
        # T[a, b, c] = 1 iff the indices are consistent with some (i,j,k)
        
        # Precompute: for each a, which (i,j) does it correspond to?
        # a = i*3 + j
        # For each b, which (j,k)? b = j*7 + k
        # For each c, which (k,i)? c = k*2 + i
        
        # Build the target tensor as a 3D array
        T = np.zeros((6, 21, 14), dtype=np.float64)
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    a = i * 3 + j
                    b = j * 7 + k
                    c = k * 2 + i
                    T[a, b, c] = 1.0
        
        rng = np.random.default_rng(1000 + seed + target_rank)
        
        # Initialize from base solution structure with perturbation
        # The base has rank 39; we need to compress to target_rank
        # Strategy: initialize U, V, W randomly but with structure
        # that reflects the known decomposition
        
        best = None
        best_res = np.inf
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random factors
            U = rng.standard_normal((6, target_rank))
            V = rng.standard_normal((21, target_rank))
            W = rng.standard_normal((14, target_rank))
            
            # Normalize to avoid scale issues
            for r in range(target_rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-12:
                    U[:, r] /= nU
                if nV > 1e-12:
                    V[:, r] /= nV
                if nW > 1e-12:
                    W[:, r] /= nW
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, U[a,:] = T[a,:,:] contracted with V,W
                # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / (V[b,r]^2 * ...) 
                # Actually for ALS: fix V,W, solve for U
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b,c: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So for each (b,c), we have a linear system in U[:,r]
                # Better: for each a, U[a, :] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is: U[a, :] = (VW^T)^{-1} * T[a, :, :]^T where VW^T[b,c] = sum_r V[b,r] W[c,r]
                
                # Compute VW = V @ W.T  (shape 21 x 14)
                VW = V @ W.T  # (21, 14)
                
                # For each a, we need to solve: minimize ||T[a,:,:] - U[a,:] @ VW||_F^2
                # This is a least squares problem: U[a,:] @ VW = T[a,:,:]
                # i.e., for each a: U[a, r] = sum over (b,c) of T[a,b,c] * (pseudo-inverse of VW)[c,b]
                
                # Use pseudo-inverse for stability
                try:
                    VW_pinv = np.linalg.pinv(VW)  # (14, 21)
                except np.linalg.LinAlgError:
                    break
                
                # U[a, r] = sum_{b,c} T[a,b,c] * VW_pinv[c, b] ... 
                # Actually: U[a,:] @ VW = T[a,:,:]
                # U[a,:] = T[a,:,:] @ VW_pinv  (shape 1 x 21 @ 14 x 21 -> 1 x 21)
                # Wait, VW is (21, 14), VW_pinv is (14, 21)
                # T[a,:,:] is (21, 14)
                # U[a,:] = T[a,:,:] @ VW_pinv  -> (21,14) @ (14,21) = (21,21)? No.
                
                # Let me re-derive. T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Define M[b,c] = sum_r V[b,r] W[c,r] = (V @ W.T)[b,c]
                # Then T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # This is NOT simply U[a,:] @ M because M[b,c] = sum_r V[b,r]W[c,r] but
                # we need sum_r U[a,r] V[b,r] W[c,r] which is a different contraction.
                
                # The correct ALS update: for fixed V, W, the problem for U is:
                # minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # For each a: minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Let X[a,b,c] = T[a,b,c], and for each r, define y_r[b,c] = V[b,r] W[c,r]
                # Then we want: sum_r U[a,r] y_r[b,c] ≈ T[a,b,c] for all b,c
                # This is a linear least squares: U[a,:] @ Y = T[a,:,:]
                # where Y[r, (b,c)] = V[b,r] W[c,r], reshaped to (R, 21*14)
                
                # Build Y matrix: (R, 21*14)
                Y = np.zeros((target_rank, 21 * 14), dtype=np.float64)
                for r in range(target_rank):
                    Y[r, :] = (V[:, r] * W[:, r][:, None]).flatten()  # (21, 14) -> (294,)
                
                # For each a: solve U[a, :] @ Y = T[a, :, :].flatten()
                # i.e., Y.T @ U[a, :] = T[a, :, :].flatten()
                # U[a, :] = (Y.T @ Y)^{-1} Y.T @ T[a, :, :].flatten()
                
                YT = Y.T  # (294, R)
                YT_Y = YT @ Y  # (R, R)
                
                try:
                    # Add small regularization for stability
                    reg = 1e-10 * np.eye(target_rank)
                    YT_Y_reg = YT_Y + reg
                    U_new = np.zeros((6, target_rank), dtype=np.float64)
                    for a in range(6):
                        T_flat = T[a, :, :].flatten()  # (294,)
                        rhs = YT @ T_flat  # (R,)
                        U_new[a, :] = np.linalg.solve(YT_Y_reg, rhs)
                    U = U_new
                except np.linalg.LinAlgError:
                    break
                
                # Update V: fix U, W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each b: minimize sum_{a,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Z[r, (a,c)] = U[a,r] W[c,r]
                Z = np.zeros((target_rank, 6 * 14), dtype=np.float64)
                for r in range(target_rank):
                    Z[r, :] = (U[:, r] * W[:, r][:, None]).flatten()  # (6, 14) -> (84,)
                
                ZT = Z.T  # (84, R)
                ZT_Z = ZT @ Z  # (R, R)
                
                try:
                    reg = 1e-10 * np.eye(target_rank)
                    ZT_Z_reg = ZT_Z + reg
                    V_new = np.zeros((21, target_rank), dtype=np.float64)
                    for b in range(21):
                        T_flat = T[:, b, :].flatten()  # (84,)
                        rhs = ZT @ T_flat  # (R,)
                        V_new[b, :] = np.linalg.solve(ZT_Z_reg, rhs)
                    V = V_new
                except np.linalg.LinAlgError:
                    break
                
                # Update W: fix U, V
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each c: minimize sum_{a,b} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Q[r, (a,b)] = U[a,r] V[b,r]
                Q = np.zeros((target_rank, 6 * 21), dtype=np.float64)
                for r in range(target_rank):
                    Q[r, :] = (U[:, r] * V[:, r][:, None]).flatten()  # (6, 21) -> (126,)
                
                QT = Q.T  # (126, R)
                QT_Q = QT @ Q  # (R, R)
                
                try:
                    reg = 1e-10 * np.eye(target_rank)
                    QT_Q_reg = QT_Q + reg
                    W_new = np.zeros((14, target_rank), dtype=np.float64)
                    for c in range(14):
                        T_flat = T[:, :, c].flatten()  # (126,)
                        rhs = QT @ T_flat  # (R,)
                        W_new[c, :] = np.linalg.solve(QT_Q_reg, rhs)
                    W = W_new
                except np.linalg.LinAlgError:
                    break
                
                # Compute residual
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                res = float(np.linalg.norm(recon - T))
                
                if res < best_res:
                    best_res = res
                    best = (U.copy(), V.copy(), W.copy())
                
                if res < tol:
                    break
                # Check for convergence
                if it > 10 and abs(res - best_res) < 1e-12:
                    break
        
        if best is not None and best_res < 1e-8:
            return best
        return None

    # ------------------------------------------------------------------
    # Try to find a lower-rank decomposition using ALS.
    # Start from rank 38 down to 35 (theoretical lower bound considerations
    # suggest the rank is at least 30+ for 2x3x7, but let's try a few).
    # ------------------------------------------------------------------
    better_solution = None
    for target_rank in range(38, 34, -1):
        if time.time() > deadline - 20:
            break
        
        result = full_tensor_als(target_rank, max_iter=200, tol=1e-10, seed=target_rank)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Round to half-integers
            U_rounded = np.round(U_arr * 2) / 2.0
            V_rounded = np.round(V_arr * 2) / 2.0
            W_rounded = np.round(W_arr * 2) / 2.0
            
            # Verify the rounded solution
            recon = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            T_target = np.zeros((6, 21, 14), dtype=np.float64)
            for i in range(2):
                for j in range(3):
                    for k in range(7):
                        T_target[i * 3 + j, j * 7 + k, k * 2 + i] = 1.0
            
            if np.linalg.norm(recon - T_target) < 1e-9:
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                better_solution = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    if better_solution is not None:
        return better_solution
    return base_solution

# EVOLVE-BLOCK-END
