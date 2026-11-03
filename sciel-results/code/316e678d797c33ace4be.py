# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    schoolbook_rank = n * m * p  # 54
    
    # Precompute tensor indices
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c))
    
    nz_a = np.array([x[0] for x in non_zeros])
    nz_b = np.array([x[1] for x in non_zeros])
    nz_c = np.array([x[2] for x in non_zeros])
    
    def compute_error(U, V, W, nz_a, nz_b, nz_c):
        """Compute sum of squared errors for the decomposition."""
        U_A = U[nz_a, :]
        V_B = V[nz_b, :]
        W_C = W[nz_c, :]
        Prods = U_A * V_B * W_C
        Sums = Prods.sum(axis=1)
        return np.sum((Sums - 1.0) ** 2)
    
    def try_decomposition(nn, mm, pp, r, num_restarts=20, max_iter=300, tol=1e-10, time_per_restart=5.0):
        """ALS-based decomposition search."""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        start_time = time.time()
        
        # Non-zero indices for this sub-problem
        non_zeros_local = []
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    a = i * mm + j
                    b = j * pp + k
                    c = k * nn + i
                    non_zeros_local.append((a, b, c))
        
        nz_a_local = np.array([x[0] for x in non_zeros_local])
        nz_b_local = np.array([x[1] for x in non_zeros_local])
        nz_c_local = np.array([x[2] for x in non_zeros_local])
        
        for restart in range(num_restarts):
            if time.time() - start_time > num_restarts * time_per_restart:
                break
            
            # Multiple initialization strategies
            init_strategy = restart % 4
            if init_strategy == 0:
                np.random.seed(restart * 7919 + r + nn * 100 + mm * 10 + pp)
                U = np.random.randn(nn * mm, r) * 0.5
                V = np.random.randn(mm * pp, r) * 0.5
                W = np.random.randn(pp * nn, r) * 0.5
            elif init_strategy == 1:
                # Sparse-ish initialization
                np.random.seed(restart * 104729 + r + nn * 100 + mm * 10 + pp)
                U = np.zeros((nn * mm, r))
                V = np.zeros((mm * pp, r))
                W = np.zeros((pp * nn, r))
                for d in range(r):
                    for _ in range(max(1, nn * mm // 2)):
                        U[np.random.randint(nn * mm), d] = np.random.choice([-1.0, 1.0])
                    for _ in range(max(1, mm * pp // 2)):
                        V[np.random.randint(mm * pp), d] = np.random.choice([-1.0, 1.0])
                    for _ in range(max(1, pp * nn // 2)):
                        W[np.random.randint(pp * nn), d] = np.random.choice([-1.0, 1.0])
            elif init_strategy == 2:
                # Block-structured initialization inspired by Strassen
                np.random.seed(restart * 15485863 + r + nn * 100 + mm * 10 + pp)
                U = np.zeros((nn * mm, r))
                V = np.zeros((mm * pp, r))
                W = np.zeros((pp * nn, r))
                # Initialize with random signs in blocks
                for d in range(r):
                    U[:, d] = np.random.choice([-1.0, 0.0, 1.0], size=nn * mm)
                    V[:, d] = np.random.choice([-1.0, 0.0, 1.0], size=mm * pp)
                    W[:, d] = np.random.choice([-1.0, 0.0, 1.0], size=pp * nn)
            else:
                # Gaussian with varying scale
                np.random.seed(restart * 27360737 + r + nn * 100 + mm * 10 + pp)
                scale = 0.3 + 0.4 * (restart % 4) / 4.0
                U = np.random.randn(nn * mm, r) * scale
                V = np.random.randn(mm * pp, r) * scale
                W = np.random.randn(pp * nn, r) * scale
            
            for it in range(max_iter):
                if time.time() - start_time > (restart + 1) * time_per_restart:
                    break
                
                # Update U: minimize ||U * (V ⊙ W) - target||
                # D_U[b,c,d] = V[b,d] * W[c,d]
                D_U = np.einsum('bd,cd->bcd', V, W)  # shape (mm*pp, pp*nn, r)
                # Reshape: for each a, we need U[a,:] dot D_U[b,c,:] = 1 if (a,b,c) is in non_zeros
                # Flatten D_U: shape (mm*pp * pp*nn, r)
                D_U_flat = D_U.reshape(-1, r)  # (mm*pp*pp*nn, r)
                
                # Build target: for each a, the equation is sum_d U[a,d] * D_U_flat[idx(b,c),d] = 1
                # where idx(b,c) = b * (pp*nn) + c
                Targets_U = np.zeros((nn * mm, mm * pp * pp * nn))
                for a, b, c in non_zeros_local:
                    idx = b * (pp * nn) + c
                    Targets_U[a, idx] = 1.0
                
                U_new = np.linalg.lstsq(D_U_flat, Targets_U.T, rcond=None)[0].T
                
                # Update V
                D_V = np.einsum('ad,cd->acd', U_new, W)  # shape (nn*mm, pp*nn, r)
                D_V_flat = D_V.reshape(-1, r)  # (nn*mm*pp*nn, r)
                
                Targets_V = np.zeros((mm * pp, nn * mm * pp * nn))
                for a, b, c in non_zeros_local:
                    idx = a * (pp * nn) + c
                    Targets_V[b, idx] = 1.0
                
                V_new = np.linalg.lstsq(D_V_flat, Targets_V.T, rcond=None)[0].T
                
                # Update W
                D_W = np.einsum('ad,bd->abd', U_new, V_new)  # shape (nn*mm, mm*pp, r)
                D_W_flat = D_W.reshape(-1, r)  # (nn*mm*mm*pp, r)
                
                Targets_W = np.zeros((pp * nn, nn * mm * mm * pp))
                for a, b, c in non_zeros_local:
                    idx = a * (mm * pp) + b
                    Targets_W[c, idx] = 1.0
                
                W_new = np.linalg.lstsq(D_W_flat, Targets_W.T, rcond=None)[0].T
                
                # Check error
                err = compute_error(U_new, V_new, W_new, nz_a_local, nz_b_local, nz_c_local)
                
                if err < best_err:
                    best_err = err
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()
                
                U, V, W = U_new, V_new, W_new
                
                if err < tol:
                    break
        
        return best_err, best_U, best_V, best_W
    
    def polish_solution(nn, mm, pp, U, V, W, r, max_iter=200, tol=1e-12, time_limit=5.0):
        """High-precision polish pass."""
        start_time = time.time()
        
        non_zeros_local = []
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    a = i * mm + j
                    b = j * pp + k
                    c = k * nn + i
                    non_zeros_local.append((a, b, c))
        
        nz_a_local = np.array([x[0] for x in non_zeros_local])
        nz_b_local = np.array([x[1] for x in non_zeros_local])
        nz_c_local = np.array([x[2] for x in non_zeros_local])
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            D_U = np.einsum('bd,cd->bcd', V, W)
            D_U_flat = D_U.reshape(-1, r)
            Targets_U = np.zeros((nn * mm, mm * pp * pp * nn))
            for a, b, c in non_zeros_local:
                idx = b * (pp * nn) + c
                Targets_U[a, idx] = 1.0
            U_new = np.linalg.lstsq(D_U_flat, Targets_U.T, rcond=None)[0].T
            
            # Update V
            D_V = np.einsum('ad,cd->acd', U_new, W)
            D_V_flat = D_V.reshape(-1, r)
            Targets_V = np.zeros((mm * pp, nn * mm * pp * nn))
            for a, b, c in non_zeros_local:
                idx = a * (pp * nn) + c
                Targets_V[b, idx] = 1.0
            V_new = np.linalg.lstsq(D_V_flat, Targets_V.T, rcond=None)[0].T
            
            # Update W
            D_W = np.einsum('ad,bd->abd', U_new, V_new)
            D_W_flat = D_W.reshape(-1, r)
            Targets_W = np.zeros((pp * nn, nn * mm * mm * pp))
            for a, b, c in non_zeros_local:
                idx = a * (mm * pp) + b
                Targets_W[c, idx] = 1.0
            W_new = np.linalg.lstsq(D_W_flat, Targets_W.T, rcond=None)[0].T
            
            diff_U = np.linalg.norm(U_new - U)
            diff_V = np.linalg.norm(V_new - V)
            diff_W = np.linalg.norm(W_new - W)
            
            U, V, W = U_new, V_new, W_new
            
            if max(diff_U, diff_V, diff_W) < tol:
                break
        
        err = compute_error(U, V, W, nz_a_local, nz_b_local, nz_c_local)
        return err, U, V, W
    
    def try_round_and_verify(nn, mm, pp, U, V, W, r, tolerance=1e-8):
        """Try rounding to nearest 0.5 and verify exactly."""
        U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(nn * mm)]
        V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(mm * pp)]
        W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(pp * nn)]
        
        non_zeros_local = []
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    a = i * mm + j
                    b = j * pp + k
                    c = k * nn + i
                    non_zeros_local.append((a, b, c))
        
        verified = True
        for a, b, c in non_zeros_local:
            val = sum(U_rounded[a][d] * V_rounded[b][d] * W_rounded[c][d] for d in range(r))
            if abs(val - 1.0) > tolerance:
                verified = False
                break
        
        if verified:
            return U_rounded, V_rounded, W_rounded
        return None
    
    def tile_solution(nn, mm, p_sub, U_sub, V_sub, W_sub, r_sub):
        """Tile a solution for (n, m, p_sub) to (n, m, p) where p is a multiple of p_sub."""
        if p % p_sub != 0:
            return None
        num_blocks = p // p_sub
        R = r_sub * num_blocks
        
        U_full = np.hstack([U_sub] * num_blocks)
        
        V_full = np.zeros((mm * p, R))
        W_full = np.zeros((p * nn, R))
        
        for block_idx in range(num_blocks):
            col_start = block_idx * r_sub
            col_end = col_start + r_sub
            
            for j in range(mm):
                for k_sub in range(p_sub):
                    b_sub = j * p_sub + k_sub
                    k_full = block_idx * p_sub + k_sub
                    b_full = j * p + k_full
                    V_full[b_full, col_start:col_end] = V_sub[b_sub, :]
            
            for k_sub in range(p_sub):
                for i in range(nn):
                    c_sub = k_sub * nn + i
                    k_full = block_idx * p_sub + k_sub
                    c_full = k_full * nn + i
                    W_full[c_full, col_start:col_end] = W_sub[c_sub, :]
        
        return U_full, V_full, W_full
    
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Strategy 1: Solve 2x3x3 sub-problem and tile
    sub_n, sub_m, sub_p = 2, 3, 3
    for sub_r in range(14, 19):  # Try ranks 14-18 for sub-problem
        if time.time() - start_time > time_limit - 30:
            break
        
        err, U_sub, V_sub, W_sub = try_decomposition(sub_n, sub_m, sub_p, sub_r, 
                                                       num_restarts=15, max_iter=200, 
                                                       tol=1e-10, time_per_restart=3.0)
        if err < 1e-8 and U_sub is not None:
            # Polish
            err2, U_sub, V_sub, W_sub = polish_solution(sub_n, sub_m, sub_p, U_sub, V_sub, W_sub, sub_r,
                                                         max_iter=100, tol=1e-12, time_limit=3.0)
            if err2 < 1e-8:
                # Tile to full problem
                U_full, V_full, W_full = tile_solution(n, m, sub_p, U_sub, V_sub, W_sub, sub_r)
                full_r = sub_r * (p // sub_p)
                
                # Verify full solution
                err_full = compute_error(U_full, V_full, W_full, nz_a, nz_b, nz_c)
                
                if err_full < 1e-6:
                    if full_r < best_rank:
                        best_rank = full_r
                        
                        # Try to round
                        result = try_round_and_verify(n, m, p, U_full, V_full, W_full, full_r)
                        if result is not None:
                            return {"U": result[0], "V": result[1], "W": result[2]}
                        
                        best_result = (U_full, V_full, W_full)
    
    # Strategy 2: Direct search for full 2x3x9 problem
    # Try ranks from best_rank-1 down
    max_direct_rank = min(best_rank - 1, 53)
    min_direct_rank = max(30, best_rank - 8)
    
    for r in range(max_direct_rank, min_direct_rank - 1, -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        err, U, V, W = try_decomposition(n, m, p, r, 
                                          num_restarts=10, max_iter=150, 
                                          tol=1e-10, time_per_restart=4.0)
        if err < 1e-8 and U is not None:
            err2, U, V, W = polish_solution(n, m, p, U, V, W, r,
                                             max_iter=100, tol=1e-12, time_limit=3.0)
            if err2 < 1e-8:
                if r < best_rank:
                    best_rank = r
                    
                    result = try_round_and_verify(n, m, p, U, V, W, r)
                    if result is not None:
                        return {"U": result[0], "V": result[1], "W": result[2]}
                    
                    best_result = (U, V, W)
    
    # If we found a float solution, try harder to round it
    if best_result is not None and best_rank < schoolbook_rank:
        U_best, V_best, W_best = best_result
        
        # Try multiple rounding strategies
        for round_tol in [1e-8, 1e-7, 1e-6]:
            result = try_round_and_verify(n, m, p, U_best, V_best, W_best, best_rank, tolerance=round_tol)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}
        
        # Try rounding with small perturbations
        for perturb in [0.01, 0.05, 0.1]:
            U_pert = U_best + np.random.randn(*U_best.shape) * perturb
            V_pert = V_best + np.random.randn(*V_best.shape) * perturb
            W_pert = W_best + np.random.randn(*W_best.shape) * perturb
            
            result = try_round_and_verify(n, m, p, U_pert, V_pert, W_pert, best_rank)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
