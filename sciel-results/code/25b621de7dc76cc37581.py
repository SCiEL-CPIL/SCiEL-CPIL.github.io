import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Vectorized: for each (i,j), T[i,j,:] += u[i,j] * (v[j,:] * w[:,i])
            for i in range(n):
                for j in range(m):
                    T_computed[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        
        return np.max(np.abs(T_computed - 1.0)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        # Re-construct the 17-rank logic for one block
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms)
        U_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: 1},       
            {0: 1},             
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 0: -1},      
            {1: 1, 3: -1}       
        ]
        V_coeffs = [
            {0: 1, 3: 1},       
            {0: 1},             
            {1: 1, 3: -1},      
            {2: 1, 0: -1},      
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 3: 1}        
        ]
        W_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: -1},      
            {1: 1, 3: 1},       
            {0: 1, 2: 1},       
            {0: -1, 1: 1},      
            {3: 1},             
            {0: 1}              
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms)
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2 using multiple strategies."""
        U17, V17, W17 = get_17_rank_block()
        
        def compute_tensor(U, V, W):
            """Compute the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]."""
            T = np.zeros((3, 3, 2))
            for c in range(U.shape[1]):
                u = U[:, c].reshape(3, 3)
                v = V[:, c].reshape(3, 2)
                w = W[:, c].reshape(2, 3)
                for i in range(3):
                    for j in range(3):
                        T[i,j,:] += u[i,j] * (v[j,:] * w[:,i])
            return T
        
        def residuals(x):
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            T = compute_tensor(U, V, W)
            return (T - 1.0).flatten()
        
        def objective_soft(x, alpha=100.0):
            """Smooth surrogate: sum of soft-max of |residuals|."""
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            T = compute_tensor(U, V, W)
            r = np.abs(T - 1.0).flatten()
            # Log-sum-exp for numerical stability
            m = np.max(r)
            if m < 1e-15:
                return 0.0
            return m + np.log(np.sum(np.exp((r - m) * alpha)) / alpha) / alpha
        
        def check_rounded_solution(U_opt, V_opt, W_opt):
            """Try rounding and validate with multiple scales."""
            # Try snapping to half-integers first (most common for Strassen-like)
            U_half = np.round(U_opt * 2) / 2
            V_half = np.round(V_opt * 2) / 2
            W_half = np.round(W_opt * 2) / 2
            
            T_check = compute_tensor(U_half, V_half, W_half)
            if np.max(np.abs(T_check - 1.0)) < 1e-10:
                return U_half, V_half, W_half
            
            # Try integers
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            T_check = compute_tensor(U_int, V_int, W_int)
            if np.max(np.abs(T_check - 1.0)) < 1e-10:
                return U_int, V_int, W_int
            
            # Try quarters
            for scale in [4, 8, 16]:
                U_rounded = np.round(U_opt * scale) / scale
                V_rounded = np.round(V_opt * scale) / scale
                W_rounded = np.round(W_opt * scale) / scale
                
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                    return U_rounded, V_rounded, W_rounded
            
            # Try with slight tolerance adjustment
            for scale in [2, 4, 8]:
                U_rounded = np.round(U_opt * scale) / scale
                V_rounded = np.round(V_opt * scale) / scale
                W_rounded = np.round(W_opt * scale) / scale
                
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - 1.0)) < 1e-8:
                    # Try to fix by small adjustments
                    for delta_scale in [0.1, 0.25, 0.5]:
                        for i in range(U_rounded.shape[0]):
                            for j in range(U_rounded.shape[1]):
                                for sign in [-1, 1]:
                                    U_try = U_rounded.copy()
                                    U_try[i,j] += sign * delta_scale
                                    T_try = compute_tensor(U_try, V_rounded, W_rounded)
                                    if np.max(np.abs(T_try - 1.0)) < 1e-10:
                                        return U_try, V_rounded, W_rounded
                        for i in range(V_rounded.shape[0]):
                            for j in range(V_rounded.shape[1]):
                                for sign in [-1, 1]:
                                    V_try = V_rounded.copy()
                                    V_try[i,j] += sign * delta_scale
                                    T_try = compute_tensor(U_rounded, V_try, W_rounded)
                                    if np.max(np.abs(T_try - 1.0)) < 1e-10:
                                        return U_rounded, V_try, W_rounded
                        for i in range(W_rounded.shape[0]):
                            for j in range(W_rounded.shape[1]):
                                for sign in [-1, 1]:
                                    W_try = W_rounded.copy()
                                    W_try[i,j] += sign * delta_scale
                                    T_try = compute_tensor(U_rounded, V_rounded, W_try)
                                    if np.max(np.abs(T_try - 1.0)) < 1e-10:
                                        return U_rounded, V_rounded, W_try
            
            return None
        
        def polish(x0, max_iter=500):
            """High-precision polish using LM method with tight tolerance."""
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                return result.x, np.max(np.abs(result.fun))
            except Exception:
                return x0, np.max(np.abs(residuals(x0)))
        
        def polish_smooth(x0, max_iter=500, alpha=100.0):
            """Polish using smooth surrogate objective."""
            try:
                result = least_squares(
                    lambda x: (objective_soft(x, alpha) - 0.0) * np.ones(1),
                    x0, method='lm', max_nfev=max_iter, ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
                # Verify with actual residuals
                U = result.x[:144].reshape(9, 16)
                V = result.x[144:240].reshape(6, 16)
                W = result.x[240:336].reshape(6, 16)
                T = compute_tensor(U, V, W)
                return result.x, np.max(np.abs(T - 1.0))
            except Exception:
                return x0, np.max(np.abs(residuals(x0)))
        
        best_result = None
        best_residual = np.inf
        
        # Phase 1: Try dropping each of the 17 columns with LM
        for drop_idx in range(17):
            if time_left() < 5:
                break
            
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].flatten()
            V_init = V17[:, keep_cols].flatten()
            W_init = W17[:, keep_cols].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                x_opt, residual = polish(x0, max_iter=300)
                
                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt
                    
                    if residual < 1e-8:
                        U_opt = x_opt[:144].reshape(9, 16)
                        V_opt = x_opt[144:240].reshape(6, 16)
                        W_opt = x_opt[240:336].reshape(6, 16)
                        rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded
            except Exception:
                pass
        
        # Phase 2: Try smooth surrogate optimization from each drop
        for drop_idx in range(17):
            if time_left() < 5:
                break
            
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].flatten()
            V_init = V17[:, keep_cols].flatten()
            W_init = W17[:, keep_cols].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                x_opt, residual = polish_smooth(x0, max_iter=200, alpha=50.0)
                
                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt
                    
                    if residual < 1e-8:
                        U_opt = x_opt[:144].reshape(9, 16)
                        V_opt = x_opt[144:240].reshape(6, 16)
                        W_opt = x_opt[240:336].reshape(6, 16)
                        rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded
            except Exception:
                pass
        
        # Phase 3: Random perturbations around the 17-rank solutions
        rng = np.random.default_rng(42)
        for attempt in range(15):
            if time_left() < 5:
                break
            
            drop_idx = rng.integers(0, 17)
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].copy()
            V_init = V17[:, keep_cols].copy()
            W_init = W17[:, keep_cols].copy()
            
            # Add small noise
            noise_scale = 0.02 + 0.005 * attempt
            U_init += rng.normal(0, noise_scale, U_init.shape)
            V_init += rng.normal(0, noise_scale, V_init.shape)
            W_init += rng.normal(0, noise_scale, W_init.shape)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                x_opt, residual = polish(x0, max_iter=200)
                
                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt
                    
                    if residual < 1e-8:
                        U_opt = x_opt[:144].reshape(9, 16)
                        V_opt = x_opt[144:240].reshape(6, 16)
                        W_opt = x_opt[240:336].reshape(6, 16)
                        rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded
            except Exception:
                pass
        
        # Phase 4: If we have a good solution, try to polish it further
        if best_result is not None and best_residual < 1e-4:
            if time_left() > 10:
                x_polished, residual_polished = polish(best_result, max_iter=1000)
                if residual_polished < best_residual:
                    best_result = x_polished
                    best_residual = residual_polished
                
                U_opt = best_result[:144].reshape(9, 16)
                V_opt = best_result[144:240].reshape(6, 16)
                W_opt = best_result[240:336].reshape(6, 16)
                rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                if rounded is not None:
                    return rounded
        
        return None

    # Try to find rank 16 block (with time limit)
    block_16 = None
    if time_left() > 15:
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        # Try different Strassen block placements
        U_block, V_block, W_block, rank_block = None, None, None, 17
        
        # Try placing the 2x2x2 Strassen block in different positions
        # The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        # For 3x3x2, we can place it at different (i0, j0, k0) positions
        placements = [
            (0, 0, 0),  # top-left
            (0, 0, 1),  # top-right
            (0, 1, 0),  # middle-left
            (0, 1, 1),  # middle-right
            (1, 0, 0),  # bottom-left
            (1, 0, 1),  # bottom-right
            (1, 1, 0),  # bottom-middle
            (1, 1, 1),  # bottom-right
        ]
        
        best_placement = None
        best_rank = 17
        
        for (i0, j0, k0) in placements:
            if time_left() < 5:
                break
            
            # Check if this placement is valid (within bounds)
            if i0 + 1 >= 3 or j0 + 1 >= 3 or k0 + 1 >= 2:
                continue
            
            # Build a block decomposition with the Strassen part at this position
            rank = 17
            U = np.zeros((9, rank))
            V = np.zeros((6, rank))
            W = np.zeros((6, rank))
            
            col = 0
            # Strassen part (7 terms) for the 2x2x2 block
            U_coeffs = [
                {0: 1, 3: 1},       
                {2: 1, 3: 1},       
                {0: 1},             
                {3: 1},             
                {0: 1, 1: 1},       
                {2: 1, 0: -1},      
                {1: 1, 3: -1}       
            ]
            V_coeffs = [
                {0: 1, 3: 1},       
                {0: 1},             
                {1: 1, 3: -1},      
                {2: 1, 0: -1},      
                {3: 1},             
                {0: 1, 1: 1},       
                {2: 1, 3: 1}        
            ]
            W_coeffs = [
                {0: 1, 3: 1},       
                {2: 1, 3: -1},      
                {1: 1, 3: 1},       
                {0: 1, 2: 1},       
                {0: -1, 1: 1},      
                {3: 1},             
                {0: 1}              
            ]
            
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_local = li // 2
                    j_local = li % 2
                    i_global = i0 + i_local
                    j_global = j0 + j_local
                    U[i_global*3 + j_global, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_local = lj // 2
                    k_local = lj % 2
                    j_global = j0 + j_local
                    k_global = k0 + k_local
                    V[j_global*2 + k_global, col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_local = lc // 2
                    k_local = lc % 2
                    i_global = i0 + i_local
                    k_global = k0 + k_local
                    W[k_global*3 + i_global, col] += coeff
                col += 1
            
            # Schoolbook part for remaining terms
            # Terms where i is in {0,1,2}, j is in {0,1,2}, k is in {0,1}
            # but NOT covered by the Strassen block
            covered = set()
            for i_local in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        covered.add((i0 + i_local, j0 + j_local, k0 + k_local))
            
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        if (i, j, k) not in covered:
                            U[i*3 + j, col] = 1.0
                            V[j*2 + k, col] = 1.0
                            W[k*3 + i, col] = 1.0
                            col += 1
            
            # Check if this is a valid decomposition
            T_check = np.zeros((3, 3, 2))
            for c in range(rank):
                u = U[:, c].reshape(3, 3)
                v = V[:, c].reshape(3, 2)
                w = W[:, c].reshape(2, 3)
                for ii in range(3):
                    for jj in range(3):
                        T_check[ii,jj,:] += u[ii,jj] * (v[jj,:] * w[:,ii])
            
            if np.max(np.abs(T_check - 1.0)) < 1e-10:
                # This is a valid rank-17 decomposition with this placement
                # Check if it's better than current best
                if rank < best_rank:
                    best_rank = rank
                    best_placement = (U.copy(), V.copy(), W.copy())
        
        if best_placement is not None:
            U_block, V_block, W_block = best_placement
            rank_block = best_rank
        else:
            U_block, V_block, W_block = get_17_rank_block()
            rank_block = 17
        
    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        # Try to fix by rounding
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        result_rounded = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        if validate(U_rounded, V_rounded, W_rounded):
            return result_rounded
        return schoolbook()
