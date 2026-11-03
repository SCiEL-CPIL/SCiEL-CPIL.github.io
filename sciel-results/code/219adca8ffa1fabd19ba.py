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
    
    def get_block_diagonal_2x2x8():
        """
        Returns U, V, W for 2x2x8 tensor with rank 7*4=28 (not 56).
        Wait, let me reconsider. For 2x2x8, we have 4 blocks of 2x2x2.
        Each 2x2x2 has rank 7, so total rank is 4*7=28.
        U: (4, 28), V: (16, 28), W: (16, 28)
        """
        rank = 28
        U = np.zeros((4, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Strassen coefficients for 2x2x2
        # U is 2x2, indexed by (i,j) where i,j in {0,1}
        # V is 2x2, indexed by (j,k) where j,k in {0,1}
        # W is 2x2, indexed by (k,i) where k,i in {0,1}
        
        # Standard Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B11 + B10)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U_coeffs = [
            {0: 1, 3: 1},   # M1: A00 + A11
            {2: 1, 3: 1},   # M2: A10 + A11
            {0: 1},          # M3: A00
            {3: 1},          # M4: A11
            {0: 1, 1: 1},   # M5: A00 + A01
            {2: 1, 0: -1},  # M6: A10 - A00
            {1: 1, 3: -1}   # M7: A01 - A11
        ]
        V_coeffs = [
            {0: 1, 3: 1},   # M1: B00 + B11
            {0: 1},          # M2: B00
            {1: 1, 3: -1},  # M3: B01 - B11
            {2: 1, 0: -1},  # M4: B10 - B00
            {3: 1},          # M5: B11
            {0: 1, 2: 1},   # M6: B00 + B10
            {2: 1, 3: 1}    # M7: B11 + B10
        ]
        W_coeffs = [
            {0: 1, 3: -1},  # M1: C00 + C11
            {2: 1, 3: 0},   # M2: C10
            {1: 1, 3: 0},   # M3: C01
            {0: 1, 2: 1},   # M4: C00 + C10
            {0: -1, 1: 1},  # M5: -C00 + C01
            {3: 1},          # M6: C11
            {0: 1}           # M7: C00
        ]
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * 7
            for l in range(7):
                col = col_start + l
                # U: 2x2 matrix, rows are i*2+j
                for li, coeff in U_coeffs[l].items():
                    i_local = li // 2
                    j_local = li % 2
                    U[i_local*2 + j_local, col] += coeff
                # V: 2x2 matrix for this k-block, rows are j*2+k_local
                # But full V has rows j*8+k_global
                for lj, coeff in V_coeffs[l].items():
                    j_local = lj // 2
                    k_local = lj % 2
                    k_global = k0 + k_local
                    V[j_local*8 + k_global, col] += coeff
                # W: 2x2 matrix, rows are k*2+i
                # But full W has rows k*2+i for 2x2x8
                for lc, coeff in W_coeffs[l].items():
                    i_local = lc // 2
                    k_local = lc % 2
                    W[k_local*2 + i_local, col] += coeff
        
        return U, V, W
    
    def get_block_diagonal_1x1x8():
        """
        Returns U, V, W for 1x1x8 tensor with rank 8.
        U: (1, 8), V: (8, 8), W: (8, 8)
        """
        rank = 8
        U = np.zeros((1, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        for k in range(8):
            U[0, k] = 1.0
            V[k, k] = 1.0
            W[k, k] = 1.0
        
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
            for scale in [1, 2, 4, 8, 16]:
                U_rounded = np.round(U_opt * scale) / scale
                V_rounded = np.round(V_opt * scale) / scale
                W_rounded = np.round(W_opt * scale) / scale
                
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                
                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                    return U_rounded, V_rounded, W_rounded
            
            # Also try snapping to half-integers
            for scale in [2, 4, 8]:
                U_rounded = np.round(U_opt * scale) / scale
                V_rounded = np.round(V_opt * scale) / scale
                W_rounded = np.round(W_opt * scale) / scale
                
                # Ensure values are multiples of 0.5
                U_half = np.round(U_rounded * 2) / 2
                V_half = np.round(V_rounded * 2) / 2
                W_half = np.round(W_rounded * 2) / 2
                
                T_check = compute_tensor(U_half, V_half, W_half)
                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                    return U_half, V_half, W_half
            
            return None
        
        def polish(x0, max_iter=500):
            """High-precision polish using LM method with tight tolerance."""
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                return result.x, np.max(np.abs(result.fun))
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
        
        # Phase 2: Random perturbations around the 17-rank solutions
        rng = np.random.default_rng(42)
        for attempt in range(10):
            if time_left() < 5:
                break
            
            drop_idx = rng.integers(0, 17)
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].copy()
            V_init = V17[:, keep_cols].copy()
            W_init = W17[:, keep_cols].copy()
            
            # Add small noise
            noise_scale = 0.02 + 0.01 * attempt
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
        
        # Phase 3: If we have a good solution, try to polish it further
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
    if time_left() > 10:
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        
    # Assemble full 3x3x8 decomposition using block-diagonal embedding
    # Partition 3x3 input into 2x2 sub-block (rows 0-1, cols 0-1) and 1x1 sub-block (row 2, col 2)
    # 2x2x8 part: rank 28
    # 1x1x8 part: rank 8
    # Total rank: 36
    
    rank_2x2x8 = 28
    rank_1x1x8 = 8
    rank_total = rank_2x2x8 + rank_1x1x8
    
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    # Get 2x2x8 decomposition
    U_2x2x8, V_2x2x8, W_2x2x8 = get_block_diagonal_2x2x8()
    
    # Get 1x1x8 decomposition
    U_1x1x8, V_1x1x8, W_1x1x8 = get_block_diagonal_1x1x8()
    
    # Embed 2x2x8 part
    # U_2x2x8 has shape (4, 28). Rows are i*2+j for i,j in {0,1}
    # Full U has shape (9, 36). Rows are i*3+j
    for i in range(2):
        for j in range(2):
            U[i*3 + j, :rank_2x2x8] = U_2x2x8[i*2 + j, :]
    
    # V_2x2x8 has shape (16, 28). Rows are j*8+k for j in {0,1}, k in {0,...,7}
    # Full V has shape (24, 36). Rows are j*8+k
    V[:16, :rank_2x2x8] = V_2x2x8
    
    # W_2x2x8 has shape (16, 28). Rows are k*2+i for k in {0,...,7}, i in {0,1}
    # Full W has shape (24, 36). Rows are k*3+i
    for k in range(8):
        for i in range(2):
            W[k*3 + i, :rank_2x2x8] = W_2x2x8[k*2 + i, :]
    
    # Embed 1x1x8 part
    # U_1x1x8 has shape (1, 8). Row 0 corresponds to (i=2, j=2)
    U[2*3 + 2, rank_2x2x8:] = U_1x1x8
    
    # V_1x1x8 has shape (8, 8). Rows are k for k in {0,...,7}
    # Full V rows for j=2 are 2*8+k = 16+k
    V[16:24, rank_2x2x8:] = V_1x1x8
    
    # W_1x1x8 has shape (8, 8). Rows are k for k in {0,...,7}
    # Full W rows for i=2 are k*3+2
    for k in range(8):
        W[k*3 + 2, rank_2x2x8:] = W_1x1x8[k, :]
    
    # Also need to handle the cross terms:
    # i in {0,1}, j=2: A[0:2, 2] * B[2, :] -> C[0:2, :]
    # i=2, j in {0,1}: A[2, 0:2] * B[0:2, :] -> C[2, :]
    # These are not captured by the block-diagonal decomposition!
    # We need to add these terms.
    
    # For i in {0,1}, j=2: rank 2*8 = 16
    # For i=2, j in {0,1}: rank 2*8 = 16
    # This gives total rank 28 + 8 + 16 + 16 = 68, which is the same as before.
    
    # Let me reconsider the block-diagonal approach.
    # The issue is that the 3x3 matrix is not block-diagonal in the way I assumed.
    # A[0:2, 2] and A[2, 0:2] are not zero.
    
    # So the block-diagonal embedding doesn't work directly.
    # Let me fall back to the original approach.
    
    if validate(U, V, W):
        return result
    else:
        # The block-diagonal approach doesn't work because of cross terms.
        # Fall back to the original 4-block approach.
        rank_total_old = 4 * rank_block
        U_old = np.zeros((9, rank_total_old))
        V_old = np.zeros((24, rank_total_old))
        W_old = np.zeros((24, rank_total_old))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U_old[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V_old[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W_old[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        result_old = {"U": U_old.tolist(), "V": V_old.tolist(), "W": W_old.tolist()}
        
        if validate(U_old, V_old, W_old):
            return result_old
        else:
            return schoolbook()
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        # Fallback to previous method
        rank_total_old = 4 * rank_block
        U_old = np.zeros((9, rank_total_old))
        V_old = np.zeros((24, rank_total_old))
        W_old = np.zeros((24, rank_total_old))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U_old[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V_old[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W_old[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        result_old = {"U": U_old.tolist(), "V": V_old.tolist(), "W": W_old.tolist()}
        
        if validate(U_old, V_old, W_old):
            return result_old
        else:
            return schoolbook()
