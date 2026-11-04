import time
import numpy as np
from scipy.optimize import minimize

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
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note: V is (m, p, rank) so V[j,k,c] is correct.
        # W is (p, n, rank) so W[k,i,c] is correct.
        T_computed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_expected = np.ones((n, m, p))
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms) for 2x2x2 block (i in {0,1}, j in {0,1})
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
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def build_full_from_block(U_block, V_block, W_block, rank_block):
        """Assemble full 3x3x8 decomposition from 3x3x2 block."""
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        return U, V, W

    def try_optimize_block(rank_block, max_time=20):
        """Try to find a rank_block decomposition for 3x3x2 using optimization."""
        if time_left() < max_time + 5:
            return None
        
        # Parameterize: each factor matrix has rank_block columns
        # Total parameters: 9*rank_block + 6*rank_block + 6*rank_block = 21*rank_block
        n_params = 21 * rank_block
        
        def unpack(params):
            U = params[0:9*rank_block].reshape(3, 3, rank_block)
            V = params[9*rank_block:15*rank_block].reshape(3, 2, rank_block)
            W = params[15*rank_block:21*rank_block].reshape(2, 3, rank_block)
            return U, V, W
        
        def objective(params):
            U, V, W = unpack(params)
            # Vectorized computation
            T_computed = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            return np.sum((T_computed - 1.0)**2)
        
        # Initialize with the 17-rank solution (zero-padded or truncated)
        U17, V17, W17 = get_17_rank_block()
        
        # Create initial guess
        params_init = np.zeros(n_params)
        
        # If rank_block < 17, use a subset of the 17-rank terms
        if rank_block <= 17:
            # Use the first rank_block terms from the 17-rank solution
            params_init[0:9*rank_block] = U17[:, :rank_block].flatten()
            params_init[9*rank_block:15*rank_block] = V17[:, :rank_block].flatten()
            params_init[15*rank_block:21*rank_block] = W17[:, :rank_block].flatten()
        
        # Try multiple random initializations
        best_params = None
        best_obj = float('inf')
        
        n_restarts = 10 # Increased restarts due to speedup
        for restart in range(n_restarts):
            if time_left() < max_time:
                break
            
            if restart == 0:
                x0 = params_init.copy()
            else:
                # Random dense initialization
                x0 = np.random.randn(n_params) * 0.5
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is None or best_obj > 1e-8:
            return None
        
        # Check if solution is valid
        U, V, W = unpack(best_params)
        
        # Verify
        T_check = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        
        if np.max(np.abs(T_check - 1.0)) > 1e-8:
            return None
        
        # Convert back to flat matrices for the rest of the code
        U_flat = U.reshape(9, rank_block)
        V_flat = V.reshape(6, rank_block)
        W_flat = W.reshape(6, rank_block)
        
        return U_flat, V_flat, W_flat

    # First, establish the baseline rank-68 solution
    U_block, V_block, W_block = get_17_rank_block()
    rank_block = 17
    U, V, W = build_full_from_block(U_block, V_block, W_block, rank_block)
    baseline_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not validate(U, V, W):
        return schoolbook()
    
    # Try to find a better block decomposition
    # Try rank 16 first
    if time_left() > 25:
        result_16 = try_optimize_block(16, max_time=15)
        if result_16 is not None:
            U_block, V_block, W_block = result_16
            rank_block = 16
            U, V, W = build_full_from_block(U_block, V_block, W_block, rank_block)
            if validate(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the baseline rank-68 solution
    return baseline_result
