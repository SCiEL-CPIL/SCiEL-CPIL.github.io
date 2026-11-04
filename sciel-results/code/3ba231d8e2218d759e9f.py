"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Reshape
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        rank = U.shape[1]
        
        # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # We can do this efficiently using einsum
        # T is n x m x p
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Target T
        T_target = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1.0
        
        return np.allclose(T_calc, T_target, atol=1e-6)

    # Helper to try to convert real solution to half-integers
    def to_half_int(U, V, W):
        # Scale to integers if possible
        # Check if entries are close to half-integers
        def clean(matrix):
            M = np.array(matrix, dtype=float)
            # Multiply by 2
            M2 = M * 2
            M_int = np.round(M2)
            if np.allclose(M2, M_int, atol=1e-5):
                return M_int / 2.0
            return None
        
        U_c = clean(U)
        V_c = clean(V)
        W_c = clean(W)
        
        if U_c is None or V_c is None or W_c is None:
            return None
        if not verify(U_c, V_c, W_c):
            return None
        return U_c, V_c, W_c

    # Search for decomposition of 2x3x3
    # If found, use it 3 times for 2x3x9
    def solve_233(max_rank=18, time_limit=5.0):
        n_s, m_s, p_s = 2, 3, 3
        U_shape = (n_s * m_s, max_rank)
        V_shape = (m_s * p_s, max_rank)
        W_shape = (p_s * n_s, max_rank)
        
        # Target tensor T
        T_target = np.zeros((n_s, m_s, p_s), dtype=float)
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    T_target[i, j, k] = 1.0
        
        # Try ranks from lower bound to max_rank
        # Lower bound max(6, 9, 6) = 9
        for r in range(9, max_rank + 1):
            if time.time() > start_time + time_limit:
                break
                
            # Random initialization
            for attempt in range(20):
                if time.time() > start_time + time_limit:
                    break
                    
                U = np.random.randn(n_s * m_s, r)
                V = np.random.randn(m_s * p_s, r)
                W = np.random.randn(p_s * n_s, r)
                
                # Scale to help convergence
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
                
                # Use alternating least squares or gradient descent
                # Here we use a simple iterative refinement
                # Fix W, solve for U, V? No, it's quadratic.
                # We can use scipy minimize on the flattened parameters
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def objective(params):
                    U_p = params[:n_s*m_s*r].reshape(n_s, m_s, r)
                    V_p = params[n_s*m_s*r:n_s*m_s*r + m_s*p_s*r].reshape(m_s, p_s, r)
                    W_p = params[n_s*m_s*r + m_s*p_s*r:].reshape(p_s, n_s, r)
                    
                    T_calc = np.einsum('ijc,jkc,kic->ijk', U_p, V_p, W_p)
                    return np.sum((T_calc - T_target)**2)
                
                # Gradient is expensive, use Nelder-Mead or similar?
                # Or just check if random init is close? Unlikely.
                # Let's try a few iterations of alternating updates
                # Fix V, W -> solve for U (linear)
                # Fix U, W -> solve for V (linear)
                # Fix U, V -> solve for W (linear)
                
                for iter in range(50):
                    # Update U
                    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                    # For fixed V, W, this is linear in U.
                    # We want to minimize sum_{i,j,k} (T[i,j,k] - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
                    # This is a least squares problem for each (i,j)
                    # But it's coupled.
                    # Let's build the linear system for U.
                    # Vectorize T: t_vec
                    # Matrix A such that A * u_vec = t_vec
                    # This is equivalent to finding U such that 
                    # sum_c U[i,j,c] (V[j,k,c] W[k,i,c]) = 1
                    
                    # Let's just use a generic solver if rank is small
                    pass # Skipping complex ALS implementation for brevity in thought process, 
                         # will implement a simple random search with refinement in code.
                
                # Instead of complex optimization, let's try to find integer/half-integer solutions directly
                # for small ranks using random search with rounding.
                
                # Generate random half-integers
                U_h = np.round(np.random.randn(n_s * m_s, r) * 2) / 2
                V_h = np.round(np.random.randn(m_s * p_s, r) * 2) / 2
                W_h = np.round(np.random.randn(p_s * n_s, r) * 2) / 2
                
                if verify(U_h, V_h, W_h):
                    return r, U_h, V_h, W_h
                    
        return None

    # Try to find 2x3x3 decomposition
    sol_233 = solve_233(max_rank=12, time_limit=5.0)
    
    if sol_233:
        r_233, U_233, V_233, W_233 = sol_233
        # Construct 2x3x9 decomposition
        # 3 blocks of 2x3x3
        total_rank = 3 * r_233
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for block in range(3):
            # U: same for all blocks? No, U is (2x3).
            # In 2x3x9, U is (2x3) x R.
            # The blocks are in the p dimension.
            # U[i, j, c] is independent of k.
            # So U for block b is just U_233.
            # But we need to place them in the right columns.
            
            # V: V[j, k, c]. k is 0..8.
            # Block 0: k=0..2
            # Block 1: k=3..5
            # Block 2: k=6..8
            
            # W: W[k, i, c].
            
            # U part
            U[:, block*r_233:(block+1)*r_233] = U_233
            
            # V part
            # V_233 is (3x3) x r_233. Reshape to (3, 3, r_233)
            V_233_r = V_233.reshape(3, 3, r_233)
            for j in range(3):
                for k_local in range(3):
                    k_global = block * 3 + k_local
                    V[j * p + k_global, block*r_233:(block+1)*r_233] = V_233_r[j, k_local, :]
            
            # W part
            # W_233 is (3x2) x r_233. Reshape to (3, 2, r_233)
            W_233_r = W_233.reshape(3, 2, r_233)
            for k_local in range(3):
                for i in range(2):
                    k_global = block * 3 + k_local
                    W[k_global * n + i, block*r_233:(block+1)*r_233] = W_233_r[k_local, i, :]
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
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
