"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to find a lower rank decomposition via search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Build the rank-26 baseline
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 7 terms
        terms = [
            # (U_entries, V_entries, W_entries)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        # Standard 19 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-24 or rank-25 decomposition
    # Strategy: exploit the structure of border terms by finding linear dependencies
    
    # The 19 border terms have indices:
    border_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                border_indices.append((i, j, k))
    
    # The border tensor has a specific structure:
    # - Terms with i=2: (2,j,k) for j,k in {0,1,2} -> 9 terms
    # - Terms with j=2: (i,2,k) for i,k in {0,1} -> 4 terms  
    # - Terms with k=2: (i,j,2) for i,j in {0,1} -> 4 terms
    # Total: 17 unique border positions (some overlap in counting)
    
    # Key insight: We can try to find if the border sub-tensor has rank < 19
    # by checking if we can represent it with fewer rank-1 terms
    
    # Approach: Use the fact that many border terms share indices
    # Try to find a valid decomposition where some border terms are combined
    
    # First, let's try a structured approach:
    # Group border terms by their "type" and see if we can merge within groups
    
    # Use correct matrix multiplication verification
    def verify_mm(U, V, W, R):
        np.random.seed(123)
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            
            C_prime = np.zeros((n, p))
            for r in range(R):
                u_vec = U[:, r].reshape(n, m)
                v_vec = V[:, r].reshape(m, p)
                w_vec = W[:, r].reshape(p, n)
                
                sum_u = np.sum(u_vec * A)
                sum_v = np.sum(v_vec * B)
                M_r = sum_u * sum_v
                
                C_prime += M_r * w_vec.T
                
            if not np.allclose(C, C_prime, atol=1e-6):
                return False
        return True

    # Try to find Rank 23 using Nelder-Mead
    def try_find_rank(R):
        if time.time() > deadline - 20:
            return None
            
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        def objective(params):
            U = params[:size_u].reshape(n * m, R)
            V = params[size_u:size_u+size_v].reshape(m * p, R)
            W = params[size_u+size_v:].reshape(p * n, R)
            
            error = 0
            for seed in [1, 2, 3]:
                np.random.seed(seed)
                A = np.random.rand(n, m)
                B = np.random.rand(m, p)
                C = A @ B
                
                C_prime = np.zeros((n, p))
                for r in range(R):
                    u_vec = U[:, r].reshape(n, m)
                    v_vec = V[:, r].reshape(m, p)
                    w_vec = W[:, r].reshape(p, n)
                    
                    sum_u = np.sum(u_vec * A)
                    sum_v = np.sum(v_vec * B)
                    M_r = sum_u * sum_v
                    
                    C_prime += M_r * w_vec.T
                    
                error += np.sum((C - C_prime)**2)
                
            return error

        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt)
            x0 = np.random.randn(total_size)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.success:
                    U = res.x[:size_u].reshape(n * m, R)
                    V = res.x[size_u:size_u+size_v].reshape(m * p, R)
                    W = res.x[size_u+size_v:].reshape(p * n, R)
                    
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    
                    if verify_mm(U, V, W, R):
                        return U, V, W, R
                        
            except Exception:
                pass
                
        return None

    # Attempt Rank 23
    if time.time() < deadline - 30:
        result_23 = try_find_rank(23)
        if result_23:
            best_U, best_V, best_W, best_R = result_23

    # Final verification with correct MM check
    if not verify_mm(best_U, best_V, best_W, best_R):
        # Fallback to Rank 26 (which should be valid)
        if verify_mm(U26, V26, W26, R26):
            best_U, best_V, best_W, best_R = U26, V26, W26, R26
        else:
            # Fallback to schoolbook
            R = n * m * p
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[idx_u(i, j), col] = 1
                        V[idx_v(j, k), col] = 1
                        W[idx_w(k, i), col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
