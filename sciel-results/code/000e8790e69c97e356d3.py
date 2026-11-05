import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    # Pre-compute matricizations for ALS
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    def check_solution(U, V, W, tol=1e-9):
        # U, V, W are assumed to be half-integers
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_calc, T, atol=tol)

    def round_to_half_int(X):
        return np.round(X * 2) / 2

    def local_search(U, V, W, max_iters=2000):
        """
        Perform local search on the half-integer lattice to find an exact decomposition.
        Starts from U, V, W which are close to a solution.
        """
        U = round_to_half_int(U).copy()
        V = round_to_half_int(V).copy()
        W = round_to_half_int(W).copy()
        
        def get_error():
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc - T)
            
        current_error = get_error()
        if current_error < 1e-9:
            return U, V, W, True

        # Pre-compute gradients or just use finite differences?
        # Finite differences on 1000+ variables is slow.
        # Instead, we can use the structure.
        # Error E = T - T_calc.
        # We want to change one entry of U, V, or W to reduce ||E||^2.
        # ||E||^2 = ||T - sum U V W||^2.
        
        # This local search is complex. Let's try a simpler heuristic:
        # Randomly pick a variable, try neighbors, if error decreases, keep it.
        
        R = U.shape[1]
        for _ in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Pick random matrix and entry
            mat_idx = np.random.choice(3)
            if mat_idx == 0:
                mat = 'U'
                rows, cols = U.shape
            elif mat_idx == 1:
                mat = 'V'
                rows, cols = V.shape
            else:
                mat = 'W'
                rows, cols = W.shape
            
            r_idx = np.random.randint(rows)
            c_idx = np.random.randint(cols)
            
            # Get current value
            if mat == 'U':
                val = U[r_idx, c_idx]
            elif mat == 'V':
                val = V[r_idx, c_idx]
            else:
                val = W[r_idx, c_idx]
                
            # Try neighbors: val - 0.5, val + 0.5, val (if not half-int)
            # Since we start from rounded, val is half-int.
            # Try val - 0.5 and val + 0.5
            
            best_neighbor_val = val
            best_neighbor_error = current_error
            
            for delta in [-0.5, 0.5]:
                new_val = val + delta
                # Apply change
                if mat == 'U':
                    U[r_idx, c_idx] = new_val
                elif mat == 'V':
                    V[r_idx, c_idx] = new_val
                else:
                    W[r_idx, c_idx] = new_val
                    
                new_error = get_error()
                
                # Revert
                if mat == 'U':
                    U[r_idx, c_idx] = val
                elif mat == 'V':
                    V[r_idx, c_idx] = val
                else:
                    W[r_idx, c_idx] = val
                
                if new_error < best_neighbor_error:
                    best_neighbor_error = new_error
                    best_neighbor_val = new_val
            
            # Apply best neighbor if it improved
            if best_neighbor_val != val:
                if mat == 'U':
                    U[r_idx, c_idx] = best_neighbor_val
                elif mat == 'V':
                    V[r_idx, c_idx] = best_neighbor_val
                else:
                    W[r_idx, c_idx] = best_neighbor_val
                current_error = best_neighbor_error
                
                if current_error < 1e-9:
                    return U, V, W, True
                    
        return U, V, W, False

    # Search ranks
    # Start from a reasonable lower bound.
    # For 3x3x4, let's try 20 to 35.
    for R in range(20, 36):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(15):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(1000 + R * 100 + attempt)
            
            # Initialization: SVD based
            # Project T onto rank R subspace
            # This is expensive if done naively. 
            # Simple random init is faster but might converge to bad local min.
            # Let's try random init first, it's faster.
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Normalize to avoid scaling issues
            U = U / np.linalg.norm(U) * np.sqrt(R)
            V = V / np.linalg.norm(V) * np.sqrt(R)
            W = W / np.linalg.norm(W) * np.sqrt(R)
            
            converged = False
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
                    U = U_T.T
                except:
                    break
                
                # Update V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
                    V = V_T.T
                except:
                    break
                
                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                
                if error < 1e-6:
                    # Try rounding
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    if check_solution(U_r, V_r, W_r):
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    else:
                        # Try local search
                        U_ls, V_ls, W_ls, success = local_search(U_r, V_r, W_r, max_iters=500)
                        if success:
                            best_rank = R
                            best_U = U_ls
                            best_V = V_ls
                            best_W = W_ls
                            converged = True
                            break
            if converged:
                break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
