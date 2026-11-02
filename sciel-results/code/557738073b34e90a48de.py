import time
import numpy as np
from scipy.optimize import least_squares, differential_evolution

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j*p+k, W row: k*n+i
    
    # Build the multiplication tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    # First, try to find a rank 13 solution
    best_solution = None
    
    # Get rank 14 solution for warm start
    U14, V14, W14 = _build_rank14_solution()
    
    # If rank 14 is valid, we have a baseline
    baseline_valid = _verify_solution(U14, V14, W14, n, m, p)
    
    # Strategy 1: Try structured perturbations of rank-14 solution
    if time.time() < deadline - 60:
        solution = _search_rank_13_structured(n, m, p, T_flat, U14, V14, W14, deadline)
        if solution is not None:
            best_solution = solution
    
    # Strategy 2: Try differential evolution for global search
    if best_solution is None and time.time() < deadline - 30:
        solution = _search_rank_13_de(n, m, p, T_flat, deadline)
        if solution is not None:
            best_solution = solution
    
    # Strategy 3: Try random restarts with least_squares (original approach, fewer restarts)
    if best_solution is None and time.time() < deadline - 10:
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            solution = _search_rank_13_ls(n, m, p, T_flat, restart, deadline)
            if solution is not None:
                best_solution = solution
                break
    
    if best_solution is not None:
        U, V, W = best_solution
        if _verify_solution(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fall back to rank 14
    if baseline_valid:
        return {"U": U14, "V": V14, "W": W14}
        
    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def _search_rank_13_structured(n, m, p, T_flat, U14, V14, W14, deadline):
    """Try structured perturbations of the rank-14 solution to find rank-13."""
    rank = 13
    U14_arr = np.array(U14)
    V14_arr = np.array(V14)
    W14_arr = np.array(W14)
    
    # Try removing each column from rank-14 and see if we can adjust
    for remove_col in range(14):
        if time.time() > deadline - 10:
            break
            
        # Remove column and try to fit
        mask = [i for i in range(14) if i != remove_col]
        U_init = U14_arr[:, mask]
        V_init = V14_arr[:, mask]
        W_init = W14_arr[:, mask]
        
        # Try to optimize from this initial guess
        solution = _polish_rank_13(n, m, p, T_flat, U_init, V_init, W_init, deadline)
        if solution is not None:
            return solution
    
    # Try averaging pairs of columns to create new rank-13 candidates
    for col1 in range(14):
        for col2 in range(col1 + 1, 14):
            if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
                
            # Create rank-13 by removing all except col1 and col2, then merging
            mask = [i for i in range(14) if i not in [col1, col2]]
            # Start with 12 columns, add one merged column
            U_init = np.column_stack([U14_arr[:, mask], (U14_arr[:, col1] + U14_arr[:, col2]) / 2])
            V_init = np.column_stack([V14_arr[:, mask], (V14_arr[:, col1] + V14_arr[:, col2]) / 2])
            W_init = np.column_stack([W14_arr[:, mask], (W14_arr[:, col1] + W14_arr[:, col2]) / 2])
            
            solution = _polish_rank_13(n, m, p, T_flat, U_init, V_init, W_init, deadline)
            if solution is not None:
                return solution
    
    return None

def _polish_rank_13(n, m, p, T_flat, U_init, V_init, W_init, deadline):
    """Polish a rank-13 initial guess using least_squares."""
    rank = 13
    U_init = np.array(U_init, dtype=float)
    V_init = np.array(V_init, dtype=float)
    W_init = np.array(W_init, dtype=float)
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def residuals(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return T_approx.flatten() - T_flat
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        
        if time.time() > deadline - 5:
            return None
            
        if result.cost < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Snap to nearest 0.5
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def _search_rank_13_de(n, m, p, T_flat, deadline):
    """Use differential evolution for global search of rank-13 solution."""
    rank = 13
    n_params = n*m*rank + m*p*rank + p*n*rank
    
    def objective(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        residual = T_approx.flatten() - T_flat
        return np.sum(residual**2)
    
    # Bounds: values should be in [-2, 2] for half-integer solutions
    bounds = [(-2, 2)] * n_params
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15, tol=1e-6,
            seed=42, polish=True, workers=1, init='sobol',
            maxfev=5000
        )
        
        if time.time() > deadline - 5:
            return None
            
        if result.fun < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Snap to nearest 0.5
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def _search_rank_13_ls(n, m, p, T_flat, seed, deadline):
    """Random restart with least_squares."""
    rank = 13
    np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    U /= np.sqrt(np.sum(U**2))
    V /= np.sqrt(np.sum(V**2))
    W /= np.sqrt(np.sum(W**2))
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(x):
        U_r = x[:n*m*rank].reshape(n*m, rank)
        V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return T_approx.flatten() - T_flat
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=1500)
        
        if time.time() > deadline - 5:
            return None
            
        if result.cost < 1e-6:
            x_opt = result.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]
    
    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]
    
    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]
    
    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]
    
    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]
    
    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True
