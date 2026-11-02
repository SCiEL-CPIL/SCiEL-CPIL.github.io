import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

def find_decomp_223():
    """
    Finds a rank 10 decomposition for 2x2x3 matrix multiplication.
    Returns U, V, W as lists of lists of Fractions or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_sol = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 30
    
    for seed in range(200):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        init_size = n*m*rank + m*p*rank + p*n*rank
        x0 = np.random.uniform(-1, 1, init_size)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=3000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_sol = sol.x
                if err < 1e-12:
                    break
        except Exception:
            continue
            
    if best_err > 1e-9 or best_sol is None:
        return None

    U_flat = best_sol[:n*m*rank]
    V_flat = best_sol[n*m*rank:n*m*rank + m*p*rank]
    W_flat = best_sol[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    U_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if val != 1:
                    return None
                    
    return U_frac, V_frac, W_frac

def find_decomp_246_rank40():
    """
    Direct search for rank-40 decomposition of 2x4x6 matrix multiplication.
    Uses tensor network disentanglement: initialize from Kronecker products
    of 2x2x2 Strassen blocks, then optimize locally.
    """
    n, m, p = 2, 4, 6
    rank = 40
    total = n * m * p  # 48 constraints
    
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = np.zeros(total)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Initialize from Strassen-based rank-42, then zero out 2 columns
    # and optimize
    U0 = np.zeros((n*m, rank))
    V0 = np.zeros((m*p, rank))
    W0 = np.zeros((p*n, rank))
    
    # Build rank-42 Strassen init
    def add_strassen_block(U, V, W, A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        return start_col
    
    col = 0
    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        col = add_strassen_block(U0, V0, W0, A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        col = add_strassen_block(U0, V0, W0, A_R_idx, B_Bk_idx, C_idx, col)
    
    # Now we have rank-42 in first 42 columns. For rank-40, we need to
    # compress. Use a different init: random with structure
    # Try multiple random inits mixed with structured
    
    best_sol = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80
    
    # Strategy 1: Start from Strassen rank-42, set last 2 cols to small values, optimize
    x0_strassen = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    # Zero out last 2 columns to force rank-40
    x0_strassen[:(n*m)*rank][(n*m)*38:(n*m)*40] = 0.01
    x0_strassen[:(n*m)*rank][(n*m)*39:(n*m)*40] = 0.01
    v_offset = (n*m)*rank
    x0_strassen[v_offset:v_offset+(m*p)*rank][(m*p)*38:(m*p)*40] = 0.01
    x0_strassen[v_offset:v_offset+(m*p)*rank][(m*p)*39:(m*p)*40] = 0.01
    w_offset = v_offset + (m*p)*rank
    x0_strassen[w_offset:w_offset+(p*n)*rank][(p*n)*38:(p*n)*40] = 0.01
    x0_strassen[w_offset:w_offset+(p*n)*rank][(p*n)*39:(p*n)*40] = 0.01
    
    try:
        sol = least_squares(residuals, x0_strassen, method='lm', max_nfev=5000)
        best_err = sol.cost
        best_sol = sol.x
    except Exception:
        pass
    
    # Strategy 2: Random inits with scaling
    for seed in range(300):
        if time.time() > deadline:
            break
        if best_err < 1e-12:
            break
            
        np.random.seed(seed)
        init_size = n*m*rank + m*p*rank + p*n*rank
        # Use structured random: small entries
        x0 = np.random.uniform(-0.5, 0.5, init_size)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_sol = sol.x
                if err < 1e-12:
                    break
        except Exception:
            continue
    
    if best_sol is None or best_err > 1e-8:
        return None
    
    U_flat = best_sol[:n*m*rank]
    V_flat = best_sol[n*m*rank:n*m*rank + m*p*rank]
    W_flat = best_sol[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    # Try to round to half-integers and verify
    U_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if val != 1:
                    return None
                    
    return U_frac, V_frac, W_frac

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    global_deadline = start_time + 140  # Safety margin before 150s kill
    
    # First try direct rank-40 search on full tensor
    direct_40 = find_decomp_246_rank40()
    
    if direct_40 is not None:
        U40, V40, W40 = direct_40
        rank = 40
        U = [[float(x) for x in row] for row in U40]
        V = [[float(x) for x in row] for row in V40]
        W = [[float(x) for x in row] for row in W40]
        return {"U": U, "V": V, "W": W}
    
    # Check time
    if time.time() > global_deadline - 30:
        # Not enough time, go straight to fallback
        pass
    else:
        # Try Kronecker product approach with 2x2x3 sub-decompositions
        decomp_223 = find_decomp_223()
        
        if decomp_223 is not None:
            U2, V2, W2 = decomp_223
            rank = 40
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            def add_block(j_start, k_start, col_offset):
                for i in range(2):
                    for j_local in range(2):
                        for k_local in range(3):
                            for r in range(10):
                                u_idx = i*4 + (j_start + j_local)
                                v_idx = (j_start + j_local)*6 + (k_start + k_local)
                                w_idx = (k_start + k_local)*2 + i
                                
                                U[u_idx][col_offset + r] += U2[i*2 + j_local][r]
                                V[v_idx][col_offset + r] += V2[j_local*3 + k_local][r]
                                W[w_idx][col_offset + r] += W2[k_local*2 + i][r]
                                
            add_block(0, 0, 0)
            add_block(0, 3, 10)
            add_block(2, 0, 20)
            add_block(2, 3, 30)
            
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 42 Strassen
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)

    return {"U": U, "V": V, "W": W}
