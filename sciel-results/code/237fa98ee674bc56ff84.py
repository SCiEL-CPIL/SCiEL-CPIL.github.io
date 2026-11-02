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
    
    # Residual function
    def residuals(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res.append(val - 1.0)
        return res

    best_sol = None
    best_err = float('inf')
    
    # Try a few random initializations
    for seed in range(20):
        np.random.seed(seed)
        init_size = n*m*rank + m*p*rank + p*n*rank
        x0 = np.random.uniform(-1, 1, init_size)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=20000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_sol = sol.x
                if err < 1e-12:
                    break
        except Exception:
            continue
            
    if best_err > 1e-9:
        return None

    U_flat = best_sol[:n*m*rank]
    V_flat = best_sol[n*m*rank:n*m*rank + m*p*rank]
    W_flat = best_sol[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    # Convert to Fractions (round to nearest 0.5)
    U_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    # Verify exactness with Fractions
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                max_err = max(max_err, abs(val - 1))
                
    if max_err == 0:
        return U_frac, V_frac, W_frac
    else:
        return None

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    
    # Try to find rank 40 decomposition
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
                            
        # Block 1: A_L (j=0,1), B_T1 (j=0,1, k=0,1,2)
        add_block(0, 0, 0)
        
        # Block 2: A_L (j=0,1), B_T2 (j=0,1, k=3,4,5)
        add_block(0, 3, 10)
        
        # Block 3: A_R (j=2,3), B_B1 (j=2,3, k=0,1,2)
        add_block(2, 0, 20)
        
        # Block 4: A_R (j=2,3), B_B2 (j=2,3, k=3,4,5)
        add_block(2, 3, 30)
        
        return {"U": U, "V": V, "W": W}
    else:
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
