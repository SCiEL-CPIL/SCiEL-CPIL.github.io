import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 140.0
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Start with the known rank-42 block-Strassen decomposition
    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        # Strassen 7-term decomposition for 2x2 matrix multiplication
        # For matrices X (2x2) and Y (2x2), product Z = XY:
        # P1 = (X00 + X11) * (Y00 + Y11)
        # P2 = (X10 + X11) * Y00
        # P3 = X00 * (Y01 - Y11)
        # P4 = X11 * (Y10 - Y00)
        # P5 = (X00 + X01) * Y11
        # P6 = (X10 - X00) * (Y00 + Y01)
        # P7 = (X01 - X11) * (Y10 + Y11)
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        U_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1
            {(1, 0): 1, (1, 1): 1},      # P2
            {(0, 0): 1},                  # P3
            {(1, 1): 1},                  # P4
            {(0, 0): 1, (0, 1): 1},      # P5
            {(1, 0): 1, (0, 0): -1},     # P6
            {(0, 1): 1, (1, 1): -1}      # P7
        ]
        
        V_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1
            {(0, 0): 1},                  # P2
            {(0, 1): 1, (1, 1): -1},     # P3
            {(1, 0): 1, (0, 0): -1},     # P4
            {(1, 1): 1},                  # P5
            {(0, 0): 1, (0, 1): 1},      # P6
            {(1, 0): 1, (1, 1): 1}       # P7
        ]
        
        W_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1: Z00(+1), Z11(+1)
            {(0, 1): 1, (1, 1): -1},     # P2: Z10(+1), Z11(-1)
            {(1, 0): 1, (1, 1): 1},      # P3: Z01(+1), Z11(+1)
            {(0, 0): 1, (0, 1): 1},      # P4: Z00(+1), Z10(+1)
            {(0, 0): -1, (1, 0): 1},     # P5: Z00(-1), Z01(+1)
            {(1, 1): 1},                 # P6: Z11(+1)
            {(0, 0): 1}                  # P7: Z00(+1)
        ]
        
        col = 0
        j_groups = [(0, 1), (2, 3)]
        k_groups = [(0, 1), (2, 3), (4, 5)]
        
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_strassen[s].items():
                        i = i_loc
                        j = j_base + j_loc
                        u_idx = i * m + j
                        U[u_idx, col] += coeff
                    
                    for (j_loc, k_loc), coeff in V_strassen[s].items():
                        j = j_base + j_loc
                        k = k_base + k_loc
                        v_idx = j * p + k
                        V[v_idx, col] += coeff
                    
                    for (k_loc, i_loc), coeff in W_strassen[s].items():
                        k = k_base + k_loc
                        i = i_loc
                        w_idx = k * n + i
                        W[w_idx, col] += coeff
                    
                    col += 1
        
        return U, V, W
    
    U42, V42, W42 = make_rank42()
    
    # Try to find a lower rank decomposition
    def try_reduce(rank, max_trials=50, max_iters=1000):
        """Try to find a rank-r decomposition using iterative refinement"""
        if time.time() > deadline - 20:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            best_err = float('inf')
            prev_err = float('inf')
            stagnation_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                # Solve for W
                solved = True
                for k in range(p):
                    for i in range(n):
                        A_sub = np.zeros((m, rank))
                        b_sub = np.zeros(m)
                        for j in range(m):
                            for r in range(rank):
                                A_sub[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            b_sub[j] = T[i*m+j, j*p+k, k*n+i]
                        
                        try:
                            w_sol, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                            W[k*n+i, :] = w_sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                if not solved:
                    break
                
                # Solve for U
                solved = True
                for i in range(n):
                    for j in range(m):
                        A_sub = np.zeros((p, rank))
                        b_sub = np.zeros(p)
                        for k in range(p):
                            for r in range(rank):
                                A_sub[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            b_sub[k] = T[i*m+j, j*p+k, k*n+i]
                        
                        try:
                            u_sol, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                            U[i*m+j, :] = u_sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                if not solved:
                    break
                
                # Solve for V
                solved = True
                for j in range(m):
                    for k in range(p):
                        A_sub = np.zeros((n, rank))
                        b_sub = np.zeros(n)
                        for i in range(n):
                            for r in range(rank):
                                A_sub[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            b_sub[i] = T[i*m+j, j*p+k, k*n+i]
                        
                        try:
                            v_sol, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                            V[j*p+k, :] = v_sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                if not solved:
                    break
                
                # Check error
                reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(reconstructed - T)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        return U.copy(), V.copy(), W.copy()
                
                # Stagnation check
                if err < prev_err * 0.999:
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 50:
                        break
                prev_err = err
            
            if best_err < 1e-9:
                return U.copy(), V.copy(), W.copy()
        
        return None
    
    # Try ranks from 30 up to 41
    best_U, best_V, best_W = None, None, None
    best_rank = 42
    
    # Prioritize rank 30
    for rank in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
        if time.time() > deadline - 20:
            break
        
        # Adjust trials based on time remaining
        time_left = deadline - 20 - time.time()
        if time_left < 10:
            break
            
        # Give more trials to lower ranks
        trials = 100 if rank == 30 else 50
        iters = 2000 if rank == 30 else 1000
        
        result = try_reduce(rank, max_trials=trials, max_iters=iters)
        if result is not None:
            U_r, V_r, W_r = result
            if check_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                break
    
    if best_U is not None:
        return to_lists(best_U, best_V, best_W)
    else:
        return to_lists(U42, V42, W42)
