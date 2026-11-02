import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 110

    # Pre-compute Strassen blocks
    def get_strassen_block(k_start):
        U_block = np.zeros((4, 7))
        V_block = np.zeros((14, 7))
        W_block = np.zeros((14, 7))
        
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        strassen_W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        for r in range(7):
            for i in range(4):
                U_block[i, r] = strassen_U[r][i]
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + (2 * k_start)
                elif loc_idx == 1: v_idx = 0 * p + (2 * k_start + 1)
                elif loc_idx == 2: v_idx = 1 * p + (2 * k_start)
                elif loc_idx == 3: v_idx = 1 * p + (2 * k_start + 1)
                V_block[v_idx, r] = coeff
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0: w_idx = (2 * k_start) * n + 0
                elif loc_idx == 1: w_idx = (2 * k_start + 1) * n + 0
                elif loc_idx == 2: w_idx = (2 * k_start) * n + 1
                elif loc_idx == 3: w_idx = (2 * k_start + 1) * n + 1
                W_block[w_idx, r] = coeff
        return U_block, V_block, W_block

    # Pre-compute Rank 25 solution
    U_25 = np.zeros((4, 25))
    V_25 = np.zeros((14, 25))
    W_25 = np.zeros((14, 25))
    
    U_b1, V_b1, W_b1 = get_strassen_block(0)
    U_25[:, 0:7] += U_b1
    V_25[:, 0:7] += V_b1
    W_25[:, 0:7] += W_b1
    
    U_b2, V_b2, W_b2 = get_strassen_block(1)
    U_25[:, 7:14] += U_b2
    V_25[:, 7:14] += V_b2
    W_25[:, 7:14] += W_b2
    
    U_b3, V_b3, W_b3 = get_strassen_block(2)
    U_25[:, 14:21] += U_b3
    V_25[:, 14:21] += V_b3
    W_25[:, 14:21] += W_b3
    
    # Schoolbook for col 6
    U_25[0, 21] = 1; U_25[1, 22] = 1; U_25[2, 23] = 1; U_25[3, 24] = 1
    V_25[0*7 + 6, 21] = 1; V_25[1*7 + 6, 22] = 1; V_25[0*7 + 6, 23] = 1; V_25[1*7 + 6, 24] = 1
    W_25[6*2 + 0, 21] = 1; W_25[6*2 + 0, 22] = 1; W_25[6*2 + 1, 23] = 1; W_25[6*2 + 1, 24] = 1

    # Try to find Rank 24 using epigraph formulation with auxiliary variable
    if time.time() < deadline - 50:
        R = 10
        
        # Epigraph formulation: minimize t subject to |sum_r U[i,r]*V[j,r]*W[k,r] - 1| <= t for all (i,j,k)
        # This is a nonsmooth problem, so we use a smooth surrogate: soft-max of squared residuals
        def epigraph_objective(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+6*R].reshape(6, R)
            W = x[4*R+6*R:].reshape(6, R)
            
            # Compute all 12 bilinear forms
            residuals = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.dot(U[i, :], V[j*3+k, :] * W[k*2+i, :])
                        residuals.append(s - 1.0)
            residuals = np.array(residuals)
            
            # Smooth max via log-sum-exp (temperature=1)
            max_res = np.max(np.abs(residuals))
            # Add a penalty for large values to encourage all to be small
            return max_res + 0.01 * np.sum(residuals**2)
        
        def epigraph_residuals(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+6*R].reshape(6, R)
            W = x[4*R+6*R:].reshape(6, R)
            
            res = np.zeros(12)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.dot(U[i, :], V[j*3+k, :] * W[k*2+i, :])
                        res[idx] = s - 1.0
                        idx += 1
            return res

        best_res = float('inf')
        best_x = None
        
        # Try multiple random initializations with different strategies
        for attempt in range(40):
            if time.time() > deadline - 50:
                break
            
            # Different initialization strategies
            if attempt % 4 == 0:
                np.random.seed(attempt * 123)
                x0 = np.random.randn(16*R) * 0.5
            elif attempt % 4 == 1:
                np.random.seed(attempt * 456)
                x0 = np.random.randn(16*R) * 0.3
            elif attempt % 4 == 2:
                np.random.seed(attempt * 789)
                x0 = np.random.randn(16*R) * 0.7
            else:
                np.random.seed(attempt * 321)
                x0 = np.random.randn(16*R) * 0.5
                # Add structure: make some entries zero
                mask = np.random.rand(16*R) < 0.3
                x0[mask] = 0
            
            try:
                # Use least_squares with tighter tolerances
                sol = least_squares(epigraph_residuals, x0, method='lm', 
                                   max_nfev=2000, xtol=1e-10, ftol=1e-10)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-10:
                        break
            except Exception:
                pass
        
        # High-precision polish pass if we found a good solution
        if best_x is not None and best_res < 1e-4:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+6*R].reshape(6, R)
            W = best_x[4*R+6*R:].reshape(6, R)
            
            # Polish with very tight tolerances
            try:
                sol_polish = least_squares(epigraph_residuals, best_x, method='lm',
                                          max_nfev=5000, xtol=1e-14, ftol=1e-14)
                if sol_polish.cost < best_res:
                    best_res = sol_polish.cost
                    best_x = sol_polish.x
            except Exception:
                pass
        
        if best_x is not None and best_res < 1e-6:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+6*R].reshape(6, R)
            W = best_x[4*R+6*R:].reshape(6, R)
            
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.dot(U_rounded[i, :], V_rounded[j*3+k, :] * W_rounded[k*2+i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                U_24 = np.zeros((4, 24))
                V_24 = np.zeros((14, 24))
                W_24 = np.zeros((14, 24))
                
                U_24[:, 0:7] += U_b1
                V_24[:, 0:7] += V_b1
                W_24[:, 0:7] += W_b1
                
                U_24[:, 7:14] += U_b2
                V_24[:, 7:14] += V_b2
                W_24[:, 7:14] += W_b2
                
                U_24[:, 14:24] += U_rounded
                for j in range(2):
                    for k_loc in range(3):
                        V_24[j*7 + 4 + k_loc, 14:24] += V_rounded[j*3 + k_loc, :]
                for k_loc in range(3):
                    for i in range(2):
                        W_24[(4 + k_loc)*2 + i, 14:24] += W_rounded[k_loc*2 + i, :]
                
                return {"U": U_24.tolist(), "V": V_24.tolist(), "W": W_24.tolist()}

    return {"U": U_25.tolist(), "V": V_25.tolist(), "W": W_25.tolist()}
