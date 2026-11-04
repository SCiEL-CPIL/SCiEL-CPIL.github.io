import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try to find a lower rank solution
    best_solution = None
    
    # We'll try ranks from a few below schoolbook down to see if we can find a valid one
    # For 3x4x6, the known lower bounds are around 70-75, so try from 70 down to 60
    for r in range(min(rank_school - 1, 72), max(59, rank_school - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(42 + r * 100 + attempt)
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            
            def flatten(x):
                return np.concatenate([x.ravel() for x in x])
            
            def unflatten(x, shapes):
                mats = []
                idx = 0
                for shape in shapes:
                    size = np.prod(shape)
                    mats.append(x[idx:idx+size].reshape(shape))
                    idx += size
                return mats
            
            shapes = [(n*m, r), (m*p, r), (p*n, r)]
            x0 = flatten([U_init, V_init, W_init])
            
            def residuals(x):
                U, V, W = unflatten(x, shapes)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            res.append(val - 1.0)
                return np.array(res)
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if time.time() > deadline:
                    continue
                    
                # Check if we got a good solution
                if sol.cost < 1e-8:
                    U, V, W = unflatten(sol.x, shapes)
                    
                    # Try to round to nearest integers or half-integers
                    # First try integers
                    U_round = np.round(U)
                    V_round = np.round(V)
                    W_round = np.round(W)
                    
                    # Verify integer solution
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_round[i*m+j], V_round[j*p+k] * W_round[k*n+i])
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_solution = (U_round.astype(int), V_round.astype(int), W_round.astype(int))
                        found = True
                        break
                    
                    # Try half-integers (multiply by 2, round, divide by 2)
                    U_half = np.round(U * 2) / 2.0
                    V_half = np.round(V * 2) / 2.0
                    W_half = np.round(W * 2) / 2.0
                    
                    # Verify half-integer solution
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_half[i*m+j], V_half[j*p+k] * W_half[k*n+i])
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Store as is (they're already floats representing half-integers)
                        best_solution = (U_half, V_half, W_half)
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break

    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
