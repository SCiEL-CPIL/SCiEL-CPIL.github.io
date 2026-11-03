import time
import random
import math
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
DEADLINE = 140  # Leave 10s safety margin for the 150s wall-clock limit

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def get_tensor(n, m, p):
        """Return the multiplication tensor as a dictionary of non-zero entries."""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[(a, b, c)] = 1
        return T

    def validate(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = sum(U[a][c_idx] * V[b][c_idx] * W[c][c_idx] for c_idx in range(rank))
                    if abs(s - 1) > tol:
                        return False
        # Check zeros (sample if rank is large)
        if rank <= 50:
            # Check all zeros
            ones_set = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ones_set.add((i * m + j, j * p + k, k * n + i))
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        if (a, b, c) in ones_set:
                            continue
                        s = sum(U[a][c_idx] * V[b][c_idx] * W[c][c_idx] for c_idx in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_find_decomposition(n, m, p, target_rank, time_limit=30, num_restarts=15):
        """Try to find a decomposition of rank target_rank."""
        start_time = time.time()
        T = get_tensor(n, m, p)
        
        def objective(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            err = 0.0
            for (a, b, c), val in T.items():
                s = np.dot(U[a], np.multiply(V[b], W[c]))
                err += (s - val) ** 2
            err += 0.01 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return err

        best_sol = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            if time.time() > DEADLINE:
                break
            
            # Random initialization with varied scales
            scale = 0.1 * (1 + 0.5 * restart)
            U = np.random.randn(n*m, target_rank) * scale
            V = np.random.randn(m*p, target_rank) * scale
            W = np.random.randn(p*n, target_rank) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < best_err:
                best_err = res.fun
                best_sol = res.x
                
                if best_err < 1e-6:
                    U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            if best_err < 1e-4:
                U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    def construct_from_333(U_333, V_333, W_333):
        """
        Given a rank-r decomposition for 3x3x3, construct a rank-2r decomposition for 3x3x6.
        
        The 3x3x6 multiplication: A (3x3) * B (3x6) = C (3x6)
        View B as two 3x3 blocks: B = [B1 | B2] where B1, B2 are 3x3.
        Then C = [C1 | C2] where C1 = A*B1, C2 = A*B2.
        
        If we have a rank-r decomposition for 3x3x3:
          U_333: 9 x r
          V_333: 9 x r  
          W_333: 9 x r
          
        For 3x3x6:
          U_336: 9 x 2r
          V_336: 18 x 2r
          W_336: 18 x 2r
          
        Construction:
          U_336[:, :r] = U_333
          U_336[:, r:] = U_333
          
          V_336[:9, :r] = V_333  (B1 part, rows 0-8 for j=0,1,2 and k=0,1,2)
          V_336[9:, :r] = 0      (B2 part is zero in first r terms)
          V_336[:9, r:] = 0      (B1 part is zero in second r terms)
          V_336[9:, r:] = V_333  (B2 part, rows 9-17 for j=0,1,2 and k=3,4,5)
          
          W_336[:9, :r] = W_333  (C1 part, rows 0-8 for k=0,1,2 and i=0,1,2)
          W_336[9:, :r] = 0      (C2 part is zero in first r terms)
          W_336[:9, r:] = 0      (C1 part is zero in second r terms)
          W_336[9:, r:] = W_333  (C2 part, rows 9-17 for k=3,4,5 and i=0,1,2)
        """
        r = len(U_333[0])
        
        U_336 = [[0] * (2*r) for _ in range(9)]
        V_336 = [[0] * (2*r) for _ in range(18)]
        W_336 = [[0] * (2*r) for _ in range(18)]
        
        for i in range(9):
            for c in range(r):
                U_336[i][c] = U_333[i][c]
                U_336[i][r + c] = U_333[i][c]
        
        for i in range(9):
            for c in range(r):
                V_336[i][c] = V_333[i][c]
                W_336[i][c] = W_333[i][c]
        
        for i in range(9):
            for c in range(r):
                V_336[9 + i][r + c] = V_333[i][c]
                W_336[9 + i][r + c] = W_333[i][c]
        
        return {"U": U_336, "V": V_336, "W": W_336}

    # Strategy:
    # 1. Find rank-23 decomposition for 3x3x3
    # 2. Construct rank-46 decomposition for 3x3x6 using block structure
    # 3. If that fails, try direct search for 3x3x6
    
    # Step 1: Find rank-23 for 3x3x3
    res_333 = try_find_decomposition(3, 3, 3, 23, time_limit=25, num_restarts=20)
    
    if res_333 is not None:
        # Step 2: Construct rank-46 for 3x3x6
        result = construct_from_333(res_333['U'], res_333['V'], res_333['W'])
        if validate(result['U'], result['V'], result['W'], 3, 3, 6):
            return result
    
    # Step 3: Direct search for 3x3x6 with various ranks
    for r in range(46, 55):
        if time.time() > DEADLINE:
            break
        remaining_time = DEADLINE - time.time()
        if remaining_time < 10:
            break
        res = try_find_decomposition(3, 3, 6, r, time_limit=min(12, remaining_time - 5), num_restarts=8)
        if res is not None:
            return res
    
    # Fallback
    return schoolbook_decomposition(3, 3, 6)
