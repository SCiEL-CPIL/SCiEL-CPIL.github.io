import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
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
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def find_decomp(target_rank, max_restarts=200, vals=None):
        """
        Finds a decomposition of rank target_rank for n x m * m x p.
        Returns dict with U, V, W or None.
        """
        if vals is None:
            vals = [-1, -0.5, 0, 0.5, 1]
        
        num_eqs = n * m * p
        num_vars_W = p * n * target_rank
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization for U and V
            U_rand = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)])
            V_rand = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)])
            
            # Check if U and V are too sparse
            if np.all(U_rand == 0) or np.all(V_rand == 0):
                continue
            
            # Build linear system for W
            A = np.zeros((num_eqs, num_vars_W))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            u_val = U_rand[i * m + j, r]
                            v_val = V_rand[j * p + k, r]
                            if u_val != 0 and v_val != 0:
                                var_idx = (k * n + i) * target_rank + r
                                A[eq_idx, var_idx] = u_val * v_val
            
            try:
                w_sol, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if solution is close to half-integers
            w_rounded = np.round(w_sol * 2) / 2.0
            
            # Verify
            U_list = [[U_rand[i * m + j, r] for r in range(target_rank)] for i in range(n) for j in range(m)]
            V_list = [[V_rand[j * p + k, r] for r in range(target_rank)] for j in range(m) for k in range(p)]
            W_list = [[w_rounded[(k * n + i) * target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
            
            if verify(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try decreasing ranks from 35 down to 20
    best_result = schoolbook()
    best_rank = n * m * p
    
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        result = find_decomp(target_rank, max_restarts=100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the minimum rank we searched
    
    return best_result
# EVOLVE-BLOCK-END
