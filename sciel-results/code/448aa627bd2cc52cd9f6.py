import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
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
    
    def find_decomp(target_rank, max_restarts=500, vals=None):
        """
        Finds a decomposition of rank target_rank for n x m * m x p.
        Returns dict with U, V, W or None.
        """
        if vals is None:
            # Bias towards sparsity and small integers
            vals = [-1, -0.5, 0, 0.5, 1]
        
        num_eqs = n * m * p
        num_vars_W = p * n * target_rank
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization for U and V with sparsity
            # 50% chance of 0, 25% chance of 1, 25% chance of -1
            # This helps in finding sparse decompositions
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            
            for i in range(n * m):
                for r in range(target_rank):
                    if random.random() < 0.5:
                        U_rand[i, r] = 0
                    else:
                        U_rand[i, r] = random.choice([-1, 1])
            
            for j in range(m * p):
                for r in range(target_rank):
                    if random.random() < 0.5:
                        V_rand[j, r] = 0
                    else:
                        V_rand[j, r] = random.choice([-1, 1])
            
            # Check if U and V are too sparse or all zero
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
    
    # Prioritize lower ranks if time permits, but start from high to ensure we find *something* better than schoolbook quickly?
    # Actually, if we find 35, it's better than 36.
    # Let's try 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        # Allocate more restarts for lower ranks if we have time?
        # For now, fixed restarts.
        result = find_decomp(target_rank, max_restarts=200)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Don't break, try to find even lower rank if time allows
            # But if we find a low rank, it's likely the minimum.
            # Let's continue to try lower ranks for a bit.
    
    return best_result
# EVOLVE-BLOCK-END
