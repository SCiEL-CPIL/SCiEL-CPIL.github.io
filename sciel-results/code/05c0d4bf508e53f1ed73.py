import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
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
    
    def verify_exact(U, V, W):
        rank = len(U[0])
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(R):
        """Try to find a decomposition of rank R using structured approach."""
        if time.time() > deadline - 20.0:
            return None
        
        # Use a greedy approach: start with schoolbook and try to eliminate terms
        # by finding linear dependencies
        
        # Alternative: use ALS with better initialization
        # Initialize with schoolbook truncated
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Initialize using first R terms of schoolbook
        for r in range(R):
            i = r // (m * p)
            rem = r % (m * p)
            j = rem // p
            k = rem % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        # ALS refinement
        for iteration in range(100):
            if time.time() > deadline - 10.0:
                break
            
            # Solve for U
            # M_U[b,c,r] = V[b,r] * W[c,r]
            M_U = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
            for a in range(n * m):
                t = T[a].flatten()
                sol, _, _, _ = np.linalg.lstsq(M_U, t, rcond=None)
                U[a] = sol
            
            # Solve for V
            M_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
            for b in range(m * p):
                t = T[:, b, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(M_V, t, rcond=None)
                V[b] = sol
            
            # Solve for W
            M_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
            for c in range(p * n):
                t = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(M_W, t, rcond=None)
                W[c] = sol
            
            # Check convergence
            err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            if err < 1e-10:
                # Try to quantize
                U_q = np.round(U * 2) / 2
                V_q = np.round(V * 2) / 2
                W_q = np.round(W * 2) / 2
                
                U_list = U_q.tolist()
                V_list = V_q.tolist()
                W_list = W_q.tolist()
                
                if verify_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 30.0:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
