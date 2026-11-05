import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 8
    
    # First, generate the Strassen rank-56 solution as baseline
    def generate_strassen_solution():
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
            for (i, j), val in u_coeffs.items():
                if val != 0:
                    U[i*m + j][col_idx] += val
            for (j, k), val in v_coeffs.items():
                if val != 0:
                    V[j*p + k][col_idx] += val
            for (k, i), val in w_coeffs.items():
                if val != 0:
                    W[k*n + i][col_idx] += val
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_term(
                    {(0, j0): 1, (1, j0+1): 1},
                    {(j0, k0): 1, (j0+1, k0+1): 1},
                    {(k0, 0): 1, (k0+1, 1): 1},
                    col
                )
                col += 1
                add_term(
                    {(1, j0): 1, (1, j0+1): 1},
                    {(j0, k0): 1},
                    {(k0, 1): 1, (k0+1, 1): -1},
                    col
                )
                col += 1
                add_term(
                    {(0, j0): 1},
                    {(j0, k0+1): 1, (j0+1, k0+1): -1},
                    {(k0+1, 0): 1, (k0+1, 1): 1},
                    col
                )
                col += 1
                add_term(
                    {(1, j0+1): 1},
                    {(j0+1, k0): 1, (j0, k0): -1},
                    {(k0, 0): 1, (k0, 1): 1},
                    col
                )
                col += 1
                add_term(
                    {(0, j0): 1, (0, j0+1): 1},
                    {(j0+1, k0+1): 1},
                    {(k0, 0): -1, (k0+1, 0): 1},
                    col
                )
                col += 1
                add_term(
                    {(1, j0): 1, (0, j0): -1},
                    {(j0, k0): 1, (j0, k0+1): 1},
                    {(k0+1, 1): 1},
                    col
                )
                col += 1
                add_term(
                    {(0, j0+1): 1, (1, j0+1): -1},
                    {(j0+1, k0): 1, (j0+1, k0+1): 1},
                    {(k0, 0): 1},
                    col
                )
                col += 1
        
        return U, V, W, rank
    
    U_strassen, V_strassen, W_strassen, rank_strassen = generate_strassen_solution()
    
    # Validate the Strassen solution
    U_np = np.array(U_strassen, dtype=float)
    V_np = np.array(V_strassen, dtype=float)
    W_np = np.array(W_strassen, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    strassen_valid = np.allclose(T_recon, T_target, atol=1e-9)
    
    # Try to find a lower rank solution using greedy random search
    # This is a heuristic search - may or may not find improvement
    best_U, best_V, best_W, best_rank = U_strassen, V_strassen, W_strassen, rank_strassen
    
    if strassen_valid and _time_left() > 30.0:
        # Try ranks from 55 down to 48
        for target_rank in range(55, 47, -1):
            if _time_left() < 15.0:
                break
            
            # Random initialization with small entries
            np.random.seed(42 + target_rank)
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Greedy improvement: try to match the target tensor
            # This is a simplified approach - full optimization would be much more complex
            improved = False
            for _ in range(100):
                if _time_left() < 10.0:
                    break
                
                T_recon_try = np.einsum('ur,vr,wr->uvw', U_try, V_try, W_try)
                error = np.linalg.norm(T_recon_try - T_target)
                
                if error < 1e-8:
                    improved = True
                    break
                
                # Simple gradient-free perturbation
                U_try += np.random.randn(n*m, target_rank) * 0.01
                V_try += np.random.randn(m*p, target_rank) * 0.01
                W_try += np.random.randn(p*n, target_rank) * 0.01
            
            if improved:
                # Convert to list format with integer/half-integer entries if possible
                U_list = [[float(x) for x in row] for row in U_try]
                V_list = [[float(x) for x in row] for row in V_try]
                W_list = [[float(x) for x in row] for row in W_try]
                
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
