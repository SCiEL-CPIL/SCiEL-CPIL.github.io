import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will solve for M_{n, m, p'} where p' divides p
    # If we find rank r for M_{n, m, p'}, we can repeat p/p' times to get rank r*(p/p') for M_{n, m, p}
    
    # Target: M_{2, 3, 2}
    n_sub, m_sub, p_sub = 2, 3, 2
    
    # Check if p is divisible by p_sub
    if p % p_sub == 0:
        repeats = p // p_sub
        # Try to find minimal rank for M_{2,3,2}
        best_rank = None
        best_U = None
        best_V = None
        best_W = None
        
        deadline = time.time() + 100 # 100 seconds budget for search
        
        # Try ranks 6, 7, 8...
        # Lower bound is max(n*m, m*p, p*n) = max(6, 6, 4) = 6
        for r in range(6, 13): # Up to 12 (schoolbook for sub-problem)
            if time.time() > deadline:
                break
            if _find_decomposition(n_sub, m_sub, p_sub, r, deadline):
                best_rank = r
                best_U, best_V, best_W = _last_solution
                break
        
        if best_rank is not None:
            # Construct full U, V, W
            full_rank = best_rank * repeats
            U_full = [[0] * full_rank for _ in range(n * m)]
            V_full = [[0] * full_rank for _ in range(m * p)]
            W_full = [[0] * full_rank for _ in range(p * n)]
            
            for rep in range(repeats):
                offset = rep * best_rank
                # Map indices
                # U indices: i*m + j (0..5) -> same in sub
                # V indices: j*p + k. In sub, k is 0..1. In full, k = rep*2 + k_sub
                # W indices: k*n + i. In sub, k is 0..1. In full, k = rep*2 + k_sub
                
                for i in range(n_sub):
                    for j in range(m_sub):
                        u_idx = i * m_sub + j
                        u_full_idx = i * m + j
                        for c in range(best_rank):
                            U_full[u_full_idx][offset + c] = best_U[u_idx][c]
                
                for j in range(m_sub):
                    for k_sub in range(p_sub):
                        v_idx = j * p_sub + k_sub
                        k_full = rep * p_sub + k_sub
                        v_full_idx = j * p + k_full
                        for c in range(best_rank):
                            V_full[v_full_idx][offset + c] = best_V[v_idx][c]
                
                for k_sub in range(p_sub):
                    for i in range(n_sub):
                        w_idx = k_sub * n_sub + i
                        k_full = rep * p_sub + k_sub
                        w_full_idx = k_full * n + i
                        for c in range(best_rank):
                            W_full[w_full_idx][offset + c] = best_W[w_idx][c]
                
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: Schoolbook
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

_last_solution = None

def _find_decomposition(n, m, p, r, deadline):
    global _last_solution
    # Try to find U, V, W of rank r
    # Equations: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # Random search with small integers
    # Fix W to random small integers, solve for U, V?
    # Or just brute force small values for all?
    # Given low number of equations (12), we can try to satisfy them.
    
    # Strategy: 
    # 1. Randomly initialize W with {-1, 0, 1}
    # 2. For each (i,j,k), we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # This is hard to solve directly.
    
    # Alternative: Use the fact that we can scale rows/cols.
    # Let's try a structured approach or random search with backtracking?
    # Since r is small (6), and variables are few, random search might work if we guide it.
    
    # Let's try to set U and W to identity-like or simple patterns.
    # Actually, for rank 6, we might be able to find a solution where U, V, W have entries in {-1, 0, 1}.
    
    # Let's just do a few random restarts with integer search.
    import random
    random.seed(42)
    
    # We have 12 equations.
    # Variables: 6*r (U), 6*r (V), 4*r (W).
    
    # Try to find a solution by assigning values to U and W, then solving for V?
    # If we fix U and W, the equations are linear in V.
    # For each (i,j,k), sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c] = 1.
    # This is not linear in V because V index depends on j and k, but the sum is over c.
    # Wait, V[j*p+k, c] is a variable for each j, k, c.
    # The equation is:
    # sum_c ( U[i*m+j, c] * W[k*n+i, c] ) * V[j*p+k, c] = 1.
    # This is linear in the variables V[j*p+k, c].
    # However, the same V[j*p+k, c] might appear in multiple equations?
    # No, V[j*p+k, c] is specific to j, k, c.
    # But the coefficient A_{i,j,k,c} = U[i*m+j, c] W[k*n+i, c] depends on i, j, k, c.
    # For a fixed j, k, c, the variable is V[j*p+k, c].
    # It appears in equations for all i.
    # So for fixed j, k, c, we have a system for i?
    # sum_c ...
    # Actually, for fixed j, k, the equations are for different i.
    # Let $X_{j,k,c} = V[j*p+k, c]$.
    # For each i, $j, k$:
    # $\sum_c U[i*m+j, c] W[k*n+i, c] X_{j,k,c} = 1$.
    # This is a linear system for $X_{j,k,c}$ if we fix U and W.
    # The system is:
    # For each (j,k), we have a vector of $X$'s (length r).
    # We have n equations (for i=0..n-1).
    # So we have a system of n equations with r unknowns for each (j,k).
    # If r >= n, we might have a solution.
    # Here n=2, r=6. So 2 equations, 6 unknowns. Underdetermined.
    # So we can find a solution for V for any given U, W!
    
    # So the algorithm is:
    # 1. Randomly pick U and W with small integer entries.
    # 2. For each (j,k), solve the linear system for V[j*p+k, :].
    # 3. Check if the solution is "nice" (integer or half-integer).
    # 4. If yes, we have a valid decomposition.
    
    for attempt in range(1000):
        if time.time() > deadline:
            return False
        
        U = np.array([[random.randint(-2, 2) for _ in range(r)] for _ in range(n*m)], dtype=float)
        W = np.array([[random.randint(-2, 2) for _ in range(r)] for _ in range(p*n)], dtype=float)
        
        V_sol = np.zeros((m*p, r), dtype=float)
        valid = True
        
        for j in range(m):
            for k in range(p):
                # Build linear system for V[j*p+k, c]
                # Eq for each i: sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c] = 1
                A_sys = np.zeros((n, r), dtype=float)
                for i in range(n):
                    for c in range(r):
                        A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                
                # Solve A_sys @ V_col = ones(n)
                # Use least squares or pseudoinverse
                try:
                    V_col = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
                except:
                    valid = False
                    break
                    
                # Check if V_col has integer or half-integer entries
                # Multiply by 2 to check integers
                V_col_check = V_col * 2
                if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                    valid = False
                    break
                V_sol[j*p+k, :] = V_col
                
        if valid:
            # Verify the solution
            # Check all equations
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c]
                        if not np.isclose(val, 1, atol=1e-6):
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            
            if ok:
                _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                return True
        if not valid:
            continue
            
    return False
# EVOLVE-BLOCK-END
