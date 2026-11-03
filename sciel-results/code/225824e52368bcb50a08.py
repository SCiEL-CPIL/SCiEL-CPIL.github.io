"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _construct():
    """Find a low-rank exact tensor decomposition for 4x4 @ 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    deadline = time.time() + 130  # Leave 20s safety margin
    
    def _schoolbook():
        """Return schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = _schoolbook()
    
    def _is_valid(U, V, W, rank):
        """Check if the decomposition is exact."""
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m + j, :]
                    v_val = V_mat[j*p + k, :]
                    w_val = W_mat[k*n + i, :]
                    total = np.dot(np.dot(u_val, v_val), w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower-rank decompositions by fixing U and V and solving for W
    for target_rank in range(60, 144):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        found = False
        for seed in range(100):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            np.random.seed(seed)
            
            # Fix U and V with random values
            U_fixed = np.random.uniform(-1, 1, (nm, target_rank))
            V_fixed = np.random.uniform(-1, 1, (mp, target_rank))
            
            # Set up linear system for W
            num_equations = n * m * p  # 144
            num_unknowns = pn * target_rank
            
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            A[eq_idx, (k*n+i)*target_rank + c] = U_fixed[i*m+j, c] * V_fixed[j*p+k, c]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol = W_flat.reshape(pn, target_rank)
                
                # Check validity
                if _is_valid(U_fixed, V_fixed, W_sol, target_rank):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_fixed, V_fixed, W_sol
                        found = True
                        break
            except np.linalg.LinAlgError:
                continue
        
        if found:
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 @ 4x9 matrix multiplication."""
    result = _construct()
    return result
# EVOLVE-BLOCK-END

