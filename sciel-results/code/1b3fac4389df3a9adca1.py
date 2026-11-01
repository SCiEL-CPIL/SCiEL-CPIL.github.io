"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin (150s limit, 10s buffer)
    
    # Schoolbook decomposition as fallback (rank 36)
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
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using integer arithmetic (entries must be half-integers)."""
        # Convert to integers by multiplying by 2
        U2 = np.array([[int(round(x * 2)) for x in row] for row in U], dtype=np.int64)
        V2 = np.array([[int(round(x * 2)) for x in row] for row in V], dtype=np.int64)
        W2 = np.array([[int(round(x * 2)) for x in row] for row in W], dtype=np.int64)
        rank = U2.shape[1]
        
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # In doubled form: sum_c (U2[i*m+j,c]/2) * (V2[j*p+k,c]/2) * (W2[k*n+i,c]/2) == 1
        # => sum_c U2[i*m+j,c] * V2[j*p+k,c] * W2[k*n+i,c] == 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                    if s != 8:
                        return False
        return True
    
    def check_half_integer(mat, tol=1e-9):
        """Check if all entries are half-integers."""
        doubled = np.array(mat) * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)
    
    # Known lower bounds and constructions:
    # For 3x3x4 matrix multiplication, the tensor rank is known to be at most 24
    # (via BCLR-like constructions). The schoolbook is 36.
    # We try to find a rank-24 decomposition using structured search.
    
    # Strategy: Use a randomized algebraic search combined with local refinement.
    # We parameterize the factors with half-integer entries in {-2,-1,0,1,2} / 2
    
    # Try to find a rank-24 decomposition
    # The system has 36 equations (one per (i,j,k)) and 33*24 = 792 unknowns
    # This is highly underdetermined, so we use a structured approach
    
    # First, try the known BCLR construction for 3x3x4
    # The BCLR algorithm gives rank 24 for 3x3x4
    
    # Explicit BCLR-style construction for 3x3x4:
    # We use 24 terms. Each term (u_c, v_c, w_c) is a vector.
    # The key insight is that we can use Kronecker products and structured matrices.
    
    # Let's try a direct construction based on the Strassen-like approach extended to 3x3x4
    
    # Construction attempt 1: Direct algebraic construction
    # For 3x3x4, one known rank-24 construction uses the following idea:
    # Split the 3x3 input into overlapping 2x2 blocks and use Strassen's algorithm
    
    # Let me try a numerical approach: use gradient descent with integer rounding
    
    def try_find_rank(r_target, max_restarts=20, seed=42):
        """Try to find a rank-r decomposition using randomized local search."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Number of variables
        num_u = n * m * r_target
        num_v = m * p * r_target
        num_w = p * n * r_target
        total_vars = num_u + num_v + num_w
        
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random half-integer values in {-2, -1, 0, 1, 2}
            # Scaled by 0.5
            init_vals = np.random.choice([-2, -1, 0, 1, 2], size=total_vars) * 0.5
            
            U_flat = init_vals[:num_u].copy()
            V_flat = init_vals[num_u:num_u+num_v].copy()
            W_flat = init_vals[num_u+num_v:].copy()
            
            # Local search: perturb and keep if error decreases
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                U = U_flat.reshape((n*m, r_target))
                V = V_flat.reshape((m*p, r_target))
                W = W_flat.reshape((p*n, r_target))
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r_target):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0)**2
                
                if err < best_error:
                    best_error = err
                    best_solution = (U_flat.copy(), V_flat.copy(), W_flat.copy())
                
                if err < 1e-8:
                    break
                
                # Perturb a random coordinate
                idx = np.random.randint(0, total_vars)
                if idx < num_u:
                    delta = np.random.choice([-1, 1]) * 0.5
                    new_val = U_flat[idx] + delta
                    # Ensure half-integer
                    new_val = round(new_val * 2) / 2.0
                    U_flat[idx] = new_val
                elif idx < num_u + num_v:
                    idx_v = idx - num_u
                    delta = np.random.choice([-1, 1]) * 0.5
                    new_val = V_flat[idx_v] + delta
                    new_val = round(new_val * 2) / 2.0
                    V_flat[idx_v] = new_val
                else:
                    idx_w = idx - num_u - num_v
                    delta = np.random.choice([-1, 1]) * 0.5
                    new_val = W_flat[idx_w] + delta
                    new_val = round(new_val * 2) / 2.0
                    W_flat[idx_w] = new_val
        
        if best_solution is not None:
            U_flat, V_flat, W_flat = best_solution
            U = U_flat.reshape((n*m, r_target))
            V = V_flat.reshape((m*p, r_target))
            W = W_flat.reshape((p*n, r_target))
            
            # Final check
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 24 upward (24 is the known lower bound for 3x3x4)
    for r in range(24, 37):
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds for robustness
        found = None
        for seed in [42, 123, 789, 1000, 2023]:
            if time.time() > deadline - 15:
                break
            found = try_find_rank(r, max_restarts=10, seed=seed)
            if found is not None:
                break
        
        if found is not None:
            # Verify one more time
            if verify_exact(found["U"], found["V"], found["W"]):
                return found
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
