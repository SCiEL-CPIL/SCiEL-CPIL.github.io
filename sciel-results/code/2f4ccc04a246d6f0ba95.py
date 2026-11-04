"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import scipy.linalg

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.flatten()
    num_eqs = len(T_flat)
    
    # Try ranks from a lower estimate up to schoolbook
    # We aim for < 36.
    # Let's try 25 to 35.
    for rank in range(25, 36):
        num_vars = rank * p * n
        
        for attempt in range(20):
            # Generate random U and V
            # Use a mix of 0, 0.5, 1
            vals = np.array([0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            
            # Construct A
            A = np.zeros((num_eqs, num_vars))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * (m * p) + (j * p + k) * (p * n) + (k * n + i)
                        u_val = U[i * m + j, :]
                        v_val = V[j * p + k, :]
                        coeff = u_val * v_val
                        base_var_idx = (k * n + i) * rank
                        var_indices = base_var_idx + np.arange(rank)
                        A[eq_idx, var_indices] = coeff
            
            try:
                x, residuals, rank_A, s = scipy.linalg.lstsq(A, T_flat)
            except Exception:
                continue
            
            if len(residuals) > 0 and residuals[0] < 1e-9:
                x2 = x * 2
                x2_rounded = np.round(x2)
                if np.allclose(x2, x2_rounded, atol=1e-6):
                    W_solution = (x2_rounded / 2.0).reshape(p * n, rank)
                    
                    # Quick verification
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for r in range(rank):
                                    val += U[i*m+j, r] * V[j*p+k, r] * W_solution[k*n+i, r]
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                    if not valid: break
                    
                    if valid:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_solution.tolist()
                        }
                        
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
