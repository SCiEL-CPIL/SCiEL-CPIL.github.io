import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if this term contributes to C[i, k]
    # C[i, k] = sum_j A[i, j] B[j, k]
    # T[i, j, k] is 1 for all i, j, k in the standard basis.
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p
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
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to find a lower rank
    # Heuristic lower bound is roughly n*m*p / 2 for small matrices, but Strassen suggests better.
    # Let's search from 30 to 55.
    for r in range(30, 56):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with small random values
            # Shapes: U(n*m, r), V(m*p, r), W(p*n, r)
            U_init = np.random.rand(n * m, r) * 0.1
            V_init = np.random.rand(m * p, r) * 0.1
            W_init = np.random.rand(p * n, r) * 0.1
            
            # Reshape to tensors for easier calculation
            # U_tensor[i, j, c]
            U_t = U_init.reshape(n, m, r)
            V_t = V_init.reshape(m, p, r)
            W_t = W_init.reshape(p, n, r)
            
            def objective(params):
                U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r : n*m*r + m*p*r], params[n*m*r + m*p*r:]
                U_t = U_flat.reshape(n, m, r)
                V_t = V_flat.reshape(m, p, r)
                W_t = W_flat.reshape(p, n, r)
                
                # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # This is a contraction.
                # Using einsum: 'ijc,jkc,kic->ijk'
                prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
                error = np.sum((prod - T)**2)
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead as it doesn't require gradients and is robust for small problems
            # L-BFGS-B is faster but can get stuck in local minima more easily without good gradients
            # Given small size, Nelder-Mead is okay, but might be slow.
            # Let's use BFGS or L-BFGS-B with numerical gradients.
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success and res.fun < 1e-6:
                    # Found a solution, now snap to integers/half-integers
                    U_sol = res.x[:n*m*r].reshape(n, m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                    W_sol = res.x[n*m*r + m*p*r:].reshape(p, n, r)
                    
                    # Snap to nearest half-integer
                    # Multiply by 2, round, divide by 2
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    prod_snap = np.einsum('ijc,jkc,kic->ijk', U_snap, V_snap, W_snap, optimize=True)
                    if np.allclose(prod_snap, T, atol=1e-9):
                        # Success!
                        # Convert to lists of lists
                        U_out = U_snap.reshape(n*m, r).tolist()
                        V_out = V_snap.reshape(m*p, r).tolist()
                        W_out = W_snap.reshape(p*n, r).tolist()
                        
                        # Ensure types are float/int as required (half-integers are floats)
                        # The interface accepts integer/half-integer entries.
                        return {"U": U_out, "V": V_out, "W": W_out}
            except Exception:
                pass

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
