import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Pre-allocate schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = None, None, None
    
    start_time = time.time()
    deadline = 100.0 # 100 seconds limit to be safe
    
    # Try ranks from lower bound (49) up to schoolbook
    # Lower bound for 3,5,6 is ceil((3+5+6)^2/4) = 49
    min_rank = 49
    
    for r in range(min_rank, best_rank):
        if time.time() - start_time > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() - start_time > deadline:
                break
                
            # Random initialization with small integers
            # Using -1, 0, 1 to encourage sparse/structured solutions
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            W = np.random.randint(-1, 2, (p * n, r))
            
            # Alternating Least Squares (ALS) style refinement
            # We solve for W given U, V, then U given V, W, etc.
            # The system is linear in each factor when the others are fixed.
            
            success = False
            for step in range(50):
                if time.time() - start_time > deadline:
                    break
                
                # --- Solve for W ---
                # Equations: sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c] = delta_{j1, j2}
                # Variables: W[k*n+i, c]
                
                # Number of equations: n * p * m * m
                num_eq = n * p * m * m
                num_vars = p * n * r
                
                # Build design matrix A_w (num_eq x num_vars)
                # This is the bottleneck. We can optimize if needed, but for now:
                A_w = np.zeros((num_eq, num_vars))
                b_w = np.zeros(num_eq)
                
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b_w[idx] = 1.0 if j1 == j2 else 0.0
                                
                                # Coefficients for W[k*n+i, :]
                                # coeff_c = U[i*m+j1, c] * V[j2*p+k, c]
                                u_row = U[i * m + j1, :]
                                v_row = V[j2 * p + k, :]
                                coeffs = u_row * v_row
                                
                                w_start = (k * n + i) * r
                                A_w[idx, w_start:w_start + r] += coeffs
                                idx += 1
                                
                # Solve
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(p * n, r)
                
                # --- Solve for U ---
                # Equations: sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c] = delta_{j1, j2}
                # Variables: U[i*m+j1, c]
                
                A_u = np.zeros((num_eq, n * m * r))
                b_u = np.zeros(num_eq)
                
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b_u[idx] = 1.0 if j1 == j2 else 0.0
                                
                                # Coefficients for U[i*m+j1, :]
                                # coeff_c = V[j2*p+k, c] * W[k*n+i, c]
                                v_row = V[j2 * p + k, :]
                                w_row = W[k * n + i, :]
                                coeffs = v_row * w_row
                                
                                u_start = (i * m + j1) * r
                                A_u[idx, u_start:u_start + r] += coeffs
                                idx += 1
                                
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_flat.reshape(n * m, r)
                
                # --- Solve for V ---
                # Equations: sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c] = delta_{j1, j2}
                # Variables: V[j2*p+k, c]
                
                A_v = np.zeros((num_eq, m * p * r))
                b_v = np.zeros(num_eq)
                
                idx = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                b_v[idx] = 1.0 if j1 == j2 else 0.0
                                
                                # Coefficients for V[j2*p+k, :]
                                # coeff_c = U[i*m+j1, c] * W[k*n+i, c]
                                u_row = U[i * m + j1, :]
                                w_row = W[k * n + i, :]
                                coeffs = u_row * w_row
                                
                                v_start = (j2 * p + k) * r
                                A_v[idx, v_start:v_start + r] += coeffs
                                idx += 1
                                
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(m * p, r)
                
                # Check error
                # If error is small, snap and verify
                # We can check error periodically or at the end of steps
                if step % 5 == 0:
                    err = 0.0
                    # Sample some equations to check error quickly
                    # Or compute full error
                    # Full error is O(n*p*m*m*r)
                    for i in range(n):
                        for k in range(p):
                            for j1 in range(m):
                                for j2 in range(m):
                                    target = 1.0 if j1 == j2 else 0.0
                                    val = np.dot(np.dot(U[i * m + j1, :], V[j2 * p + k, :]), W[k * n + i, :])
                                    err += (val - target) ** 2
                                    
                    if err < 1e-6:
                        # Snap to half-integers
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                        
                        # Exact verification
                        if _verify(U, V, W, n, m, p):
                            success = True
                            best_rank = r
                            best_U, best_V, best_W = U, V, W
                            break
                            
            if success:
                break

    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school,
            "V": V_school,
            "W": W_school
        }

def _verify(U, V, W, n, m, p):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    # Compute sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c]
                    # This is dot( U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :] )
                    val = np.dot(np.dot(U[i * m + j1, :], V[j2 * p + k, :]), W[k * n + i, :])
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(val - target) > 1e-9:
                        return False
    return True
# EVOLVE-BLOCK-END
