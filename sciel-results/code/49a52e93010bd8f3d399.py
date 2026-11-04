import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave 5s margin
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    num_eqs = n * m * p # 84
    
    # Pre-compute index mappings for vectorization
    # e = i * (m * p) + j * p + k
    # idx_U[e] = i * m + j
    # idx_V[e] = j * p + k
    # idx_W_row[e] = k * n + i
    
    idx_U = np.zeros(num_eqs, dtype=int)
    idx_V = np.zeros(num_eqs, dtype=int)
    idx_W_row = np.zeros(num_eqs, dtype=int)
    
    e = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U[e] = i * m + j
                idx_V[e] = j * p + k
                idx_W_row[e] = k * n + i
                e += 1
                
    # Pre-compute column indices for scattering coefficients into A
    # cols[e, c] = idx_W_row[e] * r + c
    # This depends on r, so we compute it inside the loop or pre-compute for max r?
    # r varies, so we compute inside.
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for better rank
    # Theoretical lower bound is max(dim_U, dim_V, dim_W) = 28.
    # Schoolbook is 84.
    # We try ranks from 28 upwards.
    
    # To increase success rate, use a wider range of integers for U and V.
    # Random integers in [-3, 3] excluding 0? Or just [-3, 3].
    # Using larger integers reduces the chance of accidental linear dependencies.
    
    for r in range(28, 84):
        if time.time() > deadline - 5:
            break
            
        # Number of tries: higher ranks are easier, so fewer tries needed?
        # But lower ranks are the goal.
        # Let's allocate more tries to lower ranks.
        tries = 100 if r < 40 else 20
        
        for _ in range(tries):
            if time.time() > deadline - 2:
                break
                
            # Random U and V
            # Using int values to ensure exactness in multiplication
            U = np.random.randint(-3, 4, size=(dim_U, r))
            V = np.random.randint(-3, 4, size=(dim_V, r))
            
            # Build A vectorized
            # U_exp shape (84, r)
            U_exp = U[idx_U, :]
            # V_exp shape (84, r)
            V_exp = V[idx_V, :]
            
            Coeffs = U_exp * V_exp # Shape (84, r)
            
            A = np.zeros((num_eqs, dim_W * r))
            # cols shape (84, r)
            cols = idx_W_row[:, None] * r + np.arange(r)[None, :]
            A[np.arange(num_eqs)[:, None], cols] = Coeffs
            
            b = np.ones(num_eqs)
            
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue
                
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check half-integer property
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-5:
                W_int = W_rounded / 2.0
                
                # Verify
                is_valid = True
                # Vectorized verification
                # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We can reuse Coeffs logic?
                # Coeffs was U_exp * V_exp.
                # We need sum_c Coeffs[e, c] * W_flat[e_block, c] ?
                # No, W_flat is indexed by (k*n+i, c).
                # For equation e, the variable block is idx_W_row[e].
                # So we need to gather W_flat for the specific rows.
                
                W_exp = W_flat.reshape(dim_W, r)[idx_W_row, :] # Shape (84, r)
                # The product for each equation is sum_c Coeffs[e, c] * W_exp[e, c]
                # Note: Coeffs was calculated from U and V.
                # So val = np.sum(Coeffs * W_exp, axis=1)
                
                vals = np.sum(Coeffs * W_exp, axis=1)
                
                if np.all(np.abs(vals - 1.0) < 1e-5):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
                    
    return get_schoolbook()
