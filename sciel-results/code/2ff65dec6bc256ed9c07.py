import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 6
    
    # Construction code (same as before)
    # ...
    
    # After U, V, W are built (77 cols)
    
    # Attempt rank reduction via linear dependencies
    # Flatten terms
    terms = []
    for r in range(77):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v[j*p+k]
                    if v_val == 0: continue
                    w_val = w[k*n+i]
                    if w_val == 0: continue
                    vec[i*m*p + j*p + k] += u_val * v_val * w_val
        terms.append(vec)
        
    M = np.array(terms).T
    
    # Compute rank
    try:
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        tol = 1e-10
        rank = np.sum(s > tol)
    except:
        rank = 77
        
    if rank < 77:
        # Try to find a subset of 'rank' columns that spans the target
        # Target is ones(96)
        # We can use the first 'rank' columns
        M_sub = M[:, :rank]
        try:
            c = np.linalg.lstsq(M_sub, np.ones(n*m*p), rcond=None)[0]
            # Verify
            T_check = M_sub @ c
            if np.allclose(T_check, np.ones(n*m*p), atol=1e-9):
                # Scale to integers
                # c are floats. We need U, V, W to have integer/half-integer entries.
                # We can multiply U[:, :rank] by c
                U_new = U[:, :rank] * c
                V_new = V[:, :rank]
                W_new = W[:, :rank]
                # Check if entries are close to integers or half-integers
                # If not, fallback
                if np.allclose(U_new, np.round(U_new * 2) / 2, atol=1e-9) and \
                   np.allclose(V_new, np.round(V_new * 2) / 2, atol=1e-9) and \
                   np.allclose(W_new, np.round(W_new * 2) / 2, atol=1e-9):
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        except:
            pass
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
