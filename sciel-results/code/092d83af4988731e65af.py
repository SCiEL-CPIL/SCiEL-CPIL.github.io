import numpy as np
import time
import sys
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 3, 4
    # Dimensions of U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor non-zero coordinates and values
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k
    # Indices:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    coords = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                coords.append((u, v, w, 1.0))
                
    # Total possible coordinates
    all_coords = [(u, v, w) for u in range(dim_u) for v in range(dim_v) for w in range(dim_w)]
    nonzero_set = set((u, v, w) for u, v, w, val in coords)
    
    def evaluate(U, V, W):
        """Compute error for given matrices."""
        error = 0.0
        # Check non-zero entries
        for u, v, w, target in coords:
            val = np.dot(U[u], np.multiply(V[v], W[w]))
            error += (val - target) ** 2
        # Check zero entries (sample if too many, but here it's small)
        # dim_u=6, dim_v=12, dim_w=8 -> 576 total. 24 non-zero. 552 zero.
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    if (u, v, w) not in nonzero_set:
                        val = np.dot(U[u], np.multiply(V[v], W[w]))
                        error += val ** 2
        return error

    def objective(x):
        """Objective function for optimizer."""
        r = len(x) // (dim_u + dim_v + dim_w) # This is wrong, x is flat
        # We need to know r. We pass r via closure or partial.
        # Better to create a lambda inside the loop.
        pass

    best_result = None
    best_rank = n * m * p
    
    # Schoolbook initialization
    U_school = np.zeros((dim_u, best_rank))
    V_school = np.zeros((dim_v, best_rank))
    W_school = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    # Function to try a specific rank R
    def try_rank(R, time_limit=5):
        # Initialize with random values or schoolbook truncated
        # Random is often better for avoiding local minima in rank reduction
        np.random.seed(42) # Fixed seed for reproducibility in debugging, but vary for search?
        
        U0 = np.random.randn(dim_u, R) * 0.1
        V0 = np.random.randn(dim_v, R) * 0.1
        W0 = np.random.randn(dim_w, R) * 0.1
        
        # Alternative init: Schoolbook with one column zeroed?
        # Let's try a mix.
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def obj(x):
            U = x[:dim_u*R].reshape(dim_u, R)
            V = x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
            W = x[dim_u*R+dim_v*R:].reshape(dim_w, R)
            return evaluate(U, V, W)
        
        try:
            # Use BFGS for small problems
            res = minimize(obj, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-5})
            if res.success or res.fun < 1e-6:
                U_opt = res.x[:dim_u*R].reshape(dim_u, R)
                V_opt = res.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                W_opt = res.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                
                # Round to nearest 0.5
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify exactness
                if evaluate(U_int, V_int, W_int) < 1e-9:
                    return True, U_int, V_int, W_int
        except Exception as e:
            pass
        return False, None, None, None

    start_time = time.time()
    
    # Try ranks from 23 down to 20
    for R in range(23, 19, -1):
        if time.time() - start_time > 100:
            break
            
        # Try a few random restarts
        found = False
        for attempt in range(10):
            if time.time() - start_time > 100:
                break
            
            # Randomize init
            np.random.seed(attempt + R)
            U0 = np.random.randn(dim_u, R) * 0.5
            V0 = np.random.randn(dim_v, R) * 0.5
            W0 = np.random.randn(dim_w, R) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def obj(x):
                U = x[:dim_u*R].reshape(dim_u, R)
                V = x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                W = x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                return evaluate(U, V, W)
            
            try:
                res = minimize(obj, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-4})
                if res.fun < 1e-4:
                    U_opt = res.x[:dim_u*R].reshape(dim_u, R)
                    V_opt = res.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                    W_opt = res.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                    
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    
                    if evaluate(U_int, V_int, W_int) < 1e-9:
                        # Convert to list of lists for return
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
            except Exception:
                continue

    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
