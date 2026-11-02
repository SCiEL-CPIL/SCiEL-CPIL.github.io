import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of the factor matrices
    dim_U = n * m  # 15
    dim_V = m * p  # 30
    dim_W = p * n  # 18
    
    # Target tensor T
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i, else 0
    T_full = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    # Function to evaluate residual
    def residual(params):
        # params is a flat vector of length r * (dim_U + dim_V + dim_W)
        r = len(params) // (dim_U + dim_V + dim_W)
        if r == 0: return []
        
        U_flat = params[:r * dim_U]
        V_flat = params[r * dim_U : r * dim_U + r * dim_V]
        W_flat = params[r * dim_U + r * dim_V :]
        
        U = U_flat.reshape((dim_U, r))
        V = V_flat.reshape((dim_V, r))
        W = W_flat.reshape((dim_W, r))
        
        # Compute contraction
        # R[a, b, c] = sum_alpha U[a, alpha] * V[b, alpha] * W[c, alpha]
        R = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual is R - T_full
        return (R - T_full).flatten()

    # Search for minimal rank
    # Lower bound heuristic: max(dim_U, dim_V, dim_W) is not a strict lower bound for rank,
    # but rank must be at least max(n, m, p)? No.
    # Let's start from a reasonable guess.
    # For 2x2x2, rank 7. dim 4.
    # For 3x3x3, rank 23. dim 9.
    # For 3x5x6, let's try from 20 up.
    
    start_time = time.time()
    max_time = 100.0 # Leave 20s for safety
    
    # Try ranks from low to high
    # We know 90 works.
    # Let's try to find something better.
    
    best_result = None
    
    # We will try ranks in a specific order or range.
    # Given the complexity, we might not find the absolute minimum, but a good one.
    # Let's try a range of ranks.
    
    # To save time, we can limit the number of restarts per rank.
    n_restarts = 5
    
    for r in range(20, 90):
        if time.time() - start_time > max_time:
            break
            
        # Check if we have enough time for this rank
        # If not, break
        
        found = False
        for restart in range(n_restarts):
            if time.time() - start_time > max_time - 5:
                break
                
            # Random initialization
            # Scale: entries should be around 1/sqrt(r) or similar?
            # Let's use random normal, scaled.
            init_scale = 1.0 / np.sqrt(r)
            params = np.random.randn(r * (dim_U + dim_V + dim_W)) * init_scale
            
            try:
                sol = least_squares(residual, params, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
                
                if sol.success or sol.cost < 1e-8:
                    # Check exactness with integer/half-integer snapping
                    # Extract U, V, W
                    U_flat = sol.x[:r * dim_U]
                    V_flat = sol.x[r * dim_U : r * dim_U + r * dim_V]
                    W_flat = sol.x[r * dim_U + r * dim_V :]
                    
                    U = U_flat.reshape((dim_U, r))
                    V = V_flat.reshape((dim_V, r))
                    W = W_flat.reshape((dim_W, r))
                    
                    # Snap to half-integers
                    # Multiply by 2, round, divide by 2
                    U_int = np.round(U * 2)
                    V_int = np.round(V * 2)
                    W_int = np.round(W * 2)
                    
                    # Verify exactness
                    # R_int[a,b,c] = sum (U_int[a,alpha] * V_int[b,alpha] * W_int[c,alpha])
                    # This sum must be 8 for T=1, 0 for T=0.
                    # Actually, if U,V,W are half-integers (x/2),
                    # Product is x*y*z / 8.
                    # Sum is S/8.
                    # We need S/8 = 1 => S=8, or S/8=0 => S=0.
                    
                    U_h = U_int
                    V_h = V_int
                    W_h = W_int
                    
                    R_int = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
                    
                    # Check if R_int matches 8*T_full_int
                    T_int = (T_full * 8).astype(int)
                    
                    if np.allclose(R_int, T_int, atol=0):
                        # Found a valid decomposition
                        # Convert back to float (half-integers)
                        U_out = (U_int / 2.0).tolist()
                        V_out = (V_int / 2.0).tolist()
                        W_out = (W_int / 2.0).tolist()
                        
                        return {"U": U_out, "V": V_out, "W": W_out}
                        
            except Exception:
                pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
