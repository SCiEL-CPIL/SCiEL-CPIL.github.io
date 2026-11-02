import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # But we work with the factorization U, V, W
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    # Indices
    # U[i, j, r] where i in [0, n), j in [0, m)
    # V[j, k, r] where j in [0, m), k in [0, p)
    # W[k, i, r] where k in [0, p), i in [0, n)
    
    # The tensor entry for (i,j), (j,k), (k,i) is 1.
    # All other entries are 0.
    
    # We want to minimize || T - sum_r U_r V_r W_r ||_F^2
    
    def residual(params, r):
        # params is a flat vector of length r * (n*m + m*p + p*n)
        # Reshape into U, V, W
        U = params[:r*n*m].reshape(r, n*m).T
        V = params[r*n*m : r*n*m + r*m*p].reshape(r, m*p).T
        W = params[r*n*m + r*m*p :].reshape(r, p*n).T
        
        # Compute the tensor T_approx
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        
        # We can compute this efficiently using einsum or loops
        # Since dimensions are small, loops are fine.
        T_approx = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if indices are compatible?
                    # The tensor is sparse. Only T[i*m+j, j*p+k, k*n+i] is non-zero.
                    # We can just compute the full tensor and compare to target.
                    pass
        
        # Better: compute the full tensor and subtract target.
        # But target is mostly zeros.
        # Let's compute the sum for all (a,b,c).
        # T_approx[a,b,c] = np.einsum('ar,br,cr->abc', U, V, W)
        
        # To save time, we only need to check the non-zero entries of the target?
        # No, because if the decomposition is not exact, the "zero" entries of the target
        # might be non-zero in the approximation. So we must check all entries.
        
        # However, the target tensor has only n*m*p non-zero entries.
        # The other (n*m * m*p * p*n - n*m*p) entries must be 0.
        # So we need to minimize the sum of squares over ALL entries.
        
        # Let's compute the full tensor.
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        diff = T_approx - T_target
        return np.sum(diff * diff)

    def get_matrices(params, r):
        U = params[:r*n*m].reshape(r, n*m).T
        V = params[r*n*m : r*n*m + r*m*p].reshape(r, m*p).T
        W = params[r*n*m + r*m*p :].reshape(r, p*n).T
        return U, V, W

    # Try ranks from target_rank - 1 down to some lower bound
    # Lower bound for 2x2x5 is at least 2*5 = 10? 
    # Actually, rank >= n(m+p-1) is a lower bound for border rank?
    # For 2x2x2, rank 7. 2(2+2-1) = 6. So 7 > 6.
    # For 2x2x5, let's try down to 15.
    
    best_result = None
    start_time = time.time()
    
    for r in range(target_rank - 1, 14, -1):
        if time.time() - start_time > 100: # Leave 20s for rounding and final checks
            break
            
        # Random initialization
        np.random.seed(42) # Fixed seed for reproducibility? 
        # Maybe multiple restarts?
        
        found = False
        for attempt in range(5):
            if time.time() - start_time > 100:
                break
                
            # Initialize with small random values
            params = np.random.randn(r * (n*m + m*p + p*n)) * 0.1
            
            # Optimize
            res = minimize(residual, params, args=(r,), method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000})
            
            if res.fun < 1e-10:
                # Check if we can round to integers/half-integers
                U, V, W = get_matrices(res.x, r)
                
                # Check if entries are close to k/2
                def is_rational_half(arr):
                    # Multiply by 2 and check if close to integer
                    scaled = arr * 2
                    rounded = np.round(scaled)
                    if np.allclose(scaled, rounded, atol=1e-6):
                        return rounded / 2.0
                    return None
                
                U_rounded = is_rational_half(U)
                V_rounded = is_rational_half(V)
                W_rounded = is_rational_half(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    # Verify exactness with rounded matrices
                    U_int = U_rounded
                    V_int = V_rounded
                    W_int = W_rounded
                    
                    # Reconstruct
                    T_approx = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    T_target = np.zeros((n*m, m*p, p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                T_target[a, b, c] = 1.0
                                
                    if np.allclose(T_approx, T_target, atol=1e-9):
                        # Convert to list of lists for output
                        # The interface expects U, V, W as lists of lists?
                        # "factor matrices of shapes ... with integer/half-integer entries"
                        # The seed program returns lists of lists.
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        found = True
                        break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        rank = target_rank
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
# EVOLVE-BLOCK-END
