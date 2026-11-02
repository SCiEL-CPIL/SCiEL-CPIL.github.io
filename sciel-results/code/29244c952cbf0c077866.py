"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 18
    
    # Build the list of (a,b,c) triples with T=1
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    one_set = set(ones)
    num_triples = len(all_triples)
    
    # Precompute indices for fast access
    a_indices = np.array([t[0] for t in all_triples])
    b_indices = np.array([t[1] for t in all_triples])
    c_indices = np.array([t[2] for t in all_triples])
    targets = np.array([1.0 if t in one_set else 0.0 for t in all_triples])

    def unpack(params, r):
        """Unpack flattened parameters into U, V, W matrices."""
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:]
        
        U = U_flat.reshape((n*m, r))
        V = V_flat.reshape((m*p, r))
        W = W_flat.reshape((p*n, r))
        return U, V, W

    def residuals(params, r):
        """Compute residuals for the tensor decomposition."""
        U, V, W = unpack(params, r)
        
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all triples
        # Using broadcasting
        # U[a, :] shape (num_triples, r)
        U_sel = U[a_indices, :]
        V_sel = V[b_indices, :]
        W_sel = W[c_indices, :]
        
        # Element-wise product and sum over r
        prod = U_sel * V_sel * W_sel
        result = np.sum(prod, axis=1)
        
        return result - targets

    def check_exactness(U, V, W, r):
        """Check if the decomposition is exact with integer/half-integer entries."""
        # Check if entries are close to integers or half-integers
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Check integers
        int_rounded = np.round(all_vals)
        if np.allclose(all_vals, int_rounded, atol=1e-5):
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            # Verify exactness with integers
            U_sel = U_int[a_indices, :]
            V_sel = V_int[b_indices, :]
            W_sel = W_int[c_indices, :]
            result = np.sum(U_sel * V_sel * W_sel, axis=1)
            if np.allclose(result, targets, atol=1e-5):
                return True, U_int, V_int, W_int
        
        # Check half-integers (multiply by 2, round, divide by 2)
        half_rounded = np.round(all_vals * 2) / 2
        if np.allclose(all_vals, half_rounded, atol=1e-5):
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            # Verify exactness with half-integers
            U_sel = U_half[a_indices, :]
            V_sel = V_half[b_indices, :]
            W_sel = W_half[c_indices, :]
            result = np.sum(U_sel * V_sel * W_sel, axis=1)
            if np.allclose(result, targets, atol=1e-5):
                return True, U_half, V_half, W_half
                
        return False, None, None, None

    # Try ranks from 9 up to 17
    for r in range(9, 18):
        if time.time() > deadline:
            break
        
        # Total variables
        total_vars = (n*m + m*p + p*n) * r
        
        # Try a few random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + r + attempt)
            params0 = np.random.randn(total_vars)
            
            try:
                # Use least_squares to minimize residuals
                result = least_squares(
                    residuals, 
                    params0, 
                    args=(r,), 
                    method='lm', 
                    max_nfev=1000
                )
                
                if time.time() > deadline:
                    break
                    
                # Check if we found a good solution
                if result.success or result.cost < 1e-8:
                    U, V, W = unpack(result.x, r)
                    is_exact, U_exact, V_exact, W_exact = check_exactness(U, V, W, r)
                    
                    if is_exact:
                        # Convert to lists for output
                        best_result = {
                            "U": U_exact.tolist(),
                            "V": V_exact.tolist(),
                            "W": W_exact.tolist()
                        }
                        best_rank = r
                        found = True
                        break # Break out of attempt loop
                    
            except Exception as e:
                # If optimization fails, continue to next attempt
                continue
        
        if found:
            # If we found a solution for this rank, we can stop if it's the minimum possible
            # But we should check lower ranks first? We are iterating 9 to 17.
            # So if we find one at 9, we return it.
            break

    return best_result
# EVOLVE-BLOCK-END
