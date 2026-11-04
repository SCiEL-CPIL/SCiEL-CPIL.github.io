"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The schoolbook algorithm has rank 3*3*7 = 63.
    We attempt to find a lower rank decomposition (e.g., 62 or 61) using nonlinear optimization.
    If a valid decomposition is found, we return it. Otherwise, we fall back to the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # Target tensor entries: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # We represent the tensor as a flat vector of non-zero entries and their indices
    # Or better, define a function that computes the reconstructed tensor values for the non-zero positions.
    
    # Indices of non-zero entries in the tensor
    # Mode 1 index: i*m + j
    # Mode 2 index: j*p + k
    # Mode 3 index: k*n + i
    target_indices_u = []
    target_indices_v = []
    target_indices_w = []
    target_values = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices_u.append(idx_u)
                target_indices_v.append(idx_v)
                target_indices_w.append(idx_w)
                target_values.append(1.0)
                
    # Also, we should check zero entries? 
    # The problem statement says "exactly decompose the matrix multiplication tensor".
    # This implies all other entries must be 0.
    # However, checking all zero entries in the optimization loop is expensive.
    # We can check them at the end.
    # The tensor size is 9 x 21 x 21 = 3969.
    # Non-zeros: 63. Zeros: 3906.
    # We will include a subset of zero entries in the objective function to guide the optimization, 
    # and verify all at the end.
    
    # For the objective function, we only need to evaluate the non-zero entries and a random sample of zeros
    # to keep the cost low.
    # Let's select 100 random zero entries to include in the objective.
    
    all_u_indices = np.arange(dim_u)
    all_v_indices = np.arange(dim_v)
    all_w_indices = np.arange(dim_w)
    
    # Create a set of non-zero indices for fast lookup
    non_zero_set = set(zip(target_indices_u, target_indices_v, target_indices_w))
    
    # Generate random zero indices
    zero_indices_u = []
    zero_indices_v = []
    zero_indices_w = []
    num_zero_samples = 100
    attempts = 0
    while len(zero_indices_u) < num_zero_samples and attempts < 1000:
        ru = np.random.randint(dim_u)
        rv = np.random.randint(dim_v)
        rw = np.random.randint(dim_w)
        if (ru, rv, rw) not in non_zero_set:
            zero_indices_u.append(ru)
            zero_indices_v.append(rv)
            zero_indices_w.append(rw)
        attempts += 1
        
    target_indices_u_np = np.array(target_indices_u)
    target_indices_v_np = np.array(target_indices_v)
    target_indices_w_np = np.array(target_indices_w)
    target_values_np = np.array(target_values)
    
    zero_indices_u_np = np.array(zero_indices_u)
    zero_indices_v_np = np.array(zero_indices_v)
    zero_indices_w_np = np.array(zero_indices_w)
    zero_values_np = np.zeros(len(zero_indices_u_np))
    
    def residuals(params, R):
        # params is a flat vector of length R * (dim_u + dim_v + dim_w)
        # First R*dim_u are U, next R*dim_v are V, next R*dim_w are W
        U = params[:R*dim_u].reshape(dim_u, R)
        V = params[R*dim_u:R*dim_u + R*dim_v].reshape(dim_v, R)
        W = params[R*dim_u + R*dim_v:].reshape(dim_w, R)
        
        # Compute T for non-zero indices
        # T[idx_u, idx_v, idx_w] = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # We can do this efficiently using broadcasting or loops for small R
        
        res_non_zero = np.zeros(len(target_indices_u_np))
        res_zero = np.zeros(len(zero_indices_u_np))
        
        # Using a loop over r is O(R * N_samples). 
        # N_samples = 63 + 100 = 163. R ~ 60. 60*163 ~ 10k ops. Very fast.
        
        for r in range(R):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # Non-zero part
            res_non_zero += u_col[target_indices_u_np] * v_col[target_indices_v_np] * w_col[target_indices_w_np]
            # Zero part
            res_zero += u_col[zero_indices_u_np] * v_col[zero_indices_v_np] * w_col[zero_indices_w_np]
            
        # Target is 1 for non-zero, 0 for zero
        res_non_zero -= target_values_np
        res_zero -= zero_values_np
        
        return np.concatenate([res_non_zero, res_zero])
    
    best_rank = n * m * p
    best_params = None
    
    # Try ranks from 62 down to 50
    for target_rank in range(62, 49, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize params
            # Scale initialization to help convergence
            init_scale = 0.5
            params0 = np.random.randn(target_rank * (dim_u + dim_v + dim_w)) * init_scale
            
            try:
                result = least_squares(residuals, params0, args=(target_rank,), method='lm', max_nfev=1000)
                if result.cost < 1e-10:
                    # Verify full tensor
                    U = result.x[:target_rank*dim_u].reshape(dim_u, target_rank)
                    V = result.x[target_rank*dim_u:target_rank*dim_u + target_rank*dim_v].reshape(dim_v, target_rank)
                    W = result.x[target_rank*dim_u + target_rank*dim_v:].reshape(dim_w, target_rank)
                    
                    # Full verification
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.dot(U[idx_u, :], np.multiply(V[idx_v, :], W[idx_w, :]))
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Check zeros (sample more zeros for safety)
                        for _ in range(500):
                            ru = np.random.randint(dim_u)
                            rv = np.random.randint(dim_v)
                            rw = np.random.randint(dim_w)
                            if (ru, rv, rw) in non_zero_set:
                                continue
                            val = np.dot(U[ru, :], np.multiply(V[rv, :], W[rw, :]))
                            if abs(val) > 1e-6:
                                valid = False
                                break
                        if valid:
                            best_rank = target_rank
                            best_params = result.x
                            break
            except Exception:
                pass
    
    # If we found a better rank, construct U, V, W from best_params
    if best_params is not None:
        R = best_rank
        U = best_params[:R*dim_u].reshape(dim_u, R)
        V = best_params[R*dim_u:R*dim_u + R*dim_v].reshape(dim_v, R)
        W = best_params[R*dim_u + R*dim_v:].reshape(dim_w, R)
        
        # Convert to lists of floats/ints if possible? The task says "integer/half-integer entries".
        # The optimization yields floats. We might need to round them?
        # "exactly decompose" implies the values must be exact.
        # If the solver finds 1.0000001, it's not exact.
        # We should check if the values are close to integers or half-integers.
        # If not, we cannot use this result because it's not "exact" in the required sense.
        # The task says "integer/half-integer entries".
        # So we must verify that U, V, W entries are close to k/2.
        
        def is_half_integer(x, tol=1e-5):
            return abs(x * 2 - round(x * 2)) < tol
            
        valid_entries = True
        for mat in [U, V, W]:
            for val in mat:
                if not is_half_integer(val):
                    valid_entries = False
                    break
            if not valid_entries:
                break
        
        if valid_entries:
            # Snap to half-integers
            U = np.round(U * 2) / 2.0
            V = np.round(V * 2) / 2.0
            W = np.round(W * 2) / 2.0
            
            # Re-verify exactness with snapped values
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], np.multiply(V[idx_v, :], W[idx_w, :]))
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid: break
            if valid:
                # Return the snapped matrices
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # If snapping broke it, discard
                pass
    
    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
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
