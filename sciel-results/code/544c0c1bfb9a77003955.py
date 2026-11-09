import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Schoolbook rank
    R_schoolbook = n * m * p
    
    # Precompute the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i)
    # Note: The problem statement says W[k*n+i, r]. This is the index for (i,k) in the output?
    # Output C[i,k]. Index in W is k*n + i.
    # So gamma = k*n + i.
    # alpha = i*m + j.
    # beta = j*p + k.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def check_solution(U, V, W, R):
        # Compute tensor product sum
        # Reshape U, V, W to (dim, R)
        # We need sum_c U[a,c] V[b,c] W[g,c]
        # This is a tensor contraction.
        # We can do this efficiently using einsum or loops.
        # For small sizes, loops are fine.
        # But let's use numpy.
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We want a tensor of shape (nm, mp, pn)
        # Tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        # This might be slow if R is large, but R=29 is small.
        try:
            T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            # Check if T_calc matches T
            # We need exact match? "exactly decompose"
            # But we are finding real solutions first.
            # Check error
            err = np.linalg.norm(T_calc - T)
            return err
        except Exception:
            return float('inf')

    def try_rank(R):
        if time.time() > deadline:
            return None
            
        # Variables: flatten U, V, W
        # U: nm x R, V: mp x R, W: pn x R
        # Total variables: R * (nm + mp + pn)
        num_vars = R * (n*m + m*p + p*n)
        
        # Initial guess: random
        # Or based on schoolbook?
        # Schoolbook has R = nm*p.
        # If R < nm*p, we can't use schoolbook directly.
        # Let's use random initialization.
        x0 = np.random.randn(num_vars) * 0.1
        
        def residuals(x):
            if time.time() > deadline:
                return np.zeros(n*m*m*p*p*n) # Timeout
            U = x[:R*n*m].reshape(R, n*m).T
            V = x[R*n*m : R*n*m + R*m*p].reshape(R, m*p).T
            W = x[R*n*m + R*m*p :].reshape(R, p*n).T
            
            # Compute T_calc
            # Use einsum
            T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            # Residuals are the entries of T_calc - T
            # We only care about the non-zero entries of T?
            # No, we must satisfy ALL entries.
            # T is zero everywhere else.
            # So residuals = T_calc - T (flattened)
            return (T_calc - T).flatten()

        # Use least_squares
        # This minimizes sum(residuals^2)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.success and result.cost < 1e-10:
                # Found a real solution
                x = result.x
                U = x[:R*n*m].reshape(R, n*m).T
                V = x[R*n*m : R*n*m + R*m*p].reshape(R, m*p).T
                W = x[R*n*m + R*m*p :].reshape(R, p*n).T
                
                # Check if we can convert to integer/half-integer
                # The problem requires integer/half-integer entries.
                # If the real solution is close to such values, we can snap.
                # But is it guaranteed?
                # For now, if we find a real solution, we assume we can refine it.
                # However, the evaluator checks for "integer/half-integer entries".
                # If we return floats, it might fail.
                # We should try to snap to nearest 0.5.
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify the snapped solution
                err = check_solution(U_int, V_int, W_int, R)
                if err < 1e-9:
                    return U_int, V_int, W_int
                else:
                    # If snapping fails, maybe the real solution is valid but not integer?
                    # The task says "integer/half-integer entries".
                    # So we must return integer/half-integer.
                    # If we can't snap, this rank is not valid with our current solution.
                    pass
        except Exception:
            pass
            
        return None

    # Try ranks from schoolbook-1 downwards
    # But 29 is the first one to try.
    # If 29 fails, try 28?
    # Given time constraints, we might only try 29.
    
    best_result = None
    for R in range(R_schoolbook - 1, 29, -1): # Just try 29
        res = try_rank(R)
        if res is not None:
            best_result = res
            break
            
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        rank = R_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
