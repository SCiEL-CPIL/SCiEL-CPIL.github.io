"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using algebraic methods
    # For 2x3x4 matrix multiplication, the border rank is known to be 16
    # Let's try to find a rank 16 decomposition
    
    # The tensor for 2x3x4 matrix multiplication has a specific structure.
    # We'll use a combination of random initialization and local search.
    
    # First, let's try a structured approach based on known lower bounds
    # The Strassen-like algorithms for small matrices
    
    # Try rank 16 (theoretical lower bound for this size)
    for rank in [16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
            
        try:
            result = _try_rank_structured(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break  # Found a valid decomposition, use it
        except Exception:
            continue
    
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(val - 1.0))
    return max_err < 1e-10

def _snap_to_half_integers(mat, tol=1e-8):
    """Snap values to nearest half-integer if close enough."""
    snapped = []
    for row in mat:
        new_row = []
        for val in row:
            # Try to snap to half-integer
            nearest = round(val * 2) / 2
            if abs(val - nearest) < tol:
                new_row.append(nearest)
            else:
                return None  # Can't snap
        snapped.append(new_row)
    return snapped

def _try_rank_structured(n, m, p, rank, deadline):
    """Try to find a rank decomposition using structured initialization."""
    np.random.seed(42)
    
    # For matrix multiplication, we can use a more structured approach
    # Initialize with random values and use gradient-based optimization
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Define the objective function
    def objective(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    error += (val - 1.0) ** 2
        return error
    
    # Use L-BFGS-B for optimization with bounds
    def gradient(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    err = 2 * (val - 1.0)
                    grad_U[i*m+j] += err * V_flat[j*p+k] * W_flat[k*n+i]
                    grad_V[j*p+k] += err * U_flat[i*m+j] * W_flat[k*n+i]
                    grad_W[k*n+i] += err * U_flat[i*m+j] * V_flat[j*p+k]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try multiple random restarts
    best_result = None
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        if restart > 0:
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, params0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Try to snap to half-integers
                U_snap = _snap_to_half_integers(U_opt.tolist())
                V_snap = _snap_to_half_integers(V_opt.tolist())
                W_snap = _snap_to_half_integers(W_opt.tolist())
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                
                # If snapping fails, try with tighter tolerance
                U_snap = _snap_to_half_integers(U_opt.tolist(), tol=1e-6)
                V_snap = _snap_to_half_integers(V_opt.tolist(), tol=1e-6)
                W_snap = _snap_to_half_integers(W_opt.tolist(), tol=1e-6)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                        
        except Exception:
            continue
    
    return None

# EVOLVE-BLOCK-END
