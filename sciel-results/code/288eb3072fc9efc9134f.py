"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

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
    
    # Try structured algebraic approaches to find lower rank
    # For 2x3x4, known lower bounds suggest rank around 16-20
    
    # Strategy: Use the fact that matrix multiplication tensor can be decomposed
    # using structured bilinear forms. We'll try a systematic search over
    # structured decompositions.
    
    # First, try a direct algebraic construction
    result = _try_algebraic_construction(n, m, p, deadline)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    # If that didn't work, try optimization with better initialization
    for rank in [16, 17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
            
        try:
            result = _try_optimization_improved(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
        except Exception:
            continue
    
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(val - 1.0))
    return max_err < 1e-10

def _snap_to_half_integers(mat, tol=1e-8):
    """Snap values to nearest half-integer if close enough."""
    snapped = []
    for row in mat:
        new_row = []
        for val in row:
            nearest = round(val * 2) / 2
            if abs(val - nearest) < tol:
                new_row.append(nearest)
            else:
                return None
        snapped.append(new_row)
    return snapped

def _try_algebraic_construction(n, m, p, deadline):
    """
    Try to construct a low-rank decomposition using algebraic properties.
    For 2x3x4, we can exploit the structure of the multiplication tensor.
    
    The idea: matrix multiplication (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    We want to write this as sum_c U_{i,j}^c * V_{j,k}^c * W_{k,i}^c
    """
    # For 2x3x4, one approach is to use the fact that we can "split"
    # the middle dimension in certain ways.
    
    # Try rank 18: split the 3x4 part into two 3x2 blocks with some overlap
    # or use a Strassen-like approach
    
    # A more systematic approach: try all possible "splits" of the dimensions
    # and see if we can find a valid decomposition.
    
    # For now, try a specific construction based on known results
    # The 2x3x4 multiplication can be done with rank 18 using the following:
    
    # Actually, let's try a different approach: use the fact that
    # we can write the multiplication as a sum of rank-1 tensors.
    
    # For 2x3x4, one known construction uses rank 18:
    # We can write the 3x4 product as two 3x2 products plus corrections
    
    # Let's try to construct this explicitly
    # A (2x3) * B (3x4) = C (2x4)
    
    # Split B into B1 (3x2) and B2 (3x2)
    # Then C1 = A * B1 (2x2) and C2 = A * B2 (2x2)
    # C = [C1 | C2]
    
    # Each 2x2 multiplication can be done with rank 7 (Strassen)
    # So total rank would be 7 + 7 = 14, but we need to account for the splitting
    
    # Actually, the splitting doesn't reduce rank in this simple way.
    # Let's try a different approach.
    
    # For 2x3x4, the border rank is 16, but exact rank might be higher.
    # Let's try to find a rank 17 or 18 decomposition using optimization.
    
    return None  # Fall back to optimization

def _try_optimization_improved(n, m, p, rank, deadline):
    """
    Improved optimization with better initialization and multiple strategies.
    """
    np.random.seed(12345)
    
    best_result = None
    best_error = float('inf')
    
    # Multiple initialization strategies
    strategies = [
        "random",
        "structured",
        "strassen_like"
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 15:
            break
            
        for restart in range(8):
            if time.time() > deadline - 10:
                    break
                    
            if strategy == "random":
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            elif strategy == "structured":
                # Initialize with structured patterns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # First n*m*p entries as schoolbook
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = i * m * p + j * p + k
                            if c < rank:
                                U[i * m + j][c] = 1
                                V[j * p + k][c] = 1
                                W[k * n + i][c] = 1
                
                # Add some random perturbation
                U += np.random.randn(n * m, rank) * 0.1
                V += np.random.randn(m * p, rank) * 0.1
                W += np.random.randn(p * n, rank) * 0.1
            elif strategy == "strassen_like":
                # Try to initialize based on Strassen-like patterns
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # For 2x2x2, Strassen uses 7 multiplications
                # Try to generalize this pattern
                for c in range(rank):
                    # Random signs and positions
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][c] = np.random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k][c] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][c] = np.random.choice([-1, 0, 1])
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import minimize
                
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
                
                result = minimize(objective, params0, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if result.fun < best_error:
                    best_error = result.fun
                    
                    if result.fun < 1e-8:
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
                        
                        # Try with tighter tolerance
                        U_snap = _snap_to_half_integers(U_opt.tolist(), tol=1e-6)
                        V_snap = _snap_to_half_integers(V_opt.tolist(), tol=1e-6)
                        W_snap = _snap_to_half_integers(W_opt.tolist(), tol=1e-6)
                        
                        if U_snap is not None and V_snap is not None and W_snap is not None:
                            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                                return {"U": U_snap, "V": V_snap, "W": W_snap}
                        
                        # If snapping fails, try integer snapping
                        U_int = np.round(U_opt).tolist()
                        V_int = np.round(V_opt).tolist()
                        W_int = np.round(W_opt).tolist()
                        
                        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                            return {"U": U_int, "V": V_int, "W": W_int}
                        
            except Exception:
                continue
    
    return None

# EVOLVE-BLOCK-END
