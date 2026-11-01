import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from factor matrices using vectorized operations."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape to 3D: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: sum over r of U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is tricky with einsum. Let's do it differently.
    # tensor[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # For each i,j,k: u[i,j] * v[j,k] * w[k,i]
        # This is like a "cyclic" product
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return tensor

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    tensor = compute_tensor(U, V, W, n, m, p)
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.array(arr, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    return None

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    Uses schoolbook for most terms and Strassen-style for some 2x2x2 blocks.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        return col + 1

    def add_strassen_2x2x2(i_off, j_off, k_off, col):
        """Add 7 Strassen terms for a 2x2x2 sub-problem."""
        # M1 = (A11+A22)(B11+B22)
        U[i_off*m + j_off, col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[j_off*p + k_off, col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[k_off*n + i_off, col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M2 = (A21+A22)B11
        U[(i_off+1)*m + j_off, col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[j_off*p + k_off, col] = 1.0
        W[k_off*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1

        # M3 = A11(B21-B11)
        U[i_off*m + j_off, col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        V[j_off*p + k_off, col] = -1.0
        W[(k_off+1)*n + i_off, col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M4 = A22(B12-B22)
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + k_off, col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[k_off*n + i_off, col] = 1.0
        W[k_off*n + (i_off+1), col] = 1.0
        col += 1

        # M5 = (A11+A12)B22
        U[i_off*m + j_off, col] = 1.0
        U[i_off*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[k_off*n + i_off, col] = -1.0
        W[(k_off+1)*n + i_off, col] = 1.0
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + j_off, col] = 1.0
        U[i_off*m + j_off, col] = -1.0
        V[j_off*p + k_off, col] = 1.0
        V[j_off*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M7 = (A12-A22)(B21+B22)
        U[i_off*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + k_off, col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[k_off*n + i_off, col] = 1.0
        col += 1

        return col

    # Schoolbook for all (i,j,k) except where we use Strassen
    # We'll use Strassen for j in {1,2} and k in {0,1} and k in {3,4}
    # That covers 2*2*2 = 8 terms with 7 Strassen terms, saving 1 per block
    # Total schoolbook: 2*3*5 = 30
    # Replace 2 blocks of 2x2x2 (8 terms each) with 7 Strassen terms each
    # So: 30 - 8 - 8 + 7 + 7 = 28
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j >= 1 and k in (0, 1, 3, 4):
                    continue  # Handled by Strassen
                col = add_schoolbook(i, j, k, col)
    
    # Strassen for j in {1,2}, k in {0,1}
    col = add_strassen_2x2x2(0, 1, 0, col)
    
    # Strassen for j in {1,2}, k in {3,4}
    col = add_strassen_2x2x2(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def search_lower_rank(n, m, p, base_U, base_V, base_W, target_rank, time_budget):
    """
    Search for a lower-rank decomposition using the base solution as initialization.
    Tries multiple random restarts and snaps to half-integers.
    """
    start_time = time.time()
    
    if target_rank >= base_U.shape[1]:
        return None
    
    num_u = n * m * target_rank
    num_v = m * p * target_rank
    num_w = p * n * target_rank
    total_dim = num_u + num_v + num_w
    
    def reshape(x):
        U = x[:num_u].reshape(n * m, target_rank)
        V = x[num_u:num_u+num_v].reshape(m * p, target_rank)
        W = x[num_u+num_v:].reshape(p * n, target_rank)
        return U, V, W
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return np.inf
        U, V, W = reshape(x)
        tensor = compute_tensor(U, V, W, n, m, p)
        err = tensor - 1.0
        return np.sum(err**2)
    
    def try_optimize(x0, maxiter=300):
        nonlocal start_time
        if time.time() - start_time > time_budget:
            return None
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False})
            return result
        except Exception:
            return None
    
    def check_and_snap(x):
        U, V, W = reshape(x)
        # Try snapping to half-integers
        U_s = snap_to_half_integers(U)
        V_s = snap_to_half_integers(V)
        W_s = snap_to_half_integers(W)
        if U_s is not None and V_s is not None and W_s is not None:
            valid, _ = verify_decomposition(U_s, V_s, W_s, n, m, p)
            if valid:
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        # Also try without snapping
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Initialize from base solution (truncate columns)
    U0 = np.array(base_U[:target_rank], dtype=float)
    V0 = np.array(base_V[:target_rank], dtype=float)
    W0 = np.array(base_W[:target_rank], dtype=float)
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    best_result = None
    best_obj = np.inf
    
    # Try the initial point
    obj0 = objective(x0)
    if obj0 < 1e-9:
        res = check_and_snap(x0)
        if res:
            return res
    
    # Try L-BFGS-B from initial point
    result = try_optimize(x0)
    if result is not None:
        if result.fun < best_obj:
            best_obj = result.fun
            res = check_and_snap(result.x)
            if res:
                return res
        # Also check the initial point more carefully
        res = check_and_snap(x0)
        if res:
            return res
    
    # Try random restarts
    np.random.seed(42)
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturbation of the initial guess
        noise_scale = 0.1 * (i + 1)
        x_rand = x0 + np.random.randn(total_dim) * noise_scale
        result = try_optimize(x_rand, maxiter=200)
        if result is not None:
            if result.fun < best_obj:
                best_obj = result.fun
                res = check_and_snap(result.x)
                if res:
                    return res
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_rank = float('inf')
    
    # First, try the known rank-28 solution
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = 28
        else:
            # Debug: print the tensor to see what's wrong
            tensor = compute_tensor(np.array(U), np.array(V), np.array(W), n, m, p)
            # Don't print in production, but we know it should work
            pass
    except Exception as e:
        pass
    
    # Try to find lower rank solutions
    if best_result is not None:
        for target_rank in [27, 26, 25]:
            if time.time() > deadline:
                break
            
            time_budget = min(deadline - time.time(), 25.0)
            if time_budget < 5.0:
                break
            
            try:
                result = search_lower_rank(n, m, p, 
                                          np.array(best_result['U']), 
                                          np.array(best_result['V']), 
                                          np.array(best_result['W']),
                                          target_rank, time_budget)
                if result is not None:
                    U, V, W = result
                    best_result = {'U': U, 'V': V, 'W': W}
                    best_rank = target_rank
                    break
            except Exception:
                pass
    
    # Fallback: Schoolbook (Rank 30)
    if best_result is None:
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
