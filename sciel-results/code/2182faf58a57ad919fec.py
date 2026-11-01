import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from factor matrices efficiently."""
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    col = add_strassen(0, 1, 0, col)
    
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)
    
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def search_lower_rank(n, m, p, target_rank, time_limit=30.0, seed=42):
    """
    Try to find a decomposition of rank < current best using numerical optimization.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Total number of variables: rank * (n*m + m*p + p*n)
    total_vars = target_rank * (n * m + m * p + p * n)
    
    def unpack(params):
        U_flat = params[:target_rank * n * m].reshape(target_rank, n * m).T
        V_flat = params[target_rank * n * m : target_rank * (n * m + m * p)].reshape(target_rank, m * p).T
        W_flat = params[target_rank * (n * m + m * p):].reshape(target_rank, p * n).T
        return U_flat, V_flat, W_flat
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        U_flat, V_flat, W_flat = unpack(params)
        tensor = np.zeros((n, m, p))
        for r in range(target_rank):
            u = U_flat[r].reshape(n, m)
            v = V_flat[r].reshape(m, p)
            w = W_flat[r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        expected = np.ones((n, m, p))
        return np.sum((tensor - expected) ** 2)
    
    # Try multiple random initializations
    best_obj = float('inf')
    best_params = None
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        # Initialize with small random values
        params = rng.randn(total_vars) * 0.5
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            continue
    
    if best_obj < 1e-6 and best_params is not None:
        U_flat, V_flat, W_flat = unpack(best_params)
        # Snap to half-integers
        U_snap = snap_to_half_integers(U_flat.tolist())
        V_snap = snap_to_half_integers(V_flat.tolist())
        W_snap = snap_to_half_integers(W_flat.tolist())
        # Verify
        valid, _ = verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
        if valid:
            return U_snap, V_snap, W_snap, target_rank
    return None

def snap_to_half_integers(matrix):
    """
    Snap values to the nearest integer or half-integer.
    This ensures the output strictly satisfies the 'integer or half-integer' constraint
    in case of any floating point drift, though the construction uses exact values.
    """
    snapped = []
    for row in matrix:
        new_row = []
        for val in row:
            # Multiply by 2, round, divide by 2
            snapped_val = round(val * 2) / 2.0
            new_row.append(snapped_val)
        snapped.append(new_row)
    return snapped

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
            # Snap to ensure strict integer/half-integer compliance
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            best_result = {'U': U_snap, 'V': V_snap, 'W': W_snap}
            best_rank = 28
    except Exception as e:
        pass
    
    # Try to find lower rank decompositions
    if best_result is not None:
        for target_rank in range(best_rank - 1, max(14, best_rank - 5) - 1, -1):
            if time.time() > deadline:
                break
            try:
                result = search_lower_rank(n, m, p, target_rank, 
                                         time_limit=min(20.0, deadline - time.time()),
                                         seed=target_rank)
                if result is not None:
                    U_snap, V_snap, W_snap, rank = result
                    best_result = {'U': U_snap, 'V': V_snap, 'W': W_snap}
                    best_rank = rank
                    break  # Found lower rank, stop searching
            except Exception as e:
                continue
    
    # Fallback: Schoolbook (Rank 30)
    if best_result is None:
        rank_school = n * m * p
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            best_result = {'U': U_snap, 'V': V_snap, 'W': W_snap}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
