import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def fast_verify(U, V, W, n, m, p):
    """Fast verification using numpy einsum."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    u = U.reshape(n, m, rank)
    v = V.reshape(m, p, rank)
    w = W.reshape(p, n, rank)
    tensor = np.einsum('ijr,jkr,kir->ijk', u, v, w)
    return np.allclose(tensor, np.ones((n, m, p)), atol=1e-9)

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

def search_rank_27(n, m, p, time_budget):
    """
    Search for a rank-27 decomposition by fixing the Strassen blocks
    for j∈{1,2}, k∈{0,1,3,4} (rank 14) and optimizing the remaining
    13 terms to cover the j=0 rows and k=2 column.
    """
    start_time = time.time()
    rank = 27
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    
    # Build the fixed Strassen part (14 terms)
    U_fixed = np.zeros((n * m, 14))
    V_fixed = np.zeros((m * p, 14))
    W_fixed = np.zeros((p * n, 14))
    
    def add_strassen_fixed(i_off, j_off, k_off, col):
        U_fixed[(i_off+0)*m + (j_off+0), col] = 1.0
        U_fixed[(i_off+1)*m + (j_off+1), col] = 1.0
        V_fixed[(j_off+0)*p + (k_off+0), col] = 1.0
        V_fixed[(j_off+1)*p + (k_off+1), col] = 1.0
        W_fixed[(k_off+0)*n + (i_off+0), col] = 1.0
        W_fixed[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        U_fixed[(i_off+1)*m + (j_off+0), col] = 1.0
        U_fixed[(i_off+1)*m + (j_off+1), col] = 1.0
        V_fixed[(j_off+0)*p + (k_off+0), col] = 1.0
        W_fixed[(k_off+0)*n + (i_off+1), col] = 1.0
        W_fixed[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1

        U_fixed[(i_off+0)*m + (j_off+0), col] = 1.0
        V_fixed[(j_off+0)*p + (k_off+1), col] = 1.0
        V_fixed[(j_off+1)*p + (k_off+1), col] = -1.0
        W_fixed[(k_off+1)*n + (i_off+0), col] = 1.0
        W_fixed[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        U_fixed[(i_off+1)*m + (j_off+1), col] = 1.0
        V_fixed[(j_off+1)*p + (k_off+0), col] = 1.0
        V_fixed[(j_off+0)*p + (k_off+0), col] = -1.0
        W_fixed[(k_off+0)*n + (i_off+0), col] = 1.0
        W_fixed[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1

        U_fixed[(i_off+0)*m + (j_off+0), col] = 1.0
        U_fixed[(i_off+0)*m + (j_off+1), col] = 1.0
        V_fixed[(j_off+1)*p + (k_off+1), col] = 1.0
        W_fixed[(k_off+0)*n + (i_off+0), col] = -1.0
        W_fixed[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1

        U_fixed[(i_off+1)*m + (j_off+0), col] = 1.0
        U_fixed[(i_off+0)*m + (j_off+0), col] = -1.0
        V_fixed[(j_off+0)*p + (k_off+0), col] = 1.0
        V_fixed[(j_off+0)*p + (k_off+1), col] = 1.0
        W_fixed[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        U_fixed[(i_off+0)*m + (j_off+1), col] = 1.0
        U_fixed[(i_off+1)*m + (j_off+1), col] = -1.0
        V_fixed[(j_off+1)*p + (k_off+0), col] = 1.0
        V_fixed[(j_off+1)*p + (k_off+1), col] = 1.0
        W_fixed[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

        return col

    col = add_strassen_fixed(0, 1, 0, 0)
    col = add_strassen_fixed(0, 1, 3, col)
    
    # Compute what the fixed part contributes to the tensor
    u_f = U_fixed.reshape(n, m, 14)
    v_f = V_fixed.reshape(m, p, 14)
    w_f = W_fixed.reshape(p, n, 14)
    tensor_fixed = np.einsum('ijr,jkr,kir->ijk', u_f, v_f, w_f)
    
    # The remaining tensor to be covered by 13 terms
    target = np.ones((n, m, p)) - tensor_fixed
    
    # Initial guess: schoolbook for the j=0 rows and k=2 column
    U0 = np.zeros((n * m, 13))
    V0 = np.zeros((m * p, 13))
    W0 = np.zeros((p * n, 13))
    
    col = 0
    for i in range(n):
        for k in range(p):
            U0[i * m + 0, col] = 1.0
            V0[0 * p + k, col] = 1.0
            W0[k * n + i, col] = 1.0
            col += 1
    for i in range(n):
        for j in [1, 2]:
            U0[i * m + j, col] = 1.0
            V0[j * p + 2, col] = 1.0
            W0[2 * n + i, col] = 1.0
            col += 1
    
    def reshape(x):
        U = x[:num_u].reshape(n * m, rank)
        V = x[num_u:num_u+num_v].reshape(m * p, rank)
        W = x[num_u+num_v:].reshape(p * n, rank)
        return U, V, W
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return np.inf
        U, V, W = reshape(x)
        u_var = U[:, 14:]
        v_var = V[:, 14:]
        w_var = W[:, 14:]
        u_r = u_var.reshape(n, m, 13)
        v_r = v_var.reshape(m, p, 13)
        w_r = w_var.reshape(p, n, 13)
        tensor_var = np.einsum('ijr,jkr,kir->ijk', u_r, v_r, w_r)
        err = tensor_var - target
        return np.sum(err**2)
    
    # Start with the fixed part + initial variable part
    x0 = np.concatenate([
        U_fixed.flatten(), U0.flatten(),
        V_fixed.flatten(), V0.flatten(),
        W_fixed.flatten(), W0.flatten()
    ])
    
    best_obj = objective(x0)
    best_x = x0.copy()
    
    # Try multiple random initializations
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        
        U_rand = np.zeros((n * m, 13))
        V_rand = np.zeros((m * p, 13))
        W_rand = np.zeros((p * n, 13))
        
        # Random sparse initialization
        for r in range(13):
            idx_u = np.random.choice(n * m, size=2, replace=False)
            idx_v = np.random.choice(m * p, size=2, replace=False)
            idx_w = np.random.choice(p * n, size=2, replace=False)
            U_rand[idx_u, r] = np.random.choice([-1.0, 1.0], size=2)
            V_rand[idx_v, r] = np.random.choice([-1.0, 1.0], size=2)
            W_rand[idx_w, r] = np.random.choice([-1.0, 1.0], size=2)
        
        x_rand = np.concatenate([
            U_fixed.flatten(), U_rand.flatten(),
            V_fixed.flatten(), V_rand.flatten(),
            W_fixed.flatten(), W_rand.flatten()
        ])
        
        obj_rand = objective(x_rand)
        if obj_rand < best_obj:
            best_obj = obj_rand
            best_x = x_rand.copy()
    
    # Optimize
    try:
        result = minimize(objective, best_x, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False})
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    except Exception:
        pass
    
    if best_obj < 1e-9:
        U, V, W = reshape(best_x)
        valid = fast_verify(U, V, W, n, m, p)
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
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
        valid = fast_verify(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = 28
    except Exception as e:
        pass
    
    # Try to find rank-27 solution
    if best_result is not None and time.time() < deadline:
        time_budget = min(deadline - time.time(), 60.0)
        if time_budget > 10.0:
            try:
                result = search_rank_27(n, m, p, time_budget)
                if result is not None:
                    U, V, W = result
                    valid = fast_verify(U, V, W, n, m, p)
                    if valid:
                        best_result = {'U': U, 'V': V, 'W': W}
                        best_rank = 27
            except Exception as e:
                pass
    
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
        
        valid = fast_verify(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
