import time
import numpy as np
import scipy.optimize as optimize
import scipy.linalg as la

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _get_matricization_rank(n, m, p):
    """
    Compute the rank of the matricization of the matmul tensor.
    This provides a lower bound on the tensor rank.
    """
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    # Matricization along U (size n_u x (n_v * n_w))
    # T_{u, v, w} is 1 if indices match
    # We construct the matrix M where M[u, v*n_w + w] = T_{u,v,w}
    M = np.zeros((n_u, n_v * n_w))
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            # For this u_idx, which v_idx and w_idx are connected?
            # v_idx must have row j. v_idx = j * p + k
            # w_idx must have col i and row k. w_idx = k * n + i
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                M[u_idx, v_idx * n_w + w_idx] = 1.0
                
    rank_u = la.matrix_rank(M)
    
    # Matricization along V (size n_v x (n_u * n_w))
    M_v = np.zeros((n_v, n_u * n_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                M_v[v_idx, u_idx * n_w + w_idx] = 1.0
    rank_v = la.matrix_rank(M_v)
    
    # Matricization along W (size n_w x (n_u * n_v))
    M_w = np.zeros((n_w, n_u * n_v))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                M_w[w_idx, u_idx * n_v + v_idx] = 1.0
    rank_w = la.matrix_rank(M_w)
    
    return max(rank_u, rank_v, rank_w)

def _solve_rank_r(n, m, p, r):
    """
    Attempt to find a rank-r decomposition for n x m x p matmul.
    """
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def objective(params):
        U = params[:n_u*r].reshape(n_u, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
        W = params[n_u*r + n_v*r :].reshape(n_w, r)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals

    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    for restart in range(100):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Random initialization
        U0 = np.random.randn(n_u, r) * 1.0
        V0 = np.random.randn(n_v, r) * 1.0
        W0 = np.random.randn(n_w, r) * 1.0
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=100)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n_u*r].reshape(n_u, r)
            V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
            W_sol = res.x[n_u*r + n_v*r :].reshape(n_w, r)
            
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap

    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    schoolbook_rank = n * m * p
    
    # 2. Compute lower bound from matricizations
    lb_rank = _get_matricization_rank(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # 3. If lower bound is less than schoolbook, try to find a decomposition
    if lb_rank < schoolbook_rank:
        # Try ranks from lb_rank up to schoolbook_rank - 1
        for r in range(lb_rank, schoolbook_rank):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
                
            result = _solve_rank_r(n, m, p, r)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = r
                    break # Found a valid lower rank, stop searching for even lower?
                    # Actually, we should search from lb_rank upwards, so the first one found is minimal?
                    # Yes, if we iterate r from lb_rank to schoolbook-1.

    # 4. Return best
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
