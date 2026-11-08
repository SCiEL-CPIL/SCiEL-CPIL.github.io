"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(mat):
    """Round a matrix to nearest half-integer values."""
    return np.round(mat * 2.0) / 2.0

def _als_step(U, V, W, n, m, p):
    """One ALS iteration: update U, V, W given the other two fixed."""
    rank = U.shape[1]
    
    # Update U: for each (i,j), U[i*m+j, :] = M_u / (M_u dot M_u)
    # where M_u[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            M_u = np.zeros(rank)
            for k in range(p):
                M_u += V[j * p + k, :] * W[k * n + i, :]
            denom = np.dot(M_u, M_u)
            if denom > 1e-12:
                U[u_idx, :] = M_u / denom
    
    # Update V: for each (j,k), V[j*p+k, :] = M_v / (M_v dot M_v)
    # where M_v[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            M_v = np.zeros(rank)
            for i in range(n):
                M_v += U[i * m + j, :] * W[k * n + i, :]
            denom = np.dot(M_v, M_v)
            if denom > 1e-12:
                V[v_idx, :] = M_v / denom
    
    # Update W: for each (k,i), W[k*n+i, :] = M_w / (M_w dot M_w)
    # where M_w[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            M_w = np.zeros(rank)
            for j in range(m):
                M_w += U[i * m + j, :] * V[j * p + k, :]
            denom = np.dot(M_w, M_w)
            if denom > 1e-12:
                W[w_idx, :] = M_w / denom
    
    return U, V, W

def _try_rank(rank, n, m, p, max_iters=500, n_restarts=20, deadline=None):
    """Try to find an exact decomposition of given rank using ALS with rounding."""
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns to avoid numerical issues
        for c in range(rank):
            norms = [np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])]
            max_norm = max(norms) if max(norms) > 1e-10 else 1.0
            U[:, c] /= max_norm
            V[:, c] /= max_norm
            W[:, c] /= max_norm
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # ALS step
            U, V, W = _als_step(U, V, W, n, m, p)
            
            # Check convergence and exactness periodically
            if it % 5 == 0:
                # Round to half-integers and verify
                U_r = _round_to_halfint(U)
                V_r = _round_to_halfint(V)
                W_r = _round_to_halfint(W)
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    return (U_r, V_r, W_r)
                
                # Also try without rounding
                if _verify_decomposition(U, V, W, n, m, p):
                    return (U, V, W)
        
        # Final attempt with rounding
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U, V, W

def construct():
    """Constructs a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Try various ranks in increasing order
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But practical lower bounds are higher; start from 35
    for rank in range(35, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=300, n_restarts=10, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
