import numpy as np
import time

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0

def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _build_tensor(n, m, p):
    """Build the sparse tensor for matrix multiplication."""
    # u index: i*m + j
    # v index: j*p + k
    # w index: k*n + i
    indices_u = []
    indices_v = []
    indices_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_u.append(i * m + j)
                indices_v.append(j * p + k)
                indices_w.append(k * n + i)
    return np.array(indices_u), np.array(indices_v), np.array(indices_w)

def _als_decomposition(n, m, p, rank, max_iters=200, tol=1e-10):
    """
    Attempt to find a rank decomposition using Alternating Least Squares.
    Returns (U, V, W, residual) or None if failed.
    """
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Sparse tensor indices
    idx_u, idx_v, idx_w = _build_tensor(n, m, p)
    T_vals = np.ones(len(idx_u))
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(num_u, rank)
    V = np.random.randn(num_v, rank)
    W = np.random.randn(num_w, rank)
    
    best_res = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U
        # M_U[i, r] = sum_{v,w} T[i,v,w] V[r,v] W[r,w]
        # We can compute this efficiently using the sparse indices.
        # M_U is num_u x rank
        M_U = np.zeros((num_u, rank))
        # For each non-zero (u, v, w) in T:
        # M_U[u, r] += T_val * V[r, v] * W[r, w]
        # Vectorized:
        # V_idx = V[:, idx_v] -> rank x len
        # W_idx = W[:, idx_w] -> rank x len
        # Prod = V_idx * W_idx -> rank x len
        # M_U is sum over len for each u
        # We can use np.add.at
        V_idx = V[:, idx_v]
        W_idx = W[:, idx_w]
        Prod_VW = V_idx * W_idx # rank x N
        # M_U[u, r] = sum_{t | idx_u[t]==u} Prod_VW[r, t]
        # Using np.add.at
        np.add.at(M_U, (idx_u, slice(None)), Prod_VW.T) # M_U is num_u x rank
        # Wait, np.add.at(M_U, (idx_u, slice(None)), Prod_VW.T)
        # Prod_VW.T is N x rank.
        # M_U[u, r] += Prod_VW.T[t, r] for all t with idx_u[t]==u.
        # This works.
        
        # K = (V^T V) odot (W^T W)
        GV = V.T @ V
        GW = W.T @ W
        K = GV * GW
        
        # Solve U K = M_U => U = M_U K^{-1}
        try:
            U = M_U @ np.linalg.inv(K)
        except np.linalg.LinAlgError:
            U = M_U @ np.linalg.pinv(K)
            
        # Update V
        M_V = np.zeros((num_v, rank))
        U_idx = U[:, idx_u]
        W_idx = W[:, idx_w]
        Prod_UW = U_idx * W_idx
        np.add.at(M_V, (idx_v, slice(None)), Prod_UW.T)
        
        GU = U.T @ U
        GW = W.T @ W
        K = GU * GW
        
        try:
            V = M_V @ np.linalg.inv(K)
        except np.linalg.LinAlgError:
            V = M_V @ np.linalg.pinv(K)
            
        # Update W
        M_W = np.zeros((num_w, rank))
        U_idx = U[:, idx_u]
        V_idx = V[:, idx_v]
        Prod_UV = U_idx * V_idx
        np.add.at(M_W, (idx_w, slice(None)), Prod_UV.T)
        
        GU = U.T @ U
        GV = V.T @ V
        K = GU * GV
        
        try:
            W = M_W @ np.linalg.inv(K)
        except np.linalg.LinAlgError:
            W = M_W @ np.linalg.pinv(K)
            
        # Calculate residual
        # Residual = || T - sum_r u_r v_r w_r ||^2
        # Approximate using sparse indices
        # Reconstruct T_approx at indices
        # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is expensive if done naively.
        # But N is small (72).
        # We can compute the full tensor? 9x24x24 is small.
        T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_target = np.zeros((num_u, num_v, num_w))
        T_target[idx_u, idx_v, idx_w] = 1.0
        res = np.linalg.norm(T_full - T_target)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < tol:
            break
            
    return best_U, best_V, best_W, best_res

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the decomposition is exact."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if abs(T_full[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                    return False
    # Check other entries are 0
    # Since T is sparse, we can just check the norm of the difference
    T_target = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_target[u_idx, v_idx, w_idx] = 1.0
    if np.linalg.norm(T_full - T_target) > 1e-9:
        return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank
    ranks_to_try = [50, 45, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 23]
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE_SECONDS - SAFETY_MARGIN:
            break
            
        if rank >= best_rank:
            continue
            
        try:
            U, V, W, res = _als_decomposition(n, m, p, rank, max_iters=100, tol=1e-12)
        except Exception as e:
            continue
            
        if res < 1e-9:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                # Convert to lists of floats (or ints if possible)
                best_result = {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                best_rank = rank
                break
            else:
                # Check if real values are exact enough
                if _verify_decomposition(U, V, W, n, m, p):
                    # If real values are exact but not half-integer, we might still return them?
                    # The contract says "integer/half-integer entries".
                    # So we should not return non-integer/half-integer values.
                    # But if we can't find a half-integer one, maybe we stick to schoolbook.
                    # Or maybe the evaluator is lenient?
                    # "exactly decompose" with "integer/half-integer entries".
                    # So we must return integer/half-integer.
                    pass

    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
