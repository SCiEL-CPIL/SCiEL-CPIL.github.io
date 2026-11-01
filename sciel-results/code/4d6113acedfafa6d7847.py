import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is valid."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def _round_to_half_int(M):
    """Round matrix entries to nearest half-integer."""
    return np.round(M * 2) / 2

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _als_decomposition(n, m, p, target_rank, time_limit=3.0, seed=42):
    """
    Alternating Least Squares to find a rank-`target_rank` decomposition.
    Returns (U, V, W, is_valid)
    """
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    R = target_rank
    
    # Build target tensor
    T_vec = np.zeros(A_dim * B_dim * C_dim)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_vec[a * B_dim * C_dim + b * C_dim + c] = 1.0
                idx += 1
    
    np.random.seed(seed)
    
    # Initialize: use a structured initialization based on the trivial solution
    # Start with random but scaled
    U = np.random.randn(A_dim, R) * 0.5
    V = np.random.randn(B_dim, R) * 0.5
    W = np.random.randn(C_dim, R) * 0.5
    
    T_mat_a = T_vec.reshape(A_dim, B_dim * C_dim)
    T_mat_b = T_vec.reshape(B_dim, A_dim * C_dim)
    T_mat_c = T_vec.reshape(C_dim, A_dim * B_dim)
    
    start_time = time.time()
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Solve for U given V, W
        M_a = np.zeros((R, B_dim * C_dim))
        for r in range(R):
            M_a[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        try:
            U_new = np.linalg.lstsq(M_a, T_mat_a.T, rcond=None)[0].T
        except:
            return U, V, W, False
        U = U_new
        
        # Solve for V given U, W
        M_b = np.zeros((R, A_dim * C_dim))
        for r in range(R):
            M_b[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new = np.linalg.lstsq(M_b, T_mat_b.T, rcond=None)[0].T
        except:
            return U, V, W, False
        V = V_new
        
        # Solve for W given U, V
        M_c = np.zeros((R, A_dim * B_dim))
        for r in range(R):
            M_c[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new = np.linalg.lstsq(M_c, T_mat_c.T, rcond=None)[0].T
        except:
            return U, V, W, False
        W = W_new
        
        # Check convergence
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_comp.flatten() - T_vec
        if np.linalg.norm(residual) < 1e-8:
            return U, V, W, True
    
    # Final check
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    residual = T_comp.flatten() - T_vec
    is_valid = np.linalg.norm(residual) < 1e-8
    return U, V, W, is_valid

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with trivial solution as guaranteed fallback
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    best_rank = U_triv.shape[1]  # 56
    
    # Try to find lower-rank decompositions
    # Use structured initialization: start from the trivial solution and perturb
    # Try ranks in decreasing order from 49 down to 28
    
    for target_rank in range(49, 27, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            time_per_attempt = min(2.0, (deadline - 5 - (time.time() - start_time)) / 2.0)
            if time_per_attempt < 0.5:
                break
            
            U, V, W, is_valid = _als_decomposition(n, m, p, target_rank, 
                                                     time_limit=time_per_attempt, 
                                                     seed=seed)
            
            if is_valid:
                # Round to half-integers
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                # Re-verify after rounding
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
    
    # If no improvement found, return the trivial solution
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
