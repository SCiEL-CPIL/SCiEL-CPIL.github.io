import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

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
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W

def _tensor_to_vector(n, m, p):
    """Flatten the multiplication tensor into a vector."""
    vec = np.zeros(n * m * m * p * p * n)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                vec[idx] = 1.0
                idx += 1
    return vec

def _rank_estimate(n, m, p):
    """
    Estimate the minimal rank by computing the SVD of the matricizations of the tensor.
    The rank of the tensor must be at least the rank of any of its matricizations.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Matricizations
    T_a = T.reshape(n * m, m * p * p * n)
    T_b = T.reshape(m * p, n * m * p * n)
    T_c = T.reshape(p * n, n * m * m * p)
    
    s_a = np.linalg.svd(T_a, compute_uv=False)
    s_b = np.linalg.svd(T_b, compute_uv=False)
    s_c = np.linalg.svd(T_c, compute_uv=False)
    
    # Count non-zero singular values (with tolerance)
    tol = 1e-10
    r_a = np.sum(s_a > tol)
    r_b = np.sum(s_b > tol)
    r_c = np.sum(s_c > tol)
    
    return max(r_a, r_b, r_c)

def _optimize_decomposition(n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to find a decomposition of the given rank using least squares.
    We parameterize U, V, W with a shared rank dimension R.
    To handle the non-convexity, we fix one matrix (e.g., W) to a random or structured
    initialization and solve for U and V, or use a simple alternating minimization.
    
    For small ranks, we can try to solve the linear system for U given V,W and vice versa.
    However, the problem is 3-way. A simpler approach for small R:
    Initialize W randomly. 
    Then the equation is sum_r U[:,r] V[:,r]^T W[:,r]^T = T.
    This is linear in U if V,W fixed, and linear in V if U,W fixed.
    We can alternate:
    1. Fix V, W. Solve for U (least squares).
    2. Fix U, W. Solve for V (least squares).
    3. Fix U, V. Solve for W (least squares).
    
    We repeat this until convergence or time limit.
    """
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    R = target_rank
    
    # Target tensor vector
    T_vec = _tensor_to_vector(n, m, p)
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(A_dim, R)
    V = np.random.randn(B_dim, R)
    W = np.random.randn(C_dim, R)
    
    # Precompute structures for speed?
    # The objective is || sum_r u_r v_r^T w_r^T - T ||_F^2
    # Residual vectorized: sum_r (u_r \otimes v_r \otimes w_r) - T_vec
    
    def compute_residual(U, V, W):
        # Einsum to compute tensor sum
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return T_comp.flatten() - T_vec
    
    # Alternating Least Squares
    start_time = time.time()
    max_iter = 500
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Solve for U given V, W
        # Residual r = T - sum_{r != r0} u_r v_r^T w_r^T - u_r0 v_r0^T w_r0^T
        # We want to minimize || T - U M || where M is the matrix formed by V,W
        # T (A_dim x B_dim x C_dim) matricized to (A_dim, B_dim * C_dim)
        # U (A_dim x R)
        # M (R x B_dim * C_dim) where M[r, b*C_dim + c] = V[b,r] * W[c,r]
        
        T_mat_a = T_vec.reshape(A_dim, B_dim * C_dim)
        
        M_a = np.zeros((R, B_dim * C_dim))
        for r in range(R):
            # M_a[r, :] = V[:,r] (B_dim) kron W[:,r] (C_dim) ? 
            # Actually T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So T_mat_a[a, b*C_dim + c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Thus M_a[r, b*C_dim + c] = V[b,r] * W[c,r]
            # This is the outer product of V[:,r] and W[:,r]
            M_a[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
        # Solve U M_a^T = T_mat_a  =>  M_a U^T = T_mat_a^T
        # U^T is R x A_dim
        # M_a is R x (B*C)
        # T_mat_a^T is (B*C) x A_dim
        # System: M_a X = T_mat_a^T
        try:
            U_new = np.linalg.lstsq(M_a, T_mat_a.T, rcond=None)[0].T
        except:
            break
        U = U_new
        
        # 2. Solve for V given U, W
        T_mat_b = T_vec.reshape(B_dim, A_dim * C_dim)
        M_b = np.zeros((R, A_dim * C_dim))
        for r in range(R):
            M_b[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new = np.linalg.lstsq(M_b, T_mat_b.T, rcond=None)[0].T
        except:
            break
        V = V_new
        
        # 3. Solve for W given U, V
        T_mat_c = T_vec.reshape(C_dim, A_dim * B_dim)
        M_c = np.zeros((R, A_dim * B_dim))
        for r in range(R):
            M_c[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new = np.linalg.lstsq(M_c, T_mat_c.T, rcond=None)[0].T
        except:
            break
        W = W_new
        
        # Check convergence
        residual = compute_residual(U, V, W)
        if np.linalg.norm(residual) < 1e-9:
            return U, V, W, True
            
    residual = compute_residual(U, V, W)
    is_valid = np.linalg.norm(residual) < 1e-9
    return U, V, W, is_valid

def _round_to_half_int(M):
    """Round matrix entries to nearest half-integer."""
    return np.round(M * 2) / 2

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Get trivial solution as fallback
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_rank = U_triv.shape[1]
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Estimate lower bound on rank
    lower_bound = _rank_estimate(n, m, p)
    
    # Search for rank between lower_bound and best_rank - 1
    # We know trivial is 56. Block Strassen was 50.
    # Let's try to find if we can go below 50.
    
    # The SVD lower bound for 2x4x7:
    # T is 8 x 28 x 14.
    # Mat A (8 x 392): Rank is likely 8 (full row rank)
    # Mat B (28 x 112): Rank is likely 28 (full row rank)
    # Mat C (14 x 224): Rank is likely 14 (full row rank)
    # So lower bound is 28? No, the rank of the tensor is at least the max rank of matricizations.
    # But the matricization rank is an upper bound on the tensor rank? No.
    # Tensor rank R implies matricization rank <= R.
    # So R >= max(rank(T_a), rank(T_b), rank(T_c)).
    
    # Let's print the lower bound for debugging in thought process, but keep code clean.
    # For 2x2x2, SVD ranks are 4,4,4. Max is 4. Tensor rank is 7.
    # So SVD lower bound is weak.
    
    # Strategy: Try ranks from lower_bound up to 49.
    # Since 50 is known to work (from previous attempts, though not verified in this specific code yet),
    # we should verify 50 first, then try 49, 48...
    
    # Actually, let's just try to optimize for a few specific ranks.
    # If we find a valid decomposition for rank R, we return it.
    
    # We will iterate R from lower_bound to 55.
    # To save time, we prioritize R close to 50 first, then lower.
    
    ranks_to_try = list(range(lower_bound, 56))
    # Reorder to try 49, 48, 47... first, then 50, 51...
    # Actually, if 50 works, we want to know. If 49 works, even better.
    # Let's try 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28
    # Then 50, 51...
    
    # Efficient ordering:
    target_ranks = []
    for r in range(lower_bound, 50):
        target_ranks.append(r)
    for r in range(50, 56):
        target_ranks.append(r)
        
    # Shuffle slightly or just go in order?
    # Let's go from 49 down to lower_bound, then 50 up.
    # If we find a valid one, we stop.
    
    # To be safe, let's first verify if rank 50 is valid via optimization.
    # If optimization is too slow, we might miss it.
    
    # Let's try a few random restarts for each rank.
    
    found = False
    current_best_rank = 56
    
    # Try ranks from 49 down to lower_bound
    for R in range(49, lower_bound - 1, -1):
        if time.time() - start_time > deadline - 10:
            break
            
        success = False
        for restart in range(5):
            if time.time() - start_time > deadline - 5:
                break
            U, V, W, is_valid = _optimize_decomposition(n, m, p, R, time_limit=5.0)
            if is_valid:
                # Round to half-integers and re-verify
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                T_comp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                T_vec_comp = T_comp.flatten()
                T_vec_target = _tensor_to_vector(n, m, p)
                
                if np.allclose(T_vec_comp, T_vec_target, atol=1e-9):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    current_best_rank = R
                    found = True
                    break
            # If not valid, try next restart
        
        if found:
            break # Found a valid rank, stop searching lower
            
    # If we didn't find anything below 50, try 50
    if not found:
        for R in range(50, 56):
            if time.time() - start_time > deadline - 5:
                break
            success = False
            for restart in range(5):
                if time.time() - start_time > deadline - 2:
                    break
                U, V, W, is_valid = _optimize_decomposition(n, m, p, R, time_limit=2.0)
                if is_valid:
                    U_r = _round_to_half_int(U)
                    V_r = _round_to_half_int(V)
                    W_r = _round_to_half_int(W)
                    
                    T_comp = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    T_vec_comp = T_comp.flatten()
                    T_vec_target = _tensor_to_vector(n, m, p)
                    
                    if np.allclose(T_vec_comp, T_vec_target, atol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        current_best_rank = R
                        found = True
                        break
            if found:
                break
                
    # If still not found, use trivial
    if not found:
        best_U, best_V, best_W = U_triv, V_triv, W_triv
        current_best_rank = 56

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
