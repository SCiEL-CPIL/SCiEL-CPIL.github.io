import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-10 decomposition
    # We use a structured approach:
    # 1. Define a parameterization for rank 10
    # 2. Solve for the remaining variables using linear algebra
    
    # Let's attempt a direct construction or search for rank 10.
    # Known bounds: R(2,2,3) <= 11 (Strassen + 4). Lower bound is 10?
    # Actually, for 2x2x3, the rank is known to be 10? Or 11?
    # Let's check literature.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x2x3, one can extend Strassen.
    # A common decomposition uses rank 10.
    
    # Let's try to construct a rank 10 decomposition.
    # We can use the "border rank" or specific tensor identities.
    
    # Strategy:
    # Use numpy to set up the linear system for rank R.
    # If we assume a certain structure for U, V, W, we can solve for the rest.
    
    # Let's try a heuristic search for rank 10.
    # We will define the tensor T of shape (4, 6, 6).
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # We want T = sum_r U_r (outer) V_r (outer) W_r.
    # This is equivalent to: for each r, T += U_r[:, None, None] * V_r[None, :, None] * W_r[None, None, :]
    # Wait, the indices are:
    # U index: i*m + j (4)
    # V index: j*p + k (6)
    # W index: k*n + i (6)
    
    # Let's build the target tensor T.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Now, we try to decompose T into rank R.
    # We can use a random initialization and gradient descent?
    # Or we can use a structured approach.
    
    # Let's try to find a rank 10 decomposition using a specific pattern.
    # We can fix some entries of U, V, W and solve the linear system for the rest.
    
    # Since R=10 is close to 11, maybe we can modify the rank 11 solution.
    # Rank 11 solution:
    # 7 Strassen factors for the 2x2 part + 4 standard for the 2x1 part.
    # Can we merge one of the standard factors with a Strassen factor?
    # Or use a different Strassen-like scheme?
    
    # Let's try a numerical optimization for rank 10.
    # We will use a simple gradient-free method or a linear solve with random projections.
    
    # Actually, a known result is that R(2,2,3) = 10.
    # Let's try to construct it.
    
    # One approach:
    # Use the fact that the tensor is small.
    # We can set up a system of equations.
    # Let U, V, W be matrices of size (4,10), (6,10), (6,10).
    # We want sum_r U[:,r] V[:,r] W[:,r] = T.
    
    # This is a non-linear problem.
    # However, if we fix U and V, the problem becomes linear in W.
    # vec(T) = sum_r (U_r (kron) V_r) W_r? No.
    # T_{uvw} = sum_r U_{u,r} V_{v,r} W_{w,r}.
    # For a fixed r, this is an outer product of U_r, V_r, W_r.
    
    # Let's try to use a library or a simple iterative method.
    # Given the time limit, a full optimization might be risky.
    
    # Let's try a specific known rank 10 decomposition if possible.
    # If not, we will return the rank 11 one.
    
    # Let's attempt a rank 10 construction.
    # Reference: "On the rank of the 2x2x3 matrix multiplication tensor"
    # It is known to be 10.
    
    # Let's try to build it.
    # We can use the following idea:
    # Combine the last two standard multiplication terms with the Strassen terms?
    
    # Let's try to solve for rank 10 using a randomized approach with linear algebra.
    # We will randomly initialize U and V, and solve for W.
    # Then we check the residual.
    # We can iterate: fix W, solve for U; fix U, solve for V; etc. (ALS - Alternating Least Squares).
    
    R = 10
    U = np.random.randn(n*m, R)
    V = np.random.randn(m*p, R)
    W = np.random.randn(p*n, R)
    
    # Normalize columns to prevent scaling issues
    for r in range(R):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 0 and norm_v > 0 and norm_w > 0:
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
    
    # ALS iteration
    for it in range(50):
        if time.time() > deadline - 10:
            break
            
        # Solve for W
        # T_{uvw} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # For each w, T_{uvw} is a matrix in u,v.
        # T_w = sum_r (U_r V_r^T) W_{w,r}
        # This is a linear system for W[:, r] for each w?
        # Let's reshape.
        # T is (4, 6, 6).
        # Let A_r = U[:, r] (4x1), B_r = V[:, r] (6x1), C_r = W[:, r] (6x1).
        # T = sum_r A_r (kron) B_r (kron) C_r? No, it's a 3-way outer product.
        # T_{u,v,w} = sum_r A_{u,r} B_{v,r} C_{w,r}.
        
        # Fix U and V. We want to find W.
        # For each r, we have a term A_r B_r^T C_r? No.
        # Let's vectorize T along w.
        # T_w (4x6) = sum_r (A_r (outer) B_r) * C_{w,r}
        # Let M_r = A_r (outer) B_r (4x6).
        # T_w = sum_r M_r * C_{w,r}
        # This means T_w = M C_w, where M is (4*6, R) with columns vec(M_r), and C_w is (R,).
        # Wait, T_w is 4x6. vec(T_w) is 24x1.
        # vec(T_w) = sum_r vec(M_r) C_{w,r}
        # So we can solve for C_w (which is W[:, w]) using least squares.
        
        M = np.zeros((n*m * m*p, R))
        for r in range(R):
            M_r = np.outer(U[:, r], V[:, r])
            M[:, r] = M_r.flatten()
            
        # Solve for W
        # vec(T[:, :, w]) = M W[:, w]
        # We can solve all w at once.
        # T_reshaped = T.reshape(n*m, m*p, p*n).transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        T_reshaped = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m*m*p)
        
        # Solve T_reshaped = W^T M?
        # vec(T_w) = M W_w.
        # So T_reshaped = W^T M.
        # W = (T_reshaped M^T) (M M^T)^-1?
        # Let's just use lstsq.
        # We want to find W (p*n, R) such that W^T M = T_reshaped.
        # This is equivalent to M^T W = T_reshaped^T.
        W_new, _, _, _ = np.linalg.lstsq(M.T, T_reshaped.T, rcond=None)
        W = W_new
        
        # Solve for U
        # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # Fix V and W.
        # For each u, T_{u,v,w} = sum_r U_{u,r} (V_r (outer) W_r)_{v,w}
        # Let N_r = V_r (outer) W_r (6x6).
        # T_u = sum_r U_{u,r} N_r
        # vec(T_u) = N U_u, where N is (36, R) with columns vec(N_r).
        N = np.zeros((m*p * p*n, R))
        for r in range(R):
            N_r = np.outer(V[:, r], W[:, r])
            N[:, r] = N_r.flatten()
            
        T_reshaped_u = np.transpose(T, (0, 1, 2)).reshape(n*m, m*p*p*n)
        # vec(T_u) = N U_u
        # U^T N = T_reshaped_u
        # N^T U = T_reshaped_u^T
        U_new, _, _, _ = np.linalg.lstsq(N.T, T_reshaped_u.T, rcond=None)
        U = U_new
        
        # Solve for V
        # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # Fix U and W.
        # For each v, T_{u,v,w} = sum_r V_{v,r} (U_r (outer) W_r)_{u,w}
        # Let P_r = U_r (outer) W_r (4x6).
        # T_v = sum_r V_{v,r} P_r
        # vec(T_v) = P V_v, where P is (24, R) with columns vec(P_r).
        P = np.zeros((n*m * p*n, R))
        for r in range(R):
            P_r = np.outer(U[:, r], W[:, r])
            P[:, r] = P_r.flatten()
            
        T_reshaped_v = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m*p*n)
        # vec(T_v) = P V_v
        # P^T V = T_reshaped_v^T
        V_new, _, _, _ = np.linalg.lstsq(P.T, T_reshaped_v.T, rcond=None)
        V = V_new
        
        # Check residual
        T_approx = np.zeros_like(T)
        for r in range(R):
            T_approx += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        residual = np.linalg.norm(T - T_approx)
        
        if residual < 1e-6:
            break
            
    # Check if rank 10 works
    T_approx = np.zeros_like(T)
    for r in range(R):
        T_approx += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    residual = np.linalg.norm(T - T_approx)
    
    if residual < 1e-6:
        # Convert to lists of floats
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Ensure half-integers? The problem says integer/half-integer.
        # ALS might produce arbitrary floats.
        # We need to check if the solution is close to half-integers.
        # If not, we might need to snap or use a different method.
        # For now, if it's a valid real decomposition, does it count?
        # "integer/half-integer entries"
        # This is a strict constraint.
        
        # Let's check if we can snap to half-integers.
        def snap_to_half(x):
            return round(2*x) / 2.0
            
        U_snapped = np.array([[snap_to_half(x) for x in row] for row in U_list])
        V_snapped = np.array([[snap_to_half(x) for x in row] for row in V_list])
        W_snapped = np.array([[snap_to_half(x) for x in row] for row in W_list])
        
        T_approx_snapped = np.zeros_like(T)
        for r in range(R):
            T_approx_snapped += np.einsum('u,v,w->uvw', U_snapped[:, r], V_snapped[:, r], W_snapped[:, r])
        residual_snapped = np.linalg.norm(T - T_approx_snapped)
        
        if residual_snapped < 1e-6:
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        else:
            # Fall back to rank 11
            pass
    else:
        # Fall back to rank 11
        pass
        
    # Rank 11 Construction (Strassen + Standard)
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_u(i, j, r, val):
        idx = i * m + j
        U[idx][r] += val
        
    def add_v(j, k, r, val):
        idx = j * p + k
        V[idx][r] += val
        
    def add_w(k, i, r, val):
        idx = k * n + i
        W[idx][r] += val

    # r=0: M1
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)
    add_w(1, 1, 0, 1)

    # r=1: M2
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)
    add_w(1, 1, 1, -1)

    # r=2: M3
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)
    add_w(1, 1, 2, 1)

    # r=3: M4
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 0, 3, 1)
    add_w(0, 1, 3, 1)

    # r=4: M5
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)
    add_w(1, 0, 4, 1)

    # r=5: M6
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)

    # r=6: M7
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)

    # r=7: A00 * B02
    add_u(0, 0, 7, 1)
    add_v(0, 2, 7, 1)
    add_w(2, 0, 7, 1)

    # r=8: A01 * B12
    add_u(0, 1, 8, 1)
    add_v(1, 2, 8, 1)
    add_w(2, 0, 8, 1)

    # r=9: A10 * B02
    add_u(1, 0, 9, 1)
    add_v(0, 2, 9, 1)
    add_w(2, 1, 9, 1)

    # r=10: A11 * B12
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1)
    add_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
