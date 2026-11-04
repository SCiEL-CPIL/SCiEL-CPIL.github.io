import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 115  # Safety margin before 120s limit

    # Precompute valid and invalid indices for the tensor T
    # T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                
    total_indices = []
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                total_indices.append((alpha, beta, gamma))
                
    valid_set = set(valid_indices)
    invalid_indices = [idx for idx in total_indices if idx not in valid_set]

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        # Compute T_approx[alpha, beta, gamma] = sum_c U[alpha,c] V[beta,c] W[gamma,c]
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        
        for (a, b, g) in valid_indices:
            if not np.isclose(T_approx[a, b, g], 1.0, atol=atol):
                return False
        for (a, b, g) in invalid_indices:
            if not np.isclose(T_approx[a, b, g], 0.0, atol=atol):
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    # Strategy 1: Try the known rank 34 decomposition from the current program
    # This is a verified solution. If it works, we hold it as a candidate.
    def try_known_rank_34():
        rank = 34
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # Part 1: A_TL (2x2) * B_top_L (2x2) -> C_top_L (2x2)
        # Strassen 2x2
        U[0, col] += 1; U[4, col] += 1
        V[0, col] += 1; V[5, col] += 1
        W[0, col] += 1; W[4, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[0, col] += 1
        W[1, col] += 1; W[4, col] += -1
        col += 1
        
        U[0, col] += 1
        V[1, col] += 1; V[5, col] += -1
        W[3, col] += 1; W[4, col] += 1
        col += 1
        
        U[4, col] += 1
        V[4, col] += 1; V[0, col] += -1
        W[0, col] += 1; W[1, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[5, col] += 1
        W[3, col] += 1; W[0, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[0, col] += 1; V[1, col] += 1
        W[4, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[4, col] += 1; V[5, col] += 1
        W[0, col] += 1
        col += 1
        
        # Part 2: A_TL (2x2) * B_top_R (2x2) -> C_top_R (2x2)
        U[0, col] += 1; U[4, col] += 1
        V[2, col] += 1; V[6, col] += 1
        W[6, col] += 1; W[9, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[2, col] += 1
        W[7, col] += 1; W[10, col] += -1
        col += 1
        
        U[0, col] += 1
        V[3, col] += 1; V[6, col] += -1
        W[9, col] += 1; W[10, col] += 1
        col += 1
        
        U[4, col] += 1
        V[6, col] += 1; V[2, col] += -1
        W[6, col] += 1; W[7, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[6, col] += 1
        W[9, col] += 1; W[6, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[2, col] += 1; V[3, col] += 1
        W[10, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[6, col] += 1; V[7, col] += 1
        W[6, col] += 1
        col += 1
        
        # Part 3: A_TR (2x1) * B_bot (1x4)
        U[2, col] += 1; V[8, col] += 1; W[0, col] += 1; col += 1
        U[2, col] += 1; V[9, col] += 1; W[3, col] += 1; col += 1
        U[2, col] += 1; V[10, col] += 1; W[6, col] += 1; col += 1
        U[2, col] += 1; V[11, col] += 1; W[9, col] += 1; col += 1
        
        U[5, col] += 1; V[8, col] += 1; W[1, col] += 1; col += 1
        U[5, col] += 1; V[9, col] += 1; W[4, col] += 1; col += 1
        U[5, col] += 1; V[10, col] += 1; W[7, col] += 1; col += 1
        U[5, col] += 1; V[11, col] += 1; W[10, col] += 1; col += 1
        
        # Part 4: A_BL (1x2) * B_top (2x4)
        U[6, col] += 1; V[0, col] += 1; W[2, col] += 1; col += 1
        U[7, col] += 1; V[4, col] += 1; W[2, col] += 1; col += 1
        
        U[6, col] += 1; V[1, col] += 1; W[5, col] += 1; col += 1
        U[7, col] += 1; V[5, col] += 1; W[5, col] += 1; col += 1
        
        U[6, col] += 1; V[2, col] += 1; W[8, col] += 1; col += 1
        U[7, col] += 1; V[6, col] += 1; W[8, col] += 1; col += 1
        
        U[6, col] += 1; V[3, col] += 1; W[11, col] += 1; col += 1
        U[7, col] += 1; V[7, col] += 1; W[11, col] += 1; col += 1
        
        # Part 5: A_BR (1x1) * B_bot (1x4)
        U[8, col] += 1; V[8, col] += 1; W[2, col] += 1; col += 1
        U[8, col] += 1; V[9, col] += 1; W[5, col] += 1; col += 1
        U[8, col] += 1; V[10, col] += 1; W[8, col] += 1; col += 1
        U[8, col] += 1; V[11, col] += 1; W[11, col] += 1; col += 1
        
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # First, secure the known rank 34 solution as a fallback
    known_solution = try_known_rank_34()
    
    # Strategy 2: Try to find a rank < 34 using optimization
    # We search from rank 25 up to 33
    # If we find one, it's better than 34.
    
    def loss_and_grad(params, U_shape, V_shape, W_shape, r):
        # Unpack
        num_u = U_shape[0] * r
        num_v = V_shape[0] * r
        num_w = W_shape[0] * r
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:num_u+num_v+num_w]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Iterate over valid indices
        for (a, b, g) in valid_indices:
            val = T_approx[a, b, g]
            err = val - 1.0
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
        # Iterate over invalid indices
        for (a, b, g) in invalid_indices:
            val = T_approx[a, b, g]
            err = val - 0.0
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s, V_s, W_s = snap_to_half_integers(U, V, W)
        T_approx = np.einsum('ac,bc,dc->abd', U_s, V_s, W_s)
        
        for (a, b, g) in valid_indices:
            if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-9):
                return False
        for (a, b, g) in invalid_indices:
            if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-9):
                return False
        return True

    # Optimization loop
    # We limit the time spent on optimization to ensure we return the known solution if needed
    opt_start = time.time()
    opt_time_limit = 60 # Spend max 60s on optimization
    
    for r in range(25, 34): # Try ranks 25 to 33
        if time.time() - opt_start > opt_time_limit:
            break
        if time.time() - start_time > timeout:
            break
            
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        found = False
        # Number of restarts
        # We reduce restarts as time gets tight
        max_restarts = 10
        if time.time() - opt_start > 30:
            max_restarts = 5
        if time.time() - opt_start > 50:
            max_restarts = 2

        for attempt in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Random initialization with small integers
            U = np.random.choice([-1, 0, 1], size=U_shape)
            V = np.random.choice([-1, 0, 1], size=V_shape)
            W = np.random.choice([-1, 0, 1], size=W_shape)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, args=(U_shape, V_shape, W_shape, r), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 2000})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Check exactness after snapping
                num_u = U_shape[0] * r
                num_v = V_shape[0] * r
                num_w = W_shape[0] * r
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                if snap_and_check(U_final, V_final, W_final):
                    return {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    # If optimization didn't find a better rank, return the known rank 34 solution
    if known_solution is not None:
        return known_solution

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
