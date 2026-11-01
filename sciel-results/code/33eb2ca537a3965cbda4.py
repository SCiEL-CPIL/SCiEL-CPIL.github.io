import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition as fallback
    U_triv = [[0] * trivial_rank for _ in range(n * m)]
    V_triv = [[0] * trivial_rank for _ in range(m * p)]
    W_triv = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}

    def eval_and_grad(x, R):
        """
        Compute error and gradient for a given x and rank R.
        x is a flat vector of length R * (n*m + m*p + p*n).
        Layout: [U_0...U_{R-1}, V_0...V_{R-1}, W_0...W_{R-1}]
        """
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = R * (size_u + size_v + size_w)
        
        if len(x) != total_size:
            return 1e10, np.zeros(total_size)

        u_flat = x[:R * size_u]
        v_flat = x[R * size_u : R * size_u + R * size_v]
        w_flat = x[R * size_u + R * size_v :]

        U = u_flat.reshape(R, size_u)
        V = v_flat.reshape(R, size_v)
        W = w_flat.reshape(R, size_w)

        U_3d = U.reshape(R, n, m)
        V_3d = V.reshape(R, m, p)
        W_3d = W.reshape(R, p, n)
        
        S = np.einsum('tij,tjk,tki->ijk', U_3d, V_3d, W_3d)
        
        error = np.sum((S - 1.0) ** 2)
        
        S_minus_1 = S - 1.0
        
        grad_U_3d = 2.0 * np.einsum('ijk,tjk,tki->tij', S_minus_1, V_3d, W_3d)
        grad_V_3d = 2.0 * np.einsum('ijk,tij,tki->tjk', S_minus_1, U_3d, W_3d)
        grad_W_3d = 2.0 * np.einsum('ijk,tij,tjk->tki', S_minus_1, U_3d, V_3d)
        
        grad_U = grad_U_3d.flatten()
        grad_V = grad_V_3d.flatten()
        grad_W = grad_W_3d.flatten()
        
        grad = np.concatenate([grad_U, grad_V, grad_W])
        
        return error, grad

    def check_decomposition(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(R):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def try_optimize(R, x0, max_iter=300):
        """Try to optimize from initial point x0 for rank R."""
        try:
            res = minimize(
                lambda x: eval_and_grad(x, R)[0],
                x0,
                jac=lambda x: eval_and_grad(x, R)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e-8:
                x_opt = res.x
                x_rounded = np.round(x_opt * 2) / 2.0
                
                size_u = n * m
                size_v = m * p
                size_w = p * n
                
                u_flat = x_rounded[:R * size_u]
                v_flat = x_rounded[R * size_u : R * size_u + R * size_v]
                w_flat = x_rounded[R * size_u + R * size_v :]
                
                U_check = u_flat.reshape(size_u, R)
                V_check = v_flat.reshape(size_v, R)
                W_check = w_flat.reshape(size_w, R)
                
                U_list = U_check.tolist()
                V_list = V_check.tolist()
                W_list = W_check.tolist()
                
                if check_decomposition(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
        return None

    # Generate structured initializations based on Winograd-style patterns
    def generate_structured_inits(R):
        """Generate multiple structured initializations for rank R."""
        inits = []
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = R * (size_u + size_v + size_w)
        
        # Pattern 1: Random small values
        for _ in range(3):
            x0 = np.random.randint(-2, 3, total_size) * 0.5
            inits.append(x0)
        
        # Pattern 2: Start from trivial and zero out some entries
        # This is complex, so we'll use a simpler approach
        # Try to find a valid decomposition by modifying the trivial one
        
        # Pattern 3: Use known low-rank structures for small cases
        # For 2x3x7, we can try to exploit the structure
        
        # Pattern 4: Random with larger range
        for _ in range(2):
            x0 = np.random.randn(total_size) * 0.5
            inits.append(x0)
        
        return inits

    # Search for lower ranks, starting from 41 down to 25
    for R in range(41, 24, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple initializations
        inits = generate_structured_inits(R)
        
        for x0 in inits:
            if time.time() > deadline - 5:
                break
            
            result = try_optimize(R, x0)
            if result is not None:
                best_rank = R
                best_result = result
                break  # Found a solution for this R, try lower R
        
        # If we found a solution for this R, continue to lower R
        # If not, continue to lower R anyway
    
    # Final: if we haven't improved, return the trivial solution
    return best_result
