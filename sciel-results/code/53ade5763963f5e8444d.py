import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # Build the target tensor T
    # T[u, v, w] = 1 if consistent, else 0
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0

    # Flatten T for optimization
    T_flat = T.flatten()
    
    def objective(params):
        # params: flattened U, V, W
        # U: num_u * R
        # V: num_v * R
        # W: num_w * R
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return 0.0
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        # Compute tensor product sum
        # T_calc[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # This is equivalent to:
        # T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        T_calc_flat = T_calc.flatten()
        
        diff = T_calc_flat - T_flat
        return np.dot(diff, diff)

    def check_solution(params):
        R = len(params) // (num_u + num_v + num_w)
        if R == 0:
            return False
        u_size = num_u * R
        v_size = num_v * R
        w_size = num_w * R
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size+v_size]
        W_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape(num_u, R)
        V = V_flat.reshape(num_v, R)
        W = W_flat.reshape(num_w, R)
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        return np.linalg.norm(diff) < 1e-5

    # Try ranks
    start_time = time.time()
    max_rank = 63 # Schoolbook
    deadline = 140.0 # Safety margin
    
    for R in range(21, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Initialize
        np.random.seed(42 + R)
        U_init = np.random.randn(num_u, R)
        V_init = np.random.randn(num_v, R)
        W_init = np.random.randn(num_w, R)
        
        # Normalize to help optimization
        # We can scale factors, but let's just start.
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Run optimization
        # Use L-BFGS-B or similar
        # Since we want integer/half-integer, we might need to be careful.
        # But first find a real solution.
        
        # If R is small, the problem is underdetermined or just right.
        # For R=21, num vars = 21*(9+21+21) = 21*51 = 1071.
        # Equations: 9*21*21 = 3969.
        # So for R=21, we have more equations than variables.
        # It might not have a solution.
        
        # Let's try to solve the linear system for fixed U, V?
        # No, it's trilinear.
        
        # Let's use a simple gradient descent or L-BFGS.
        # To speed up, we can try multiple restarts.
        
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Scale to avoid large numbers
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or res.fun < 1e-4:
                if check_solution(res.x):
                    # Try to round to half-integers
                    U_opt = res.x[:num_u*R].reshape(num_u, R)
                    V_opt = res.x[num_u*R:num_u*R+num_v*R].reshape(num_v, R)
                    W_opt = res.x[num_u*R+num_v*R:].reshape(num_w, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Check if rounded solution is valid
                    T_calc_round = np.einsum('ur, vr, wr -> uvw', U_round, V_round, W_round)
                    diff_round = T_calc_round - T
                    if np.linalg.norm(diff_round) < 1e-5:
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                    else:
                        # If rounding fails, return floats? 
                        # The task says "integer/half-integer entries".
                        # If we can't find one, we might need to search more.
                        # But for now, if we have a good real solution, 
                        # we can try to find an integer one via local search?
                        # Or just return the real one if the evaluator accepts it?
                        # "exactly decompose" implies exact.
                        # Let's assume a solution exists with small integers.
                        
                        # Try to find integer solution by rounding and checking
                        # If not, try next restart
                        pass
            
        # If not found for this R, continue to next R

    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
