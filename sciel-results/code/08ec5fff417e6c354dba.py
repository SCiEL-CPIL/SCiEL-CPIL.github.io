import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit

    def schoolbook():
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

    def compute_loss(params, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U3 = U_flat.reshape(n, m, r)
        V3 = V_flat.reshape(m, p, r)
        W3 = W_flat.reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        diff = S - 1.0
        return np.sum(diff * diff)

    def verify(U_list, V_list, W_list):
        n, m, p = DIMENSIONS
        r = len(U_list[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U_list[i*m+j][r_idx] * V_list[j*p+k][r_idx] * W_list[k*n+i][r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def discrete_polish(U_flat, V_flat, W_flat, r, max_steps=1000):
        """
        Try to improve the discrete solution by flipping nearby values.
        """
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        def get_loss():
            U3 = U_flat.reshape(n, m, r)
            V3 = V_flat.reshape(m, p, r)
            W3 = W_flat.reshape(p, n, r)
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            return np.sum((S - 1.0)**2)
        
        current_loss = get_loss()
        if current_loss < 1e-9:
            return True, U_flat, V_flat, W_flat

        for _ in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            # Pick a random variable
            var_type = random.randint(0, 2)
            if var_type == 0:
                size = n * m * r
                idx = random.randint(0, size - 1)
                old_val = U_flat[idx]
                # Pick a neighbor
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                U_flat[idx] = new_val
                new_loss = get_loss()
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-9:
                        return True, U_flat, V_flat, W_flat
                else:
                    U_flat[idx] = old_val
            elif var_type == 1:
                size = m * p * r
                idx = random.randint(0, size - 1)
                old_val = V_flat[idx]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                V_flat[idx] = new_val
                new_loss = get_loss()
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-9:
                        return True, U_flat, V_flat, W_flat
                else:
                    V_flat[idx] = old_val
            else:
                size = p * n * r
                idx = random.randint(0, size - 1)
                old_val = W_flat[idx]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                W_flat[idx] = new_val
                new_loss = get_loss()
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-9:
                        return True, U_flat, V_flat, W_flat
                else:
                    W_flat[idx] = old_val
        return False, U_flat, V_flat, W_flat

    def try_rank(r, max_restarts=5):
        if time.time() > deadline - 10:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            size_U = n * m * r
            size_V = m * p * r
            size_W = p * n * r
            total_size = size_U + size_V + size_W
            
            # Initialize with small random values
            init_vals = np.random.normal(0, 0.5, size=total_size)
            
            # Run optimization
            try:
                result = minimize(compute_loss, init_vals, args=(r,), method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
            except Exception:
                continue
            
            if result.fun < 1e-4: # Threshold for "close enough" to try rounding
                params = result.x
                U_flat = np.round(params[:size_U] * 2) / 2
                V_flat = np.round(params[size_U:size_U + size_V] * 2) / 2
                W_flat = np.round(params[size_U + size_V:] * 2) / 2
                
                # Check if direct rounding works
                U3 = U_flat.reshape(n, m, r)
                V3 = V_flat.reshape(m, p, r)
                W3 = W_flat.reshape(p, n, r)
                S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                
                if np.allclose(S, 1.0, atol=1e-9):
                    U_list = [[U_flat[(i*m+j)*r + rr] for rr in range(r)] for i in range(n) for j in range(m)]
                    V_list = [[V_flat[(j*p+k)*r + rr] for rr in range(r)] for j in range(m) for k in range(p)]
                    W_list = [[W_flat[(k*n+i)*r + rr] for rr in range(r)] for k in range(p) for i in range(n)]
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Try discrete polish
                    success, U_f, V_f, W_f = discrete_polish(U_flat, V_flat, W_flat, r)
                    if success:
                        U_list = [[U_f[(i*m+j)*r + rr] for rr in range(r)] for i in range(n) for j in range(m)]
                        V_list = [[V_f[(j*p+k)*r + rr] for rr in range(r)] for j in range(m) for k in range(p)]
                        W_list = [[W_f[(k*n+i)*r + rr] for rr in range(r)] for k in range(p) for i in range(n)]
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}

    # Search for minimal rank
    for r in range(20, 36):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result:
            return result

    return schoolbook()
# EVOLVE-BLOCK-END
