"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Build the target multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[i*m+j, j*p+k, k*n+i] = 1 if and only if the indices match appropriately
    # For matrix mult A(n,m) * B(m,p) = C(n,p): C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1 (for valid i,j,k)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U, dtype=float) if isinstance(U, list) else U
        V_arr = np.array(V, dtype=float) if isinstance(V, list) else V
        W_arr = np.array(W, dtype=float) if isinstance(W, list) else W
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_recon += np.outer(np.outer(U_arr[:,t], V_arr[:,t]), W_arr[:,t])
        
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, current_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), current_rank
    
    def try_l_bfgs(target_rank, num_trials=5, max_iter=500):
        """Try to find a rank-target_rank decomposition using L-BFGS-B."""
        if target_rank >= current_rank:
            return None
        
        from scipy.optimize import minimize
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(trial * 1000 + 42)
            # Sparse random initialization
            U_t = np.random.randn(n*m, target_rank) * 0.5
            V_t = np.random.randn(m*p, target_rank) * 0.5
            W_t = np.random.randn(p*n, target_rank) * 0.5
            
            # Flatten parameters
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                T_recon = np.zeros((n*m, m*p, p*n))
                for t in range(target_rank):
                    T_recon += np.outer(np.outer(U_flat[:,t], V_flat[:,t]), W_flat[:,t])
                
                error = T_target - T_recon
                return np.sum(error**2)
            
            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                T_recon = np.zeros((n*m, m*p, p*n))
                for t in range(target_rank):
                    T_recon += np.outer(np.outer(U_flat[:,t], V_flat[:,t]), W_flat[:,t])
                
                error = T_target - T_recon
                
                dU = np.zeros_like(U_flat)
                dV = np.zeros_like(V_flat)
                dW = np.zeros_like(W_flat)
                for t in range(target_rank):
                    dU[:,t] = np.einsum('abc,bc->a', error, V_flat[:,t] * W_flat[:,t])
                    dV[:,t] = np.einsum('abc,ac->b', error, U_flat[:,t] * W_flat[:,t])
                    dW[:,t] = np.einsum('abc,ab->c', error, U_flat[:,t] * V_flat[:,t])
                
                return np.concatenate([-2*dU.flatten(), -2*dV.flatten(), -2*dW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                                 options={'maxiter': max_iter, 'gtol': 1e-10, 'maxfun': 2000})
            except Exception:
                continue
            
            if time.time() > deadline:
                return None
            
            # Check if we found a good solution
            U_final = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_final = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_final = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            if check_decomposition(U_final, V_final, W_final, tol=1e-8):
                return U_final, V_final, W_final, target_rank
        
        return None
    
    # Try progressively lower ranks using L-BFGS-B
    # Theoretical lower bound for 3x3x6 is not well-known, but let's try from 45 downward
    for target_rank in range(45, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        result = try_l_bfgs(target_rank, num_trials=3, max_iter=300)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    U_final, V_final, W_final, r_final = best_U, best_V, best_W, best_rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        """Convert numpy array to list of lists."""
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Try to convert to fraction
                frac = Fraction(val).limit_denominator(1000)
                if abs(frac - val) < 1e-9:
                    # Check if it's integer or half-integer
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
