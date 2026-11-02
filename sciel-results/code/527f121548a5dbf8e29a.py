"""Rewritten program for 3x3 x 3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor for 3x3 x 3x6
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Build target tensor for 3x3 x 3x3
    T_333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T_333[a, b, c] = 1.0

    def check_decomposition(U, V, W, T, tol=1e-8):
        """Check if U, V, W reconstruct T."""
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol, rtol=0)

    def find_333_decomposition(target_rank=23, num_trials=50, max_iter=500):
        """Find a rank-target_rank decomposition for 3x3x3 matrix multiplication."""
        if time.time() > deadline:
            return None
        
        from scipy.optimize import minimize
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(trial * 104729 + 13)
            U_t = np.random.randn(9, target_rank) * 0.5
            V_t = np.random.randn(9, target_rank) * 0.5
            W_t = np.random.randn(9, target_rank) * 0.5
            
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_f = x[:9*target_rank].reshape(9, target_rank)
                V_f = x[9*target_rank:18*target_rank].reshape(9, target_rank)
                W_f = x[18*target_rank:].reshape(9, target_rank)
                
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_333 - T_recon
                return np.sum(error ** 2)
            
            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_f = x[:9*target_rank].reshape(9, target_rank)
                V_f = x[9*target_rank:18*target_rank].reshape(9, target_rank)
                W_f = x[18*target_rank:].reshape(9, target_rank)
                
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_333 - T_recon
                
                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': 2000, 'ftol': 1e-14})
            except Exception:
                continue
            
            if time.time() > deadline:
                return None
                
            U_final = result.x[:9*target_rank].reshape(9, target_rank)
            V_final = result.x[9*target_rank:18*target_rank].reshape(9, target_rank)
            W_final = result.x[18*target_rank:].reshape(9, target_rank)
            
            if check_decomposition(U_final, V_final, W_final, T_333, tol=1e-7):
                return U_final, V_final, W_final
        
        return None

    # Step 1: Find a rank-23 decomposition for 3x3x3
    dec_333 = find_333_decomposition(target_rank=23)
    
    if dec_333 is not None:
        U_333, V_333, W_333 = dec_333
        # Construct rank-46 decomposition for 3x3x6
        # The 3x3x6 tensor is two copies of 3x3x3.
        # We need to map the indices correctly.
        # U is (9, 46), V is (18, 46), W is (18, 46)
        
        U_46 = np.zeros((9, 46))
        V_46 = np.zeros((18, 46))
        W_46 = np.zeros((18, 46))
        
        # First copy (k=0,1,2)
        # U indices: same (0..8)
        # V indices: j*3+k -> j*6+k (for k=0,1,2)
        # W indices: k*3+i -> k*3+i (for k=0,1,2)
        
        U_46[:, 0:23] = U_333
        # V mapping: row (j*3+k) in 333 maps to row (j*6+k) in 336
        for j in range(3):
            for k in range(3):
                V_46[j*6+k, 0:23] = V_333[j*3+k, 0:23]
        # W mapping: row (k*3+i) in 333 maps to row (k*3+i) in 336 (since k<3, k*3+i < 9)
        W_46[0:9, 0:23] = W_333
        
        # Second copy (k=3,4,5)
        # Let k' = k-3 (0,1,2)
        # U indices: same (0..8)
        # V indices: j*3+k' in 333 maps to j*6+(k'+3) in 336
        # W indices: k'*3+i in 333 maps to (k'+3)*3+i = 9 + k'*3+i in 336
        
        U_46[:, 23:46] = U_333
        for j in range(3):
            for k_prime in range(3):
                V_46[j*6+k_prime+3, 23:46] = V_333[j*3+k_prime, 23:46]
        # W mapping: row (k'*3+i) in 333 maps to row (9 + k'*3+i) in 336
        W_46[9:18, 23:46] = W_333
        
        # Verify
        if check_decomposition(U_46, V_46, W_46, T_target, tol=1e-7):
            best_U, best_V, best_W, best_rank = U_46, V_46, W_46, 46
        else:
            # Fallback to schoolbook
            best_U, best_V, best_W, best_rank = None, None, None, 54
    else:
        # If we couldn't find rank 23 for 333, try rank 27 (schoolbook) for 333
        # Then construct rank 54 for 336
        U_333 = np.zeros((9, 27))
        V_333 = np.zeros((9, 27))
        W_333 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i*3+j, col] = 1.0
                    V_333[j*3+k, col] = 1.0
                    W_333[k*3+i, col] = 1.0
                    col += 1
        
        U_46 = np.zeros((9, 54))
        V_46 = np.zeros((18, 54))
        W_46 = np.zeros((18, 54))
        
        U_46[:, 0:27] = U_333
        for j in range(3):
            for k in range(3):
                V_46[j*6+k, 0:27] = V_333[j*3+k, 0:27]
        W_46[0:9, 0:27] = W_333
        
        U_46[:, 27:54] = U_333
        for j in range(3):
            for k_prime in range(3):
                V_46[j*6+k_prime+3, 27:54] = V_333[j*3+k_prime, 27:54]
        W_46[9:18, 27:54] = W_333
        
        best_U, best_V, best_W, best_rank = U_46, V_46, W_46, 54

    # Step 2: Try to reduce rank from 46 (or 54) further
    # Use L-BFGS-B on the full 3x3x6 tensor starting from the best found so far
    def try_reduce_rank(U_init, V_init, W_init, init_rank, target_rank, num_trials=10, max_iter=500):
        if target_rank >= init_rank:
            return None
        if time.time() > deadline:
            return None
            
        from scipy.optimize import minimize
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
                
            # Initialize with the known decomposition, then add random small noise to new dimensions
            # Or just truncate and add noise?
            # Better: Start with U_init, V_init, W_init and project to lower rank?
            # No, just initialize x0 with the first target_rank columns of the init, plus noise.
            
            U_t = U_init[:, :target_rank].copy()
            V_t = V_init[:, :target_rank].copy()
            W_t = W_init[:, :target_rank].copy()
            
            # Add small noise to escape local minima
            U_t += np.random.randn(9, target_rank) * 0.01
            V_t += np.random.randn(18, target_rank) * 0.01
            W_t += np.random.randn(18, target_rank) * 0.01
            
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_f = x[:9*target_rank].reshape(9, target_rank)
                V_f = x[9*target_rank:9*target_rank+18*target_rank].reshape(18, target_rank)
                W_f = x[9*target_rank+18*target_rank:].reshape(18, target_rank)
                
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                return np.sum(error ** 2)
            
            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_f = x[:9*target_rank].reshape(9, target_rank)
                V_f = x[9*target_rank:9*target_rank+18*target_rank].reshape(18, target_rank)
                W_f = x[9*target_rank+18*target_rank:].reshape(18, target_rank)
                
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                
                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': 1500, 'ftol': 1e-14})
            except Exception:
                continue
                
            if time.time() > deadline:
                return None
                
            U_final = result.x[:9*target_rank].reshape(9, target_rank)
            V_final = result.x[9*target_rank:9*target_rank+18*target_rank].reshape(18, target_rank)
            W_final = result.x[9*target_rank+18*target_rank:].reshape(18, target_rank)
            
            if check_decomposition(U_final, V_final, W_final, T_target, tol=1e-7):
                return U_final, V_final, W_final, target_rank
                
        return None

    # Try to reduce from best_rank
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        res = try_reduce_rank(best_U, best_V, best_W, best_rank, target_rank, num_trials=5, max_iter=300)
        if res is not None:
            best_U, best_V, best_W, best_rank = res

    # Convert to lists
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(1000)
                if abs(float(frac) - val) < 1e-9:
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

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
