import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_np(U, V, W, n, m, p):
        """Vectorized verification using NumPy"""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_half_int(arr, tol=0.01):
        """Snap values to nearest half-integer if within tolerance"""
        doubled = arr * 2.0
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            return rounded / 2.0
        return None
    
    def try_find_m332_vec(rank_target, max_iter=10000, lr=0.01, seed=42, n2=3, m2=3, p2=2):
        """
        Vectorized gradient descent for M_{3,3,2} with half-integer snapping.
        """
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n2 * m2, rank_target) * 0.5
        V = rng.randn(m2 * p2, rank_target) * 0.5
        W = rng.randn(p2 * n2, rank_target) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute index arrays for vectorization
        # For each (i,j,k), we need U[i*m2+j,:], V[j*p2+k,:], W[k*n2+i,:]
        # Build lookup tables
        idx_U = np.array([[i * m2 + j for j in range(m2)] for i in range(n2)])  # (n2, m2)
        idx_V = np.array([[j * p2 + k for k in range(p2)] for j in range(m2)])  # (m2, p2)
        idx_W = np.array([[k * n2 + i for i in range(n2)] for k in range(p2)])  # (p2, n2)
        
        for it in range(max_iter):
            # Compute all 18 (i,j,k) sums
            errors = np.zeros((n2, m2, p2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        errors[i, j, k] = np.dot(U[idx_U[i,j], :], V[idx_V[j,k], :] * W[idx_W[k,i], :]) - 1.0
            
            loss = np.sum(errors ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-14:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        e = 2.0 * errors[i, j, k]
                        dU[idx_U[i,j], :] += e * V[idx_V[j,k], :] * W[idx_W[k,i], :]
                        dV[idx_V[j,k], :] += e * U[idx_U[i,j], :] * W[idx_W[k,i], :]
                        dW[idx_W[k,i], :] += e * U[idx_U[i,j], :] * V[idx_V[j,k], :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if it % 500 == 0 and it > 0:
                lr *= 0.95
        
        # Try snapping to half-integers
        for scale in [1.0, 0.5, 2.0]:
            U_snap = snap_half_int(best_U * scale, tol=0.05)
            V_snap = snap_half_int(best_V * scale, tol=0.05)
            W_snap = snap_half_int(best_W * scale, tol=0.05)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                U_test = (U_snap / scale)
                V_test = (V_snap / scale)
                W_test = (W_snap / scale)
                # Verify
                valid = True
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            s = np.dot(U_test[idx_U[i,j], :], V_test[idx_V[j,k], :] * W_test[idx_W[k,i], :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    # Convert to list of floats (half-integers)
                    return U_test.tolist(), V_test.tolist(), W_test.tolist()
        
        # Also try pure integer rounding
        U_round = np.round(best_U)
        V_round = np.round(best_V)
        W_round = np.round(best_W)
        valid = True
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = np.dot(U_round[idx_U[i,j], :], V_round[idx_V[j,k], :] * W_round[idx_W[k,i], :])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None
    
    def try_find_m332_lbfgs(rank_target, seed=42, n2=3, m2=3, p2=2, max_iter=500):
        """
        Use L-BFGS-B from scipy for more robust optimization.
        """
        from scipy.optimize import minimize
        
        rng = np.random.RandomState(seed)
        
        def objective(x):
            U = x[:n2*m2*rank_target].reshape(n2*m2, rank_target)
            V = x[n2*m2*rank_target : n2*m2*rank_target + m2*p2*rank_target].reshape(m2*p2, rank_target)
            W = x[n2*m2*rank_target + m2*p2*rank_target :].reshape(p2*n2, rank_target)
            total = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = np.dot(U[i*m2+j, :], V[j*p2+k, :] * W[k*n2+i, :])
                        total += (s - 1.0) ** 2
            return total
        
        # Multiple random starts
        for attempt in range(5):
            U0 = rng.randn(n2 * m2, rank_target) * 0.3
            V0 = rng.randn(m2 * p2, rank_target) * 0.3
            W0 = rng.randn(p2 * n2, rank_target) * 0.3
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16})
            
            if res.success or res.fun < 1e-12:
                U = res.x[:n2*m2*rank_target].reshape(n2*m2, rank_target)
                V = res.x[n2*m2*rank_target : n2*m2*rank_target + m2*p2*rank_target].reshape(m2*p2, rank_target)
                W = res.x[n2*m2*rank_target + m2*p2*rank_target :].reshape(p2*n2, rank_target)
                
                # Try snapping
                for tol in [0.05, 0.1, 0.2]:
                    U_d = U * 2.0
                    V_d = V * 2.0
                    W_d = W * 2.0
                    U_r = np.round(U_d)
                    V_r = np.round(V_d)
                    W_r = np.round(W_d)
                    if (np.max(np.abs(U_d - U_r)) < tol and 
                        np.max(np.abs(V_d - V_r)) < tol and 
                        np.max(np.abs(W_d - W_r)) < tol):
                        U_s = U_r / 2.0
                        V_s = V_r / 2.0
                        W_s = W_r / 2.0
                        # Verify
                        valid = True
                        for i in range(n2):
                            for j in range(m2):
                                for k in range(p2):
                                    s = np.dot(U_s[i*m2+j, :], V_s[j*p2+k, :] * W_s[k*n2+i, :])
                                    if abs(s - 1.0) > 1e-9:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            return U_s.tolist(), V_s.tolist(), W_s.tolist()
                
                # Try integer
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                valid = True
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            s = np.dot(U_i[i*m2+j, :], V_i[j*p2+k, :] * W_i[k*n2+i, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    return U_i.tolist(), V_i.tolist(), W_i.tolist()
        
        return None
    
    def combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine two block decompositions:
        - Block 1: M_{n,m,p1} with rank r1
        - Block 2: M_{n,m,p2} with rank r2
        Result: M_{n,m,p1+p2} with rank r1+r2
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank_total = r1 + r2
        p_total = p1 + p2
        
        U_full = [[0] * rank_total for _ in range(n * m)]
        V_full = [[0] * rank_total for _ in range(m * p_total)]
        W_full = [[0] * rank_total for _ in range(p_total * n)]
        
        # Copy block 1
        for i in range(n * m):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        
        for j in range(m):
            for k in range(p1):
                for r in range(r1):
                    V_full[j * p_total + k][r] = V1[j * p1 + k][r]
        
        for k in range(p1):
            for i in range(n):
                for r in range(r1):
                    W_full[k * n + i][r] = W1[k * n + i][r]
        
        # Copy block 2 (offset in rank dimension)
        for i in range(n * m):
            for r in range(r2):
                U_full[i][r1 + r] = U2[i][r]
        
        for j in range(m):
            for k in range(p2):
                for r in range(r2):
                    V_full[j * p_total + p1 + k][r1 + r] = V2[j * p2 + k][r]
        
        for k in range(p2):
            for i in range(n):
                for r in range(r2):
                    W_full[(p1 + k) * n + i][r1 + r] = W2[k * n + i][r]
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    start_time = time.time()
    deadline = 140  # seconds (safety margin before 150s kill)
    
    # Start with schoolbook
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
    # Schoolbook M_{3,3,3} has rank 27
    # If we find M_{3,3,2} with rank r2 < 18, total rank 27 + r2 < 45
    
    sb333 = schoolbook(3, 3, 3)
    U333, V333, W333 = sb333["U"], sb333["V"], sb333["W"]
    
    # Try to find low-rank M_{3,3,2}
    found_m332 = False
    for r2 in range(12, 18):  # Try ranks 12 through 17
        if time.time() - start_time > deadline - 20:
            break
        
        # Try vectorized gradient descent with multiple seeds
        for seed in range(r2, r2 + 3):
            if time.time() - start_time > deadline - 15:
                break
            res = try_find_m332_vec(r2, max_iter=5000, lr=0.02, seed=seed)
            if res:
                U2, V2, W2 = res
                result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
                if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                    best_result = result
                    best_rank = 27 + r2
                    found_m332 = True
                    break
        if found_m332:
            break
        
        # Try L-BFGS-B
        if time.time() - start_time > deadline - 10:
            break
        res = try_find_m332_lbfgs(r2, seed=r2 + 50)
        if res:
            U2, V2, W2 = res
            result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
            if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                best_result = result
                best_rank = 27 + r2
                found_m332 = True
                break
    
    return best_result
# EVOLVE-BLOCK-END
