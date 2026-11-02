import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem."""
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def newton_polish(x, n=2, m=4, p=4, rank=24, max_iter=50, tol=1e-12):
    """Newton's method polish for the polynomial system."""
    x = x.copy()
    for it in range(max_iter):
        if time_up():
            break
        r = residuals(x, n, m, p, rank)
        if np.linalg.norm(r) < tol:
            break
        # Use Levenberg-Marquardt style update
        size = len(x)
        # Approximate Jacobian with a few random directions for speed
        J = np.zeros((32, size))
        step = 1e-6 * (1 + np.linalg.norm(x) / size)
        for idx in range(0, size, max(1, size // 200)):
            x_pert = x.copy()
            x_pert[idx] += step
            r_pert = residuals(x_pert, n, m, p, rank)
            J[:, idx] = (r_pert - r) / step
        # Solve (J^T J + lambda I) dx = -J^T r
        JtJ = J.T @ J
        Jtr = J.T @ r
        lam = 1e-8 * np.trace(JtJ) / max(1, size)
        try:
            dx = np.linalg.solve(JtJ + lam * np.eye(size), -Jtr)
        except np.linalg.LinAlgError:
            dx = -0.01 * r.reshape(size)
        x_new = x + dx
        r_new = residuals(x_new, n, m, p, rank)
        if np.linalg.norm(r_new) < np.linalg.norm(r):
            x = x_new
        else:
            lam *= 10
    return x

def try_rank_reduction_from_strassen(target_rank, n=2, m=4, p=4, max_restarts=20, prev_solution=None):
    """Try to reduce rank from a previous solution to target_rank."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Start with previous solution or Strassen as base
    if prev_solution is not None:
        U_s = np.array(prev_solution['U'], dtype=float)
        V_s = np.array(prev_solution['V'], dtype=float)
        W_s = np.array(prev_solution['W'], dtype=float)
    else:
        U_strassen, V_strassen, W_strassen = strassen_decomposition()
        U_s = np.array(U_strassen, dtype=float)
        V_s = np.array(V_strassen, dtype=float)
        W_s = np.array(W_strassen, dtype=float)
    
    src_rank = U_s.shape[1]
    
    # For rank reduction, try multiple strategies for initialization
    strategies = []
    
    # Strategy 1: Direct truncation (keep first target_rank columns)
    if target_rank <= src_rank:
        U_init = U_s[:, :target_rank].copy()
        V_init = V_s[:, :target_rank].copy()
        W_init = W_s[:, :target_rank].copy()
        strategies.append((U_init, V_init, W_init))
    
    # Strategy 2: Truncation with small perturbation
    if target_rank <= src_rank:
        for pert_scale in [0.01, 0.05, 0.1]:
            U_pert = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * pert_scale
            V_pert = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * pert_scale
            W_pert = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * pert_scale
            strategies.append((U_pert, V_pert, W_pert))
    
    # Strategy 3: Random sparse initialization
    for sparse_density in [0.25, 0.5]:
        U_rand = np.zeros((n*m, target_rank))
        V_rand = np.zeros((m*p, target_rank))
        W_rand = np.zeros((p*n, target_rank))
        for r in range(target_rank):
            u_rows = np.random.choice(n*m, size=max(1, int(n*m * sparse_density)), replace=False)
            v_rows = np.random.choice(m*p, size=max(1, int(m*p * sparse_density)), replace=False)
            w_rows = np.random.choice(p*n, size=max(1, int(p*n * sparse_density)), replace=False)
            U_rand[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows))
            V_rand[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows))
            W_rand[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows))
        strategies.append((U_rand, V_rand, W_rand))
    
    # Strategy 4: Full random
    for scale in [0.1, 0.5, 1.0]:
        U_rand = np.random.randn(n*m, target_rank) * scale
        V_rand = np.random.randn(m*p, target_rank) * scale
        W_rand = np.random.randn(p*n, target_rank) * scale
        strategies.append((U_rand, V_rand, W_rand))
    
    # Shuffle strategies for diversity
    np.random.shuffle(strategies)
    
    for idx, (U_init, V_init, W_init) in enumerate(strategies):
        if time_up():
            break
        if idx >= max_restarts:
            break
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=3000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
        
        # If we got close, try Newton polish
        if best_error < 0.05 and not time_up():
            try:
                x_polished = newton_polish(best_result, n, m, p, target_rank, max_iter=30, tol=1e-12)
                err_polished = np.linalg.norm(residuals(x_polished, n, m, p, target_rank))
                if err_polished < best_error:
                    best_error = err_polished
                    best_result = x_polished
                if err_polished < 1e-9:
                    U = best_result[:size_U].reshape(n*m, target_rank)
                    V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                    W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                    valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
            except Exception:
                pass
            
    # Final polish pass if we got close
    if best_result is not None and best_error < 0.1 and not time_up():
        try:
            result2 = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                         method='trf', max_nfev=5000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            err2 = np.linalg.norm(result2.fun)
            if err2 < 1e-9:
                U = result2.x[:size_U].reshape(n*m, target_rank)
                V = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=30):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Use structured random initialization based on tensor structure
        if restart == 0:
            # Initialize with sparse structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Randomly assign non-zero entries
                u_rows = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                v_rows = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                w_rows = np.random.choice(p*n, size=max(1, p*n//4), replace=False)
                U[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows))
                V[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows))
                W[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        elif restart < 10:
            # Random initialization with varying scales
            scale = 0.1 + 0.05 * (restart % 5)
            x0 = np.random.randn(total_size) * scale
        else:
            # More structured: use Strassen-like patterns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Each rank-1 term covers a random subset
                u_mask = np.random.rand(n*m) < 0.3
                v_mask = np.random.rand(m*p) < 0.3
                w_mask = np.random.rand(p*n) < 0.3
                U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
                V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
                W[w_mask, r] = np.random.choice([-1, 1], size=np.sum(w_mask))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                # Found a solution
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Verify
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
        
        # Try Newton polish if close
        if best_error < 0.05 and not time_up():
            try:
                x_polished = newton_polish(best_result, n, m, p, target_rank, max_iter=20, tol=1e-12)
                err_polished = np.linalg.norm(residuals(x_polished, n, m, p, target_rank))
                if err_polished < best_error:
                    best_error = err_polished
                    best_result = x_polished
                if err_polished < 1e-9:
                    U = best_result[:size_U].reshape(n*m, target_rank)
                    V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                    W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                    valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
            except Exception:
                pass
    
    # Polish pass if we got close
    if best_result is not None and best_error < 0.1 and not time_up():
        try:
            result2 = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                         method='trf', max_nfev=5000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            err2 = np.linalg.norm(result2.fun)
            if err2 < 1e-9:
                U = result2.x[:size_U].reshape(n*m, target_rank)
                V = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Greedy sequential rank reduction: 28 -> 27 -> 26 -> 25 -> 24
    # Each step uses the previous solution as warm start
    prev_sol = best_result
    
    # Focus more effort on rank 27 (the critical first reduction)
    for target_rank in [27, 27, 26, 25, 24]:
        if time_up():
            break
        if prev_sol is None:
            break
        
        # For rank 27, give it more restarts since it's the hardest step
        max_restarts = 30 if target_rank == 27 else 20
        
        res = try_rank_reduction_from_strassen(target_rank, n, m, p, 
                                               max_restarts=max_restarts, 
                                               prev_solution=prev_sol)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = target_rank
                best_result = {"U": U2, "V": V2, "W": W2}
                prev_sol = best_result
            else:
                pass
        else:
            # Try random search as fallback for this rank
            if not time_up():
                res = find_low_rank_decomposition(target_rank, n, m, p, max_restarts=20)
                if res is not None:
                    U2, V2, W2 = res
                    valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                    if valid:
                        best_rank = target_rank
                        best_result = {"U": U2, "V": V2, "W": W2}
                        prev_sol = best_result
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
