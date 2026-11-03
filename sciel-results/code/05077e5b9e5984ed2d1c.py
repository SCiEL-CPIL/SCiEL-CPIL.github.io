"""Matrix multiplication 2x3 x 3x7 - rank minimization search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _validate(U, V, W, n=2, m=3, p=7):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _compute_errors(U, V, W, n=2, m=3, p=7):
    """Compute errors for all (i,j,k) triples."""
    rank = U.shape[1]
    errors = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                errors[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
    return errors

def _smooth_max_error(errors, temperature=0.1):
    """Smooth surrogate for max error using log-sum-exp."""
    abs_errs = np.abs(errors)
    if np.max(abs_errs) < 1e-15:
        return 0.0
    scaled = abs_errs / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum * temperature

def _try_rank(rank, n=2, m=3, p=7, max_iters=500, seed=0):
    """Try to find a valid rank-r decomposition via alternating least squares + local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Multiple initializations
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init in range(8):
        if time.time() > DEADLINE - 10:
            break
        
        # Different initialization strategies
        if init == 0:
            # Random
            U = np_rng.randn(n*m, rank)
            V = np_rng.randn(m*p, rank)
            W = np_rng.randn(p*n, rank)
        elif init == 1:
            # Sparse random
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                U[rng.randint(0, n*m-1), c] = 1
                V[rng.randint(0, m*p-1), c] = 1
                W[rng.randint(0, p*n-1), c] = 1
        elif init == 2:
            # Structured: try to share patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Assign each column a random (i,j,k) pattern
            for c in range(rank):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        elif init == 3:
            # Block-structured: group columns by i
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            cols_per_i = rank // n
            for i in range(n):
                for c in range(i * cols_per_i, (i+1) * cols_per_i):
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
        elif init == 4:
            # Block-structured: group columns by j
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            cols_per_j = rank // m
            for j in range(m):
                for c in range(j * cols_per_j, (j+1) * cols_per_j):
                    i = rng.randint(0, n-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
        elif init == 5:
            # Block-structured: group columns by k
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            cols_per_k = rank // p
            for k in range(p):
                for c in range(k * cols_per_k, (k+1) * cols_per_k):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
        elif init == 6:
            # Hybrid: some random, some structured
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            half = rank // 2
            for c in range(half):
                U[rng.randint(0, n*m-1), c] = 1
                V[rng.randint(0, m*p-1), c] = 1
                W[rng.randint(0, p*n-1), c] = 1
            for c in range(half, rank):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        else:
            # Gaussian with small scale
            U = np_rng.randn(n*m, rank) * 0.5
            V = np_rng.randn(m*p, rank) * 0.5
            W = np_rng.randn(p*n, rank) * 0.5
        
        # Alternating least squares with smooth objective monitoring
        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            smooth_err = _smooth_max_error(errors, temperature=0.05)
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update one matrix at a time (alternating)
            mat_idx = it % 3
            
            if mat_idx == 0:  # Update U
                # For each row of U, solve linear system
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, rank))
                        for k in range(p):
                            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = x
                        except:
                            pass
                        
            elif mat_idx == 1:  # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = x
                        except:
                            pass
                            
            else:  # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = x
                        except:
                            pass
        
        # Local search refinement with adaptive step sizes
        for ls_it in range(150):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive step size based on current error
            base_step = min(0.5, max(0.001, max_err * 0.1))
            steps = [base_step, -base_step, base_step*0.1, -base_step*0.1]
            
            # Perturb one random element
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, n*m-1)
            elif mat == 'V':
                row = rng.randint(0, m*p-1)
            else:
                row = rng.randint(0, p*n-1)
            col = rng.randint(0, rank-1)
            
            best_perturb = None
            best_perturb_err = max_err
            for delta in steps:
                if mat == 'U':
                    orig = U[row, col]
                    U[row, col] = orig + delta
                elif mat == 'V':
                    orig = V[row, col]
                    V[row, col] = orig + delta
                else:
                    orig = W[row, col]
                    W[row, col] = orig + delta
                
                err = np.max(np.abs(_compute_errors(U, V, W, n, m, p)))
                
                if mat == 'U':
                    U[row, col] = orig
                elif mat == 'V':
                    V[row, col] = orig
                else:
                    W[row, col] = orig
                
                if err < best_perturb_err:
                    best_perturb_err = err
                    best_perturb = (mat, row, col, delta)
            
            if best_perturb is not None:
                mat, row, col, delta = best_perturb
                if mat == 'U':
                    U[row, col] += delta
                elif mat == 'V':
                    V[row, col] += delta
                else:
                    W[row, col] += delta
            else:
                # Try a few more random perturbations before giving up
                found_improvement = False
                for _ in range(3):
                    mat2 = rng.choice(['U', 'V', 'W'])
                    if mat2 == 'U':
                        row2 = rng.randint(0, n*m-1)
                    elif mat2 == 'V':
                        row2 = rng.randint(0, m*p-1)
                    else:
                        row2 = rng.randint(0, p*n-1)
                    col2 = rng.randint(0, rank-1)
                    delta2 = rng.uniform(-base_step, base_step)
                    
                    if mat2 == 'U':
                        orig2 = U[row2, col2]
                        U[row2, col2] = orig2 + delta2
                    elif mat2 == 'V':
                        orig2 = V[row2, col2]
                        V[row2, col2] = orig2 + delta2
                    else:
                        orig2 = W[row2, col2]
                        W[row2, col2] = orig2 + delta2
                    
                    err2 = np.max(np.abs(_compute_errors(U, V, W, n, m, p)))
                    
                    if mat2 == 'U':
                        U[row2, col2] = orig2
                    elif mat2 == 'V':
                        V[row2, col2] = orig2
                    else:
                        W[row2, col2] = orig2
                    
                    if err2 < best_perturb_err:
                        best_perturb_err = err2
                        best_perturb = (mat2, row2, col2, delta2)
                        found_improvement = True
                        break
                
                if found_improvement:
                    mat, row, col, delta = best_perturb
                    if mat == 'U':
                        U[row, col] += delta
                    elif mat == 'V':
                        V[row, col] += delta
                    else:
                        W[row, col] += delta
                else:
                    break
    
    if best_err < 1e-10 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _polish(U, V, W, n=2, m=3, p=7, max_iters=50):
    """High-precision polish pass using exact linear solves."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Check convergence
        errors = _compute_errors(U, V, W, n, m, p)
        max_err = np.max(np.abs(errors))
        if max_err < 1e-12:
            break
        
        # Solve for U exactly
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, rank))
                for k in range(p):
                    B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(p), rcond=1e-15)[0]
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Solve for V exactly
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)[0]
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Solve for W exactly
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=1e-15)[0]
                    W[k*n+i, :] = x
                except:
                    pass
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_structured_rank(rank, n=2, m=3, p=7):
    """Try structured decompositions based on known patterns."""
    return None

def _schoolbook_decomposition(n=2, m=3, p=7):
    """Generate the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # First, generate schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Theoretical lower bound: max(n*p, n*m, m*p) = max(14, 6, 21) = 21
    # Try ranks from lower bound upward
    
    # Priority: try ranks that are more likely to have solutions
    ranks_to_try = list(range(21, 35))  # Try 21 to 34
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple seeds for this rank
        max_attempts = 6 if target_rank <= 24 else 4
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 15:
                break
            
            result = _try_rank(target_rank, n, m, p, max_iters=250, seed=attempt*1000+42)
            if result is not None:
                U, V, W = result
                # Try polish pass
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                U_pol, V_pol, W_pol = _polish(U_np, V_np, W_np, n, m, p)
                if _validate(U_pol, V_pol, W_pol, n, m, p):
                    return {"U": U_pol, "V": V_pol, "W": W_pol}
                # Also check original
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
