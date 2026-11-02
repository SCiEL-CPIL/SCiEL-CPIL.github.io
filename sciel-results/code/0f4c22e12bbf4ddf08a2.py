import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p
    
    # Precompute indices for vectorized verification
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(res, np.ones(total_entries), atol=tol, rtol=0)
    
    def verify_half_integer(U, V, W):
        """Check if all entries are integers or half-integers"""
        for mat in [U, V, W]:
            doubled = mat * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-9):
                return False
        return True
    
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W
    
    def get_strassen_init(R, seed=0):
        """
        Initialize using Strassen-like patterns for 2x2x2 blocks.
        The 2x3x5 tensor can be decomposed into overlapping 2x2x2 sub-tensors.
        """
        rng = np.random.RandomState(seed)
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Strassen's 7 multiplications for 2x2x2
        strassen_patterns = [
            (1, 1, 1),   # a11*b11
            (1, 1, 1),   # a11*b22 (with signs)
            (1, -1, 1),  # a21*b12
            (1, 1, -1),  # a12*b21
            (1, 1, 1),   # a22*b22
            (1, -1, 1),  # (a11+a12)*b21
            (1, -1, 1),  # (a21+a22)*b12
        ]
        
        # For 2x3x5, we can tile Strassen blocks
        # Use structured initialization with some randomness
        for r in range(R):
            # Randomly assign which "type" of multiplication this rank-1 term represents
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            
            # Base assignment
            U[i*m+j, r] = rng.choice([-1.0, 1.0])
            V[j*p+k, r] = rng.choice([-1.0, 1.0])
            W[k*n+i, r] = rng.choice([-1.0, 1.0])
            
            # Add some structured components
            if r < 7:
                # First 7 ranks get Strassen-like structure
                pattern = strassen_patterns[r]
                U[0, r] += pattern[0]
                U[1, r] += pattern[0] if r < 4 else 0
                V[0, r] += pattern[1]
                V[1, r] += pattern[1] if r < 4 else 0
                W[0, r] += pattern[2]
                W[1, r] += pattern[2] if r < 4 else 0
        
        # Normalize to avoid numerical issues
        for mat in [U, V, W]:
            norm = np.linalg.norm(mat)
            if norm > 0:
                mat /= norm
        
        return U, V, W
    
    def try_find_rank(R, time_limit=10.0, seed=0):
        """
        Attempt to find a rank R decomposition using alternating least squares.
        Vectorized updates for speed.
        """
        if R >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        
        num_restarts = 20
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Choose initialization strategy
            if restart % 3 == 0:
                # Strassen-like initialization
                U, V, W = get_strassen_init(R, seed=seed + restart)
            elif restart % 3 == 1:
                # Random initialization
                U = rng.uniform(-2, 2, (nm, R))
                V = rng.uniform(-2, 2, (mp, R))
                W = rng.uniform(-2, 2, (pn, R))
            else:
                # Sparse random initialization
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    U[rng.randint(0, nm), r] = rng.choice([-1.0, 1.0])
                    V[rng.randint(0, mp), r] = rng.choice([-1.0, 1.0])
                    W[rng.randint(0, pn), r] = rng.choice([-1.0, 1.0])
            
            # Precompute index arrays for vectorized updates
            # For U update: rows are (i,j), columns are k
            ij_indices = [(i, j) for i in range(n) for j in range(m)]
            k_indices = list(range(p))
            # For V update: rows are (j,k), columns are i
            jk_indices = [(j, k) for j in range(m) for k in range(p)]
            i_indices = list(range(n))
            # For W update: rows are (k,i), columns are j
            ki_indices = [(k, i) for k in range(p) for i in range(n)]
            j_indices = list(range(m))
            
            # Phase 1: Alternating Least Squares (vectorized)
            for it in range(150):
                if time.time() - start_time > time_limit * 0.5:
                    break
                
                # Update U: for each (i,j), solve A * U_row = ones(p)
                for i in range(n):
                    for j in range(m):
                        V_sub = V[j*p:(j+1)*p, :]  # (p, R)
                        W_rows = W[np.arange(p)*n + i, :]  # (p, R)
                        A = V_sub * W_rows  # (p, R)
                        b = np.ones(p)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve A * V_row = ones(n)
                for j in range(m):
                    for k in range(p):
                        U_rows = U[np.arange(n)*m + j, :]  # (n, R)
                        W_rows = W[k*n:(k+1)*n, :]  # (n, R)
                        A = U_rows * W_rows  # (n, R)
                        b = np.ones(n)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve A * W_row = ones(m)
                for k in range(p):
                    for i in range(n):
                        U_rows = U[i*m:(i+1)*m, :]  # (m, R)
                        V_rows = V[np.arange(m)*p + k, :]  # (m, R)
                        A = U_rows * V_rows  # (m, R)
                        b = np.ones(m)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check error periodically
                if it % 5 == 0:
                    U_sel = U[idx_u, :]
                    V_sel = V[idx_v, :]
                    W_sel = W[idx_w, :]
                    res = np.sum(U_sel * V_sel * W_sel, axis=1)
                    err = np.sum((res - 1.0)**2)
                    if err < 1e-14:
                        break
            
            # Phase 2: Least Squares Polish
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R+mp*R]
                W_flat = x[nm*R+mp*R:]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                U_sel = U_mat[idx_u, :]
                V_sel = V_mat[idx_v, :]
                W_sel = W_mat[idx_w, :]
                
                res = np.sum(U_sel * V_sel * W_sel, axis=1)
                return res - 1.0
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=8000, xtol=1e-16, ftol=1e-16, gtol=1e-16)
                
                if result.success or result.cost < 1e-12:
                    x_sol = result.x
                    U_sol = x_sol[:nm*R].reshape(nm, R)
                    V_sol = x_sol[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_sol = x_sol[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try rounding to nearest half-integer with local search
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, try local search around rounded solution
                    if result.cost < 1e-6:
                        current_U, current_V, current_W = U_r, V_r, W_r
                        
                        for _round in range(50):
                            if time.time() - start_time > time_limit:
                                break
                            
                            found_flip = False
                            for mat_name, mat in [('U', current_U), ('V', current_V), ('W', current_W)]:
                                for r_idx in range(mat.shape[0]):
                                    for c_idx in range(mat.shape[1]):
                                        orig_val = mat[r_idx, c_idx]
                                        nearest_hi = np.ceil(orig_val * 2) / 2
                                        nearest_lo = np.floor(orig_val * 2) / 2
                                        
                                        for new_val in [nearest_hi, nearest_lo]:
                                            if abs(new_val - orig_val) < 1e-9:
                                                continue
                                            
                                            if mat_name == 'U':
                                                U_test = current_U.copy()
                                                U_test[r_idx, c_idx] = new_val
                                                if verify_solution(U_test, current_V, current_W):
                                                    current_U = U_test
                                                    found_flip = True
                                                    break
                                            elif mat_name == 'V':
                                                V_test = current_V.copy()
                                                V_test[r_idx, c_idx] = new_val
                                                if verify_solution(current_U, V_test, current_W):
                                                    current_V = V_test
                                                    found_flip = True
                                                    break
                                            else:
                                                W_test = current_W.copy()
                                                W_test[r_idx, c_idx] = new_val
                                                if verify_solution(current_U, current_V, W_test):
                                                    current_W = W_test
                                                    found_flip = True
                                                    break
                                        
                                        if found_flip:
                                            break
                                    if found_flip:
                                        break
                                if found_flip:
                                    break
                            
                            if verify_solution(current_U, current_V, current_W):
                                return current_U, current_V, current_W
                            
                            if not found_flip:
                                break
                            
            except Exception as e:
                pass
                
        return None

    # Main Logic
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    timeout = 135.0
    
    # Try ranks from 29 down to 25
    for R in range(29, 24, -1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Use more seeds and shorter time per attempt to maximize coverage
        for seed in range(50):
            if time.time() - start_time > timeout:
                break
            res = try_find_rank(R, time_limit=2.0, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            continue
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
