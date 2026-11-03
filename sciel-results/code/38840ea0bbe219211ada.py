import time
import numpy as np
import scipy.optimize as optimize

START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _strassen_2x2x2():
    """
    Returns Strassen's rank-7 decomposition for 2x2x2 matmul.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 multiplications
    # m1 = (A11 + A22)(B11 + B22)
    # m2 = (A21 + A22)B11
    # m3 = A11(B12 - B22)
    # m4 = A22(B21 - B11)
    # m5 = (A11 + A12)B22
    # m6 = (A21 - A11)(B11 + B12)
    # m7 = (A12 - A22)(B21 + B22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # m1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1
    
    # m2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # m3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # m4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[3,3] = 1
    
    # m5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[3,4] = 1
    
    # m6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1; W[2,5] = 1
    
    # m7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = 1
    
    return U, V, W

def _embed_2x2x2_in_2x3x3(U_sub, V_sub, W_sub):
    """
    Embed a 2x2x2 decomposition into a 2x3x3 decomposition.
    U_sub: (4, 7) -> U: (6, 7)
    V_sub: (4, 7) -> V: (9, 7)
    W_sub: (4, 7) -> W: (6, 7)
    """
    n, m, p = 2, 3, 3
    r = 7
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Copy U (indices i, j where i in 0..1, j in 0..1)
    for i in range(2):
        for j in range(2):
            U[i*3 + j, :] = U_sub[i*2 + j, :]
            
    # Copy V (indices j, k where j in 0..1, k in 0..1)
    for j in range(2):
        for k in range(2):
            V[j*3 + k, :] = V_sub[j*2 + k, :]
            
    # Copy W (indices k, i where k in 0..1, i in 0..1)
    for k in range(2):
        for i in range(2):
            W[k*2 + i, :] = W_sub[k*2 + i, :]
            
    return U, V, W

def _try_find_2x3x3_rank17():
    """
    Attempt to find a rank-17 decomposition for 2x3x3 matmul.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 3, 3
    r_target = 17
    
    # Start with Strassen embedded (rank 7)
    U_str, V_str, W_str = _strassen_2x2x2()
    U0, V0, W0 = _embed_2x2x2_in_2x3x3(U_str, V_str, W_str)
    
    # Add 10 random columns
    np.random.seed(42)
    U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r_target - 7))
    V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r_target - 7))
    W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r_target - 7))
    
    U_init = np.hstack([U0, U_rand])
    V_init = np.hstack([V0, V_rand])
    W_init = np.hstack([W0, W_rand])
    
    # Flatten for optimization
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(x):
        U_flat, V_flat, W_flat = x[:n*m*r_target], x[n*m*r_target:n*m*r_target + m*p*r_target], x[n*m*r_target + m*p*r_target:]
        U = U_flat.reshape(n*m, r_target)
        V = V_flat.reshape(m*p, r_target)
        W = W_flat.reshape(p*n, r_target)
        return _compute_residual(U, V, W, n, m, p)
    
    # Use Powell or Nelder-Mead for robustness with non-smoothness (though objective is smooth)
    # Powell is good for many variables
    try:
        res = optimize.minimize(objective, x0, method='Powell', options={'maxiter': 500, 'disp': False})
        x_opt = res.x
    except Exception:
        return None
    
    U_opt = x_opt[:n*m*r_target].reshape(n*m, r_target)
    V_opt = x_opt[n*m*r_target:n*m*r_target + m*p*r_target].reshape(m*p, r_target)
    W_opt = x_opt[n*m*r_target + m*p*r_target:].reshape(p*n, r_target)
    
    # Snap to half integers
    U_snap = _snap_to_half_integers(U_opt)
    V_snap = _snap_to_half_integers(V_opt)
    W_snap = _snap_to_half_integers(W_opt)
    
    # Verify
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return U_snap, V_snap, W_snap
    
    # If direct snap failed, try local search on snapped values
    # We can try to fix individual entries
    U_ls = U_snap.copy()
    V_ls = V_snap.copy()
    W_ls = W_snap.copy()
    
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    current_err = _compute_residual(U_ls, V_ls, W_ls, n, m, p)
    
    max_iters = 500
    for _ in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        if current_err < 1e-12:
            break
            
        improved = False
        # Randomly pick a matrix and element
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            row = np.random.randint(0, n*m)
            col = np.random.randint(0, r_target)
            mat = U_ls
        elif mat_idx == 1:
            row = np.random.randint(0, m*p)
            col = np.random.randint(0, r_target)
            mat = V_ls
        else:
            row = np.random.randint(0, p*n)
            col = np.random.randint(0, r_target)
            mat = W_ls
            
        old_val = mat[row, col]
        best_val = old_val
        best_err = current_err
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            mat[row, col] = new_val
            new_err = _compute_residual(U_ls, V_ls, W_ls, n, m, p)
            if new_err < best_err:
                best_err = new_err
                best_val = new_val
            else:
                mat[row, col] = old_val
        
        if best_val != old_val:
            mat[row, col] = best_val
            current_err = best_err
            improved = True
            
    if _verify_decomposition(U_ls, V_ls, W_ls, n, m, p):
        return U_ls, V_ls, W_ls
        
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # Try to find rank-17 decomposition for 2x3x3
    U_sub, V_sub, W_sub = None, None, None
    if time.time() - START_TIME < TIME_LIMIT - 20:
        U_sub, V_sub, W_sub = _try_find_2x3x3_rank17()
        
    if U_sub is not None:
        # Construct 2x3x6 decomposition from 2x3x3
        r_sub = U_sub.shape[1]
        r_total = 2 * r_sub
        
        U = np.zeros((n*m, r_total))
        V = np.zeros((m*p, r_total))
        W = np.zeros((p*n, r_total))
        
        # U is shared (same i, j for both blocks)
        # Columns 0..r_sub-1 use U_sub
        # Columns r_sub..2*r_sub-1 use U_sub
        U[:, :r_sub] = U_sub
        U[:, r_sub:] = U_sub
        
        # V: rows j*p+k. 
        # Block 1: k in 0..2. V[j*3+k, c] = V_sub[j*3+k, c] for c in 0..r_sub-1
        # Block 2: k in 3..5. V[j*3+k, c] = V_sub[j*3+(k-3), c-r_sub] for c in r_sub..2*r_sub-1
        for j in range(m):
            for k in range(3):
                V[j*p + k, :r_sub] = V_sub[j*3 + k, :]
                V[j*p + k, r_sub:] = 0
            for k in range(3, 6):
                V[j*p + k, :r_sub] = 0
                V[j*p + k, r_sub:] = V_sub[j*3 + (k-3), :]
                
        # W: rows k*n+i.
        # Block 1: k in 0..2. W[k*2+i, c] = W_sub[k*2+i, c] for c in 0..r_sub-1
        # Block 2: k in 3..5. W[k*2+i, c] = W_sub[(k-3)*2+i, c-r_sub] for c in r_sub..2*r_sub-1
        for k in range(3):
            for i in range(n):
                W[k*n + i, :r_sub] = W_sub[k*n + i, :]
                W[k*n + i, r_sub:] = 0
        for k in range(3, 6):
            for i in range(n):
                W[k*n + i, :r_sub] = 0
                W[k*n + i, r_sub:] = W_sub[(k-3)*n + i, :]
                
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback: Schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    
    return {"U": U_list, "V": V_list, "W": W_list}
