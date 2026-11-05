import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_target_333():
    """Build the target tensor for 3x3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _reconstruct(U, V, W):
    """Reconstruct tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _error_and_grad_fast(U, V, W, T):
    """Compute error and gradients for tensor decomposition - vectorized."""
    # T_rec[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = T_rec - T
    err = np.sum(diff ** 2)
    
    # grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    grad_U = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
    grad_V = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
    grad_W = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
    
    return err, grad_U, grad_V, grad_W

def _try_rank_333(U, V, W, T, max_steps=300, lr_init=0.01, tol=1e-10):
    """Try to optimize a rank-r decomposition for 3x3x3 to zero error."""
    r = U.shape[1]
    lr = lr_init
    min_lr = 1e-8
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    patience = 0
    
    for step in range(max_steps):
        err, gU, gV, gW = _error_and_grad_fast(U, V, W, T)
        
        if err < tol:
            return U, V, W, err
        
        if err < best_err * 0.999:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
            if patience > 30:
                break
        
        norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if norm_g > 1e-10:
            step_size = lr / (1.0 + norm_g * 0.1)
        else:
            step_size = lr
        
        U -= step_size * gU
        V -= step_size * gV
        W -= step_size * gW
        
        lr = max(min_lr, lr * 0.99)
    
    return best_U, best_V, best_W, best_err

def _polish(U, V, W, T, max_steps=200, tol=1e-12):
    """Final high-precision polish pass."""
    r = U.shape[1]
    lr = 1e-4
    min_lr = 1e-10
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    patience = 0
    
    for step in range(max_steps):
        err, gU, gV, gW = _error_and_grad_fast(U, V, W, T)
        
        if err < tol:
            return U, V, W, err
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break
        
        norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if norm_g > 1e-12:
            step_size = lr / (1.0 + norm_g * 0.5)
        else:
            step_size = lr
        
        U -= step_size * gU
        V -= step_size * gV
        W -= step_size * gW
        
        lr = max(min_lr, lr * 0.98)
    
    return best_U, best_V, best_W, best_err

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s

def _verify_336(U, V, W):
    """Verify the 3x3x6 decomposition."""
    n, m, p = 3, 3, 6
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check on-diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check off-diagonal
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    return True

def _strassen_init_333(r):
    """Initialize using Strassen-like structure for 3x3x3."""
    # Strassen uses 7 multiplications for 2x2x2
    # For 3x3x3, we can use a structured approach
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    W = np.zeros((9, r))
    
    # Use first 7 columns for Strassen-like pattern
    if r >= 7:
        # Strassen decomposition for 3x3 (simplified)
        # This is a heuristic initialization
        for col in range(7):
            U[0, col] = 1.0
            U[1, col] = 1.0
            V[0, col] = 1.0
            V[1, col] = 1.0
            W[0, col] = 1.0
            W[1, col] = 1.0
    
    # Add random components for remaining columns
    for col in range(7, r):
        U[:, col] = np.random.randn(9) * 0.3
        V[:, col] = np.random.randn(9) * 0.3
        W[:, col] = np.random.randn(9) * 0.3
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Strategy: Find rank-r decomposition for 3x3x3, then double it for 3x3x6
    T_333 = _build_target_333()
    
    # Try ranks for 3x3x3: 19 is the theoretical best known.
    # If we find 19, we get 38 for 3x3x6.
    # If we find 20, we get 40.
    # Schoolbook is 54.
    
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    for r_333 in ranks_to_try:
        if time.time() > deadline - 30:
            break
            
        max_trials = 30 if r_333 <= 20 else 15
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(12345 + r_333 * 7919 + trial * 104729)
            
            # Use structured initialization for some trials
            if trial % 3 == 0:
                U_r, V_r, W_r = _strassen_init_333(r_333)
            else:
                # Random initialization with varied scales
                scale = 0.3 + 0.2 * (trial % 3)
                U_r = np.random.randn(9, r_333) * scale
                V_r = np.random.randn(9, r_333) * scale
                W_r = np.random.randn(9, r_333) * scale
            
            U_opt, V_opt, W_opt, err_opt = _try_rank_333(U_r, V_r, W_r, T_333)
            
            if err_opt < 1e-8:
                # Polish pass
                U_pol, V_pol, W_pol, err_pol = _polish(U_opt, V_opt, W_opt, T_333)
                
                if err_pol < 1e-10:
                    # Snap to half-integers
                    U_s, V_s, W_s = _snap_to_half_integers(U_pol, V_pol, W_pol)
                    
                    # Verify snapped 3x3x3
                    T_rec_333 = _reconstruct(U_s, V_s, W_s)
                    if np.allclose(T_rec_333, T_333, atol=1e-9):
                        # Construct 3x3x6 from 3x3x3
                        rank_336 = 2 * r_333
                        U_336 = np.hstack([U_s, U_s])
                        V_336 = np.zeros((18, rank_336))
                        W_336 = np.zeros((18, rank_336))
                        
                        # Fill V_336
                        for j in range(3):
                            for k in range(3):
                                # First block (k < 3)
                                idx_v = j * 6 + k
                                V_336[idx_v, :r_333] = V_s[j * 3 + k, :]
                                # Second block (k >= 3)
                                idx_v2 = j * 6 + (k + 3)
                                V_336[idx_v2, r_333:] = V_s[j * 3 + k, :]
                                
                        # Fill W_336
                        for k in range(3):
                            for i in range(3):
                                # First block (k < 3)
                                idx_w = k * 3 + i
                                W_336[idx_w, :r_333] = W_s[k * 3 + i, :]
                                # Second block (k >= 3)
                                idx_w2 = (k + 3) * 3 + i
                                W_336[idx_w2, r_333:] = W_s[k * 3 + i, :]
                                
                        # Verify 3x3x6
                        if _verify_336(U_336, V_336, W_336):
                            if rank_336 < best_rank:
                                best_rank = rank_336
                                best_U, best_V, best_W = U_336.copy(), V_336.copy(), W_336.copy()
                            found = True
                            break
        
        if found:
            # If we found a solution, we can stop or try lower?
            # We tried in increasing order, so the first found is the lowest.
            break
            
    if best_rank < rank_school:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        return {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}
