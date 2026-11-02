import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """
    Builds the flattened target tensor T of shape (n*m, m*p, p*n)
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_solution(U, V, W, T, tol=1e-6):
    """
    Checks if U, V, W exactly decompose T within tolerance.
    Returns error norm.
    """
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(T_decomp - T)

def _snap_to_half_int(M):
    """
    Snaps matrix M to nearest half-integer.
    """
    return np.round(M * 2.0) / 2.0

def _als_step(U, V, W, T, A, B, C, R, tol=1e-8):
    """
    Perform one ALS iteration. Returns updated U, V, W and error.
    """
    # Solve for U
    M_U = np.einsum('br,cr->rbc', V, W).reshape(R, B * C)
    T_flat = T.reshape(A, B * C)
    for a in range(A):
        try:
            U[a], _, _, _ = np.linalg.lstsq(M_U.T, T_flat[a], rcond=None)
        except:
            pass
            
    # Solve for V
    M_V = np.einsum('ar,cr->rac', U, W).reshape(R, A * C)
    T_flat_V = T.reshape(A, B, C).transpose(1, 0, 2).reshape(B, A * C)
    for b in range(B):
        try:
            V[b], _, _, _ = np.linalg.lstsq(M_V.T, T_flat_V[b], rcond=None)
        except:
            pass
            
    # Solve for W
    M_W = np.einsum('ar,br->rab', U, V).reshape(R, A * B)
    T_flat_W = T.reshape(A, B, C).transpose(2, 0, 1).reshape(C, A * B)
    for c in range(C):
        try:
            W[c], _, _, _ = np.linalg.lstsq(M_W.T, T_flat_W[c], rcond=None)
        except:
            pass
            
    err = _check_solution(U, V, W, T)
    return U, V, W, err

def _try_half_int_search(U, V, W, T, A, B, C, R, max_steps=50):
    """
    Try to find a half-integer solution by local search around current solution.
    """
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for step in range(max_steps):
        # Snap to half-integers
        U_snap = _snap_to_half_int(U_cur)
        V_snap = _snap_to_half_int(V_cur)
        W_snap = _snap_to_half_int(W_cur)
        
        err_snap = _check_solution(U_snap, V_snap, W_snap, T)
        if err_snap < 1e-6:
            return U_snap, V_snap, W_snap, True
        
        # If snapping doesn't work, try small perturbations
        # Try flipping some entries to adjacent half-integers
        found_improvement = False
        for _ in range(10):
            # Randomly pick a matrix and entry to perturb
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row, col = np.random.randint(A), np.random.randint(R)
                U_test = U_snap.copy()
                U_test[row, col] += np.random.choice([-0.5, 0.5])
                if _check_solution(U_test, V_snap, W_snap, T) < err_snap:
                    U_cur = U_test
                    found_improvement = True
            elif mat_idx == 1:
                row, col = np.random.randint(B), np.random.randint(R)
                V_test = V_snap.copy()
                V_test[row, col] += np.random.choice([-0.5, 0.5])
                if _check_solution(U_snap, V_test, W_snap, T) < err_snap:
                    V_cur = V_test
                    found_improvement = True
            else:
                row, col = np.random.randint(C), np.random.randint(R)
                W_test = W_snap.copy()
                W_test[row, col] += np.random.choice([-0.5, 0.5])
                if _check_solution(U_snap, V_snap, W_test, T) < err_snap:
                    W_cur = W_test
                    found_improvement = True
        
        if not found_improvement:
            break
    
    U_snap = _snap_to_half_int(U_cur)
    V_snap = _snap_to_half_int(V_cur)
    W_snap = _snap_to_half_int(W_cur)
    err_snap = _check_solution(U_snap, V_snap, W_snap, T)
    if err_snap < 1e-6:
        return U_snap, V_snap, W_snap, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    target_rank = 25
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Time limit guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    A = n * m
    B = m * p
    C = p * n
    
    # Try to find rank 25 solution
    # Strategy: Use structured initialization with half-integer values
    # Then refine with ALS and try to snap to half-integers
    
    half_int_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for attempt in range(200):  # Max 200 restarts
        if time.time() - start_time > timeout:
            break
            
        R = target_rank
        
        # Structured initialization: start with half-integer values
        # Use a mix of structured and random patterns
        if attempt < 50:
            # Pattern 1: Sparse initialization with few non-zero entries
            U = np.zeros((A, R))
            V = np.zeros((B, R))
            W = np.zeros((C, R))
            for r in range(R):
                # Each rank-1 component contributes to a subset of entries
                idx_u = np.random.choice(A, size=np.random.randint(1, 4), replace=False)
                idx_v = np.random.choice(B, size=np.random.randint(1, 4), replace=False)
                idx_w = np.random.choice(C, size=np.random.randint(1, 4), replace=False)
                for idx in idx_u:
                    U[idx, r] = np.random.choice(half_int_values)
                for idx in idx_v:
                    V[idx, r] = np.random.choice(half_int_values)
                for idx in idx_w:
                    W[idx, r] = np.random.choice(half_int_values)
        elif attempt < 100:
            # Pattern 2: Dense initialization with small values
            U = np.random.choice(half_int_values, size=(A, R)) * 0.5
            V = np.random.choice(half_int_values, size=(B, R)) * 0.5
            W = np.random.choice(half_int_values, size=(C, R)) * 0.5
        elif attempt < 150:
            # Pattern 3: Based on schoolbook structure
            U = np.zeros((A, R))
            V = np.zeros((B, R))
            W = np.zeros((C, R))
            # Use first 15 components from schoolbook, then randomize
            for r in range(min(15, R)):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            # Randomize remaining
            for r in range(15, R):
                U[:, r] = np.random.choice(half_int_values, size=A) * 0.5
                V[:, r] = np.random.choice(half_int_values, size=B) * 0.5
                W[:, r] = np.random.choice(half_int_values, size=C) * 0.5
        else:
            # Pattern 4: Fully random half-integer initialization
            U = np.random.choice(half_int_values, size=(A, R))
            V = np.random.choice(half_int_values, size=(B, R))
            W = np.random.choice(half_int_values, size=(C, R))
        
        # Run ALS to refine
        converged = False
        for iter in range(100):  # Max 100 iterations per restart
            if time.time() - start_time > timeout:
                break
                
            U, V, W, err = _als_step(U, V, W, T, A, B, C, R)
            
            if err < 1e-6:
                # Check if already half-integer
                U_snap = _snap_to_half_int(U)
                V_snap = _snap_to_half_int(V)
                W_snap = _snap_to_half_int(W)
                err_snap = _check_solution(U_snap, V_snap, W_snap, T)
                if err_snap < 1e-6:
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    converged = True
                    break
            
            # Try local half-integer search
            if iter % 10 == 9:  # Every 10 iterations
                U_hi, V_hi, W_hi, success = _try_half_int_search(U, V, W, T, A, B, C, R, max_steps=20)
                if success:
                    best_rank = R
                    best_U = U_hi
                    best_V = V_hi
                    best_W = W_hi
                    converged = True
                    break
        
        if converged:
            break
    
    # If we found a better rank, return it. Otherwise return schoolbook.
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
