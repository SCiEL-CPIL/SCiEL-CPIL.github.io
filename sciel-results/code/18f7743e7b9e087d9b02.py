"""Program for matrix multiplication 3x3 * 3x6 rank-minimal search.

Strategy:
  1. Try to exploit the structure: the 3x3 * 3x6 multiplication can be seen
     as two independent 3x3 * 3x3 multiplications (splitting the 6 columns
     into two groups of 3).  If a rank-r decomposition of the 3x3x3 tensor
     exists, we get a rank-2r decomposition of the 3x3x6 tensor.
  2. Use gradient-based optimisation to search for low-rank exact
     decompositions of the 3x3x3 tensor.  The best known rank for 3x3x3
     is 19 (Strassen's algorithm gives 7 for 2x2x2, but for 3x3x3 the
     best known is 19 due to Bläser's algorithm).
  3. If no exact low-rank solution is found within the time budget,
     fall back to the schoolbook rank-54 decomposition.
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]


def _build_target_333():
    """Build the target tensor for 3x3x3 matrix multiplication.
    T[a,b,c] = 1 if (a,b,c) = (i*3+j, j*3+k, k*3+i) for some i,j,k in 0..2.
    """
    n, m, p = 3, 3, 3
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _build_target_336():
    """Build the target tensor for 3x3x6 matrix multiplication.
    T[a,b,c] = 1 if (a,b,c) = (i*3+j, j*6+k, k*3+i) for i,j in 0..2, k in 0..5.
    """
    n, m, p = 3, 3, 6
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _reconstruct(U, V, W):
    """Reconstruct tensor from factor matrices: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _error_and_grad(U, V, W, T):
    """Compute squared error and gradients for tensor decomposition."""
    T_rec = _reconstruct(U, V, W)
    diff = T_rec - T
    err = np.sum(diff ** 2)
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    return err, grad_U, grad_V, grad_W


def _try_optimize(U, V, W, T, max_steps=800, lr_init=0.02, tol=1e-12,
                  time_limit=10.0):
    """Try to optimise a rank-r decomposition to zero error using gradient descent
    with adaptive step size and restarts. Returns (best_U, best_V, best_W, best_err)."""
    r = U.shape[1]
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    t0 = time.time()
    
    for attempt in range(3):
        if time.time() - t0 > time_limit:
            break
        U_c, V_c, W_c = U.copy(), V.copy(), W.copy()
        lr = lr_init
        patience = 0
        max_patience = 80
        
        for step in range(max_steps):
            if time.time() - t0 > time_limit:
                break
            err, gU, gV, gW = _error_and_grad(U_c, V_c, W_c, T)
            
            if err < tol:
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                return best_U, best_V, best_W, best_err
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
            
            norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
            if norm_g > 1e-10:
                step_size = lr / (1.0 + norm_g * 0.03)
            else:
                step_size = lr
            
            U_c -= step_size * gU
            V_c -= step_size * gV
            W_c -= step_size * gW
            lr = max(1e-10, lr * 0.998)
    
    return best_U, best_V, best_W, best_err


def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s


def _verify_336(U, V, W):
    """Verify the 3x3x6 decomposition exactly (within numerical tolerance)."""
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries
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


def _verify_333(U, V, W):
    """Verify the 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
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


def _make_schoolbook_336():
    """Return the schoolbook rank-54 decomposition for 3x3x6."""
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _expand_333_to_336(U3, V3, W3, r3):
    """Expand a rank-r3 3x3x3 decomposition to a rank-2*r3 3x3x6 decomposition.
    The 3x3x6 multiplication is equivalent to two independent 3x3x3 multiplications
    (columns 0-2 and columns 3-5 of B)."""
    rank_336 = 2 * r3
    U_336 = np.hstack([U3, U3])
    V_336 = np.zeros((18, rank_336))
    W_336 = np.zeros((18, rank_336))
    
    for j in range(3):
        for k in range(3):
            # First block: k in 0..2
            idx_v = j * 6 + k
            V_336[idx_v, :r3] = V3[j * 3 + k, :]
            # Second block: k in 3..5 (maps to k-3 in 3x3x3)
            idx_v2 = j * 6 + (k + 3)
            V_336[idx_v2, r3:] = V3[j * 3 + k, :]
    
    for k in range(3):
        for i in range(3):
            # First block: k in 0..2
            idx_w = k * 3 + i
            W_336[idx_w, :r3] = W3[k * 3 + i, :]
            # Second block: k in 3..5 (maps to k-3 in 3x3x3)
            idx_w2 = (k + 3) * 3 + i
            W_336[idx_w2, r3:] = W3[k * 3 + i, :]
    
    return U_336, V_336, W_336, rank_336


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank_school = n * m * p  # 54
    
    # Start with schoolbook as the guaranteed fallback
    U_school, V_school, W_school, best_rank = _make_schoolbook_336()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Strategy: find low-rank 3x3x3 decomposition, then expand to 3x3x6
    T_333 = _build_target_333()
    
    # Try ranks for 3x3x3 in increasing order.
    # Best known for 3x3x3 is 19 (Bläser). Strassen gives 7 for 2x2x2.
    # If we find rank r for 3x3x3, we get rank 2r for 3x3x6.
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    
    for r3 in ranks_to_try:
        if time.time() > deadline - 40:
            break
        
        # If 2*r3 >= best_rank, no point continuing
        if 2 * r3 >= best_rank:
            continue
        
        max_trials = 15 if r3 <= 20 else 8
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + r3 * 1000 + trial * 7919)
            
            # Random initialization with small values
            U_r = np.random.randn(9, r3) * 0.3
            V_r = np.random.randn(9, r3) * 0.3
            W_r = np.random.randn(9, r3) * 0.3
            
            # Also try a structured initialization: start from schoolbook
            # and randomly perturb / merge
            if trial % 3 == 1:
                # Structured: start from schoolbook and randomly zero out columns
                U_struct = np.zeros((9, r3))
                V_struct = np.zeros((9, r3))
                W_struct = np.zeros((9, r3))
                # Pick r3 random (i,j,k) triples
                indices = np.random.choice(27, size=r3, replace=False)
                for col_idx, idx in enumerate(indices):
                    i = idx // 9
                    j = (idx // 3) % 3
                    k = idx % 3
                    U_struct[i * 3 + j, col_idx] = 1.0
                    V_struct[j * 3 + k, col_idx] = 1.0
                    W_struct[k * 3 + i, col_idx] = 1.0
                # Add small random perturbation
                U_r = U_struct + np.random.randn(9, r3) * 0.1
                V_r = V_struct + np.random.randn(9, r3) * 0.1
                W_r = W_struct + np.random.randn(9, r3) * 0.1
            
            if trial % 3 == 2:
                # Another random init with different scale
                U_r = np.random.randn(9, r3) * 0.5
                V_r = np.random.randn(9, r3) * 0.5
                W_r = np.random.randn(9, r3) * 0.5
            
            U_opt, V_opt, W_opt, err_opt = _try_optimize(
                U_r, V_r, W_r, T_333,
                max_steps=600, lr_init=0.015, tol=1e-12,
                time_limit=8.0
            )
            
            if err_opt < 1e-8:
                # Snap to half-integers
                U_s, V_s, W_s = _snap_to_half_integers(U_opt, V_opt, W_opt)
                
                # Verify snapped 3x3x3
                if _verify_333(U_s, V_s, W_s):
                    # Expand to 3x3x6
                    U_336, V_336, W_336, rank_336 = _expand_333_to_336(
                        U_s, V_s, W_s, r3
                    )
                    
                    # Verify 3x3x6
                    if _verify_336(U_336, V_336, W_336):
                        if rank_336 < best_rank:
                            best_rank = rank_336
                            best_U, best_V, best_W = U_336.copy(), V_336.copy(), W_336.copy()
                        found = True
                        break
        
        if found:
            # Found a solution at this rank; since we go in increasing order,
            # this is the best we'll get from this strategy.
            break
    
    # If we still have time, try direct optimization on 3x3x6 for ranks
    # between 2*best_known_333 and best_rank-1
    if time.time() < deadline - 20:
        T_336 = _build_target_336()
        # Try a few direct ranks that might be achievable
        direct_ranks = [2 * r for r in range(19, 28)]
        direct_ranks = [r for r in direct_ranks if r < best_rank and r >= 30]
        
        for r_direct in sorted(set(direct_ranks)):
            if time.time() > deadline - 10:
                break
            if r_direct >= best_rank:
                continue
            
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                np.random.seed(99 + r_direct * 100 + trial * 331)
                U_d = np.random.randn(9, r_direct) * 0.2
                V_d = np.random.randn(18, r_direct) * 0.2
                W_d = np.random.randn(18, r_direct) * 0.2
                
                U_opt, V_opt, W_opt, err_opt = _try_optimize(
                    U_d, V_d, W_d, T_336,
                    max_steps=400, lr_init=0.01, tol=1e-12,
                    time_limit=5.0
                )
                
                if err_opt < 1e-8:
                    U_s, V_s, W_s = _snap_to_half_integers(U_opt, V_opt, W_opt)
                    if _verify_336(U_s, V_s, W_s):
                        if r_direct < best_rank:
                            best_rank = r_direct
                            best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                        break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
