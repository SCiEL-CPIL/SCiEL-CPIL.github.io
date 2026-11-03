import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact_fast(U_np, V_np, W_np, n, m, p):
    """Fast verification using tensor operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] equals target tensor."""
    rank = U_np.shape[1]
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build target tensor: T[i*m+j, jp*p+k, kp*n+ip] = 1 iff i==ip, j==jp, k==kp
    # We'll check by reconstructing the product
    
    # Efficient check: for each r, the contribution is outer(U[:,r], outer(V[:,r], W[:,r]))
    # Sum over r should give target
    
    # Alternative: check that for all (i,j,k), (ip,jp,kp):
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] == delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    # Vectorized: compute all entries
    # T[a, b, c] where a=i*m+j, b=jp*p+k, c=kp*n+ip
    # a encodes (i,j), b encodes (jp,k), c encodes (kp,ip)
    # Target: 1 iff i==ip AND j==jp AND k==kp
    
    # Let's use einsum for speed
    # T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    # This might be too large: 6*12*8 = 576 entries, fine!
    
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Build target
    target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                target[i*m+j, jp*p+k, kp*n+ip] = 1.0
    
    if np.max(np.abs(T - target)) > 1e-9:
        return False
    return True

def _snap_to_half(U_np, V_np, W_np):
    """Snap all entries to nearest half-integer."""
    U_snap = np.round(U_np * 2) / 2
    V_snap = np.round(V_np * 2) / 2
    W_snap = np.round(W_np * 2) / 2
    return U_snap, V_snap, W_snap

def _als_optimize(U_init, V_init, W_init, n, m, p, max_iters=200, tol=1e-10):
    """Alternating least squares to optimize factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Minimize ||T_target - sum_r U[:,r] o V[:,r] o W[:,r]||_F"""
    rank = U_init.shape[1]
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build target tensor
    target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                target[i*m+j, jp*p+k, kp*n+ip] = 1.0
    
    U = U_init.astype(np.float64).copy()
    V = V_init.astype(np.float64).copy()
    W = W_init.astype(np.float64).copy()
    
    prev_err = np.inf
    for it in range(max_iters):
        # Update U: for each a, U[a,:] = argmin ||target[a,:,:] - U[a,:] @ (V*W)||
        # target[a,:,:] has shape (dim_V, dim_W)
        # We need: U[a,:] such that sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
        # For fixed a: target[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # So: target[a,:,:].reshape(-1) = U[a,:] @ (V*W).reshape(rank, dim_V*dim_W)
        
        VW = (V * W[:, np.newaxis, :]).reshape(rank, dim_V * dim_W)  # (rank, dim_V*dim_W)
        # Actually: outer(V[:,r], W[:,r]) for each r, stacked
        # VW[r, b*dim_W + c] = V[b,r] * W[c,r]
        VW = np.einsum('br,cr->rbc', V, W).reshape(rank, dim_V * dim_W)
        
        for a in range(dim_U):
            rhs = target[a, :, :].reshape(-1)
            U[a, :] = np.linalg.lstsq(VW, rhs, rcond=None)[0]
        
        # Update V
        UW = np.einsum('ar,cr->rac', U, W).reshape(rank, dim_U * dim_W)
        # UW[r, a*dim_W + c] = U[a,r] * W[c,r]
        UW = np.einsum('ar,cr->rac', U, W).reshape(rank, dim_U * dim_W)
        
        for b in range(dim_V):
            rhs = target[:, b, :].reshape(-1)
            V[b, :] = np.linalg.lstsq(UW, rhs, rcond=None)[0]
        
        # Update W
        UV = np.einsum('ar,br->rab', U, V).reshape(rank, dim_U * dim_V)
        UV = np.einsum('ar,br->rab', U, V).reshape(rank, dim_U * dim_V)
        
        for c in range(dim_W):
            rhs = target[:, :, c].reshape(-1)
            W[c, :] = np.linalg.lstsq(UV, rhs, rcond=None)[0]
        
        # Compute error
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - target)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _random_init(rank, dim_U, dim_V, dim_W, n, m, p):
    """Random initialization with half-integer values."""
    # Use small random half-integers
    U = np.round(np.random.uniform(-2, 2, (dim_U, rank)) * 2) / 2
    V = np.round(np.random.uniform(-2, 2, (dim_V, rank)) * 2) / 2
    W = np.round(np.random.uniform(-2, 2, (dim_W, rank)) * 2) / 2
    return U, V, W

def _try_rank_als(rank, n, m, p, max_restarts=10, max_iters=100, deadline=None):
    """Try to find a rank-`rank` decomposition using ALS with random restarts."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(max_restarts):
        if deadline and time.time() > deadline:
            break
        
        U_init, V_init, W_init = _random_init(rank, dim_U, dim_V, dim_W, n, m, p)
        
        U, V, W = _als_optimize(U_init, V_init, W_init, n, m, p, max_iters=max_iters)
        
        # Snap to half-integers
        U_snap, V_snap, W_snap = _snap_to_half(U, V, W)
        
        # Check if exact
        if _verify_exact_fast(U_snap, V_snap, W_snap, n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Also check unsnapped (in case it's already exact with floats)
        if _verify_exact_fast(U, V, W, n, m, p):
            # Try to find half-integer representation
            U_h, V_h, W_h = _snap_to_half(U, V, W)
            if _verify_exact_fast(U_h, V_h, W_h, n, m, p):
                return U_h.tolist(), V_h.tolist(), W_h.tolist()
        
        # Track best error for potential use
        T = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        target = np.zeros((dim_U, dim_V, dim_W))
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                if i == ip and j == jp and k == kp:
                                    target[i*m+j, jp*p+k, kp*n+ip] = 1.0
        err = np.linalg.norm(T - target)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
    
    return None

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    return _verify_exact_fast(U_np, V_np, W_np, n, m, p)

def _round_to_half_integer(arr):
    """Round array values to nearest half-integer (multiple of 0.5)."""
    return np.round(arr * 2) / 2

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=5000):
    """Try to reduce rank to target_rank by finding linear combinations.
    Uses a greedy approach: try to express one term as a combination of others."""
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    current_rank = rank
    
    for it in range(max_iters):
        if current_rank <= target_rank:
            break
        
        r_elim = random.randint(0, current_rank - 1)
        
        if current_rank - r_elim < 2:
            continue
        
        r_a = random.randint(0, current_rank - 1)
        if r_a == r_elim:
            r_a = (r_a + 1) % current_rank
        r_b = random.randint(0, current_rank - 1)
        if r_b == r_elim or r_b == r_a:
            r_b = (r_b + 1) % current_rank
            if r_b == r_elim:
                r_b = (r_b + 1) % current_rank
        
        U_a = best_U[:, r_a]
        U_b = best_U[:, r_b]
        U_e = best_U[:, r_elim]
        
        V_a = best_V[:, r_a]
        V_b = best_V[:, r_b]
        V_e = best_V[:, r_elim]
        
        W_a = best_W[:, r_a]
        W_b = best_W[:, r_b]
        W_e = best_W[:, r_elim]
        
        A_U = np.column_stack([U_a, U_b])
        if np.linalg.matrix_rank(A_U) < 2:
            continue
        try:
            coeff_U, _, _, _ = np.linalg.lstsq(A_U, U_e, rcond=None)
        except:
            continue
        alpha, beta = coeff_U
        
        V_pred = alpha * V_a + beta * V_b
        W_pred = alpha * W_a + beta * W_b
        
        if np.linalg.norm(V_pred - V_e) > 1e-6 or np.linalg.norm(W_pred - W_e) > 1e-6:
            continue
        
        new_U = np.delete(best_U, r_elim, axis=1)
        new_V = np.delete(best_V, r_elim, axis=1)
        new_W = np.delete(best_W, r_elim, axis=1)
        
        best_U, best_V, best_W = new_U, new_V, new_W
        current_rank -= 1
    
    if current_rank <= target_rank and _verify_exact(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    # Initialize with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(dim_U)]
    V = [[0] * rank_schoolbook for _ in range(dim_V)]
    W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Verify schoolbook is correct
    if not _verify_exact(U, V, W, n, m, p):
        return best_result
    
    # Try ALS-based rank reduction
    # Known: rank of 2x3 * 3x4 matmul is 18
    # Try ranks from 18 upward, then fall back to higher ranks
    for target_rank in range(18, rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = _try_rank_als(target_rank, n, m, p, max_restarts=5, max_iters=80, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                actual_rank = len(U_r[0])
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                break
    
    # If ALS didn't find anything, try the original greedy reduction
    if best_rank == rank_schoolbook:
        for target_rank in range(rank_schoolbook - 1, 17, -1):
            if time.time() > deadline:
                break
            
            result = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=2000)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_exact(U_r, V_r, W_r, n, m, p):
                    actual_rank = len(U_r[0])
                    if actual_rank < best_rank:
                        best_rank = actual_rank
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
                    break
    
    # Final verification
    if _verify_exact(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    else:
        return {"U": U, "V": V, "W": W}
