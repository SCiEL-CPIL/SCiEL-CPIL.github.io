"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor_support(n=3, m=3, p=6):
    """
    Build the list of (a, b, c) indices that are 'on' in the tensor.
    a = i*m + j
    b = j*p + k
    c = k*n + i
    """
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    return support

def _compute_tensor_value(U, V, W, a, b, c):
    """Compute the value of the decomposed tensor at (a, b, c)."""
    return np.dot(U[a, :], V[b, :] * W[c, :])

def _compute_error(U, V, W, n=3, m=3, p=6, support=None, check_offdiag=True, sample_size=500):
    """
    Compute the error of the decomposition.
    Error = sum over support (val - 1)^2 + sum over off-support val^2
    """
    if support is None:
        support = _build_tensor_support(n, m, p)
    
    nm, mp, pn = n * m, m * p, p * n
    err = 0.0
    
    # Check support
    for a, b, c in support:
        val = _compute_tensor_value(U, V, W, a, b, c)
        err += (val - 1.0) ** 2
    
    # Check off-support (sampled for speed)
    if check_offdiag:
        # We need to check (a, b, c) NOT in support.
        # Total entries: nm * mp * pn = 9 * 18 * 18 = 2916.
        # Support size: n*m*p = 54.
        # Off-support size: 2916 - 54 = 2862.
        
        # For speed, we can just iterate all and skip support.
        # But 2916 checks per iteration might be slow if done many times.
        # Let's optimize: precompute a set of support tuples.
        # Actually, the current implementation in _verify does full check.
        # For ALS, we want speed.
        
        # Let's just do a full check but vectorized?
        # T_decomp[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
        # This is a 3-way contraction.
        # We can compute the full tensor T_decomp (9x18x18) and compare with target.
        
        # Construct Target Tensor
        T_target = np.zeros((nm, mp, pn))
        for a, b, c in support:
            T_target[a, b, c] = 1.0
        
        # Compute T_decomp
        # T_decomp[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
        # This can be computed as:
        # For each d, contribution is U[:, d] @ V[:, d].T @ W[:, d] ... no.
        # T_decomp = sum_d np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
        
        # Let's use einsum for the whole thing?
        # T_decomp = np.einsum('ad,bd,cd->abc', U, V, W)
        
        T_decomp = np.einsum('ad,bd,cd->abc', U, V, W)
        
        # Error is sum of squared differences
        err += np.sum((T_decomp - T_target) ** 2)
        
        # Note: The loop above already added error for support entries.
        # If we use the full tensor check, we should NOT double count.
        # Let's refactor: just use the full tensor check for everything.
        
        # Recompute err from scratch using full tensor to avoid double counting logic errors
        # Actually, the loop above was specific. Let's just return the full tensor error.
        # It's cleaner.
        return float(np.sum((T_decomp - T_target) ** 2))
    
    return err

def _als_step_U(U, V, W, n, m, p, support):
    """Update U given V and W."""
    nm = n * m
    new_U = np.zeros_like(U)
    rank = U.shape[1]
    
    # For each row a of U, we need to solve:
    # sum_{(a,b,c) in support} U[a, :] * (V[b, :] * W[c, :]) = 1
    # This is a least squares problem: M @ U[a, :] = ones
    # where M is a matrix with rows V[b, :] * W[c, :] for all (b, c) such that (a, b, c) in support.
    
    for a in range(nm):
        rows = []
        for i in range(n):
            for j in range(m):
                if i * m + j == a:
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        rows.append(V[b, :] * W[c, :])
        
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_U[a, :] = U[a, :]
        else:
            new_U[a, :] = U[a, :]
            
    return new_U

def _als_step_V(U, V, W, n, m, p, support):
    """Update V given U and W."""
    mp = m * p
    new_V = np.zeros_like(V)
    
    for b in range(mp):
        rows = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j * p + k == b:
                        a = i * m + j
                        c = k * n + i
                        rows.append(U[a, :] * W[c, :])
        
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_V[b, :] = V[b, :]
        else:
            new_V[b, :] = V[b, :]
            
    return new_V

def _als_step_W(U, V, W, n, m, p, support):
    """Update W given U and V."""
    pn = p * n
    new_W = np.zeros_like(W)
    
    for c in range(pn):
        rows = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if k * n + i == c:
                        a = i * m + j
                        b = j * p + k
                        rows.append(U[a, :] * V[b, :])
        
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_W[c, :] = W[c, :]
        else:
            new_W[c, :] = W[c, :]
            
    return new_W

def _try_als_decomposition(n=3, m=3, p=6, deadline=None):
    """
    Try to find a low-rank decomposition using ALS with various initializations.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    support = _build_tensor_support(n, m, p)
    
    best_result = None
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But the actual tensor rank is likely higher.
    # For 3x3 * 3x6, the rank is at least 18 (from dimension constraints).
    # Let's try ranks from 18 upward.
    
    for r in range(18, 40):
        if time.time() > deadline - 10:
            break
        
        n_restarts = max(2, int((deadline - time.time()) / 15))
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initializations
            if restart == 0:
                # Sparse random initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for col in range(r):
                    n_u = np.random.randint(1, 4)
                    n_v = np.random.randint(1, 4)
                    n_w = np.random.randint(1, 4)
                    u_idx = np.random.choice(nm, n_u, replace=False)
                    v_idx = np.random.choice(mp, n_v, replace=False)
                    w_idx = np.random.choice(pn, n_w, replace=False)
                    U[u_idx, col] = np.random.choice([-1, 1], n_u)
                    V[v_idx, col] = np.random.choice([-1, 1], n_v)
                    W[w_idx, col] = np.random.choice([-1, 1], n_w)
            elif restart == 1:
                # Dense random initialization
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
            else:
                # Mixed initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(pn, r) * 0.3
                for a in range(nm):
                    i_a = a // m
                    U[a, :] *= (1.0 + 0.1 * i_a)
                for b in range(mp):
                    j_b = b // p
                    V[b, :] *= (1.0 + 0.1 * j_b)
                for c in range(pn):
                    k_c = c // n
                    W[c, :] *= (1.0 + 0.1 * k_c)
            
            # ALS iterations
            max_iter = 80
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                U = _als_step_U(U, V, W, n, m, p, support)
                
                # Update V
                V = _als_step_V(U, V, W, n, m, p, support)
                
                # Update W
                W = _als_step_W(U, V, W, n, m, p, support)
                
                # Compute error
                err = _compute_error(U, V, W, n, m, p, support)
                
                if err < 1e-6:
                    # Verify exactly
                    if _verify(U, V, W, n, m, p):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to find better
        better = _try_als_decomposition(n, m, p, deadline)
        if better is not None:
            rank_better = len(better["U"][0]) if better["U"] else 0
            rank_current = len(result["U"][0]) if result["U"] else 0
            if rank_better < rank_current:
                result = better
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
