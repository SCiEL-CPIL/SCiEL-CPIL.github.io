"""Matrix multiplication 2,3,6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START


def _verify(U, V, W, n, m, p, r):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    """
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    T = np.zeros((n, m, p))
    for c in range(r):
        # For fixed c: contribution is U_r[:,:,c] (n,m) * V_r[:,:,c] (m,p) * W_r[:,:,c] (p,n)
        # T_c[i,j,k] = U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is an outer product with index matching
        for i in range(n):
            for j in range(m):
                u_val = U_r[i, j, c]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V_r[j, k, c]
                    if v_val == 0:
                        continue
                    w_val = W_r[k, i, c]
                    T[i, j, k] += u_val * v_val * w_val
    return np.all(T == 1)


def _verify_fast(U, V, W, n, m, p, r):
    """Faster vectorized verification."""
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    # P[j,k,i,c] = V_r[j,k,c] * W_r[k,i,c]
    P = np.einsum('jkc,kic->jkic', V_r, W_r)
    T = np.einsum('ijc,jkic->ijk', U_r, P)
    return np.all(T == 1)


def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9


def _all_half_integers(U, V, W):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < 1e-9):
            return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _try_exhaustive_small(n, m, p, r, deadline):
    """For very small r, try systematic search over {-1, 0, 1} patterns."""
    if r > 8:  # Too many combinations
        return None
    
    # For small r, we can try structured patterns
    # Key insight: each column c defines a rank-1 tensor
    # U_col (n*m), V_col (m*p), W_col (p*n)
    # Product: T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Try patterns where U, V, W have specific sparsity
    # For 2x3x6, n*m=6, m*p=18, p*n=12
    
    # Strategy: try all combinations of small structured patterns
    # This is feasible for r <= 6 or so
    
    values = [-1.0, 0.0, 1.0]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each column, the pattern is determined by U_col, V_col, W_col
    # We need sum over columns of the product to equal 1
    
    # Try a structured approach: 
    # For 2x3x6, try to find rank-4 or rank-5 decompositions
    # by exploiting the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try specific known patterns for small matrix multiplication
    # For 2x1x1: rank 2 (schoolbook) or rank 3 (with cancellation)
    # Actually 2x1x1 has rank 2
    
    # For our case, let's try a different approach:
    # Use the fact that the tensor is separable in a block structure
    
    # Try rank r with each column being a "simple" pattern
    # Simple patterns for U: one of 2^6 = 64 binary patterns (with signs)
    # This is still too many for r > 3
    
    # Instead, try a greedy construction
    # Start with zero tensor, add columns one by one to reduce error
    
    T_current = np.zeros((n, m, p))
    T_target = np.ones((n, m, p))
    
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    for c in range(r):
        if time.time() > deadline:
            return None
        
        # Find the pattern that best reduces the remaining error
        # Remaining error: T_target - T_current
        remaining = T_target - T_current
        
        # For a rank-1 addition with U_col, V_col, W_col:
        # contribution[i,j,k] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        # We want to maximize correlation with remaining
        
        # Try small structured patterns
        best_pattern = None
        best_score = -1e9
        
        # Try patterns with limited support
        # For U_col: try all subsets of size 1 or 2 from {0,...,5}
        # For each, try corresponding V_col, W_col patterns
        
        u_supports = []
        for size in [1, 2]:
            from itertools import combinations
            for combo in combinations(range(nm), size):
                u_supports.append(set(combo))
        
        # Limit the search
        if len(u_supports) > 50:
            u_supports = u_supports[:50]
        
        for u_supp in u_supports:
            u_vals = {}
            for idx in u_supp:
                u_vals[idx] = 1.0  # Try positive first
            
            # For V: find indices that connect to u_supp
            u_indices = list(u_supp)
            possible_v_indices = set()
            for u_idx in u_indices:
                i = u_idx // m
                j = u_idx % m
                for k in range(p):
                    possible_v_indices.add(j * p + k)
            
            if not possible_v_indices:
                continue
            
            v_supports = []
            v_list = list(possible_v_indices)
            for size in [1, 2, 3]:
                from itertools import combinations
                if size <= len(v_list):
                    for combo in combinations(v_list, size):
                        v_supports.append(set(combo))
            
            if len(v_supports) > 30:
                v_supports = v_supports[:30]
            
            for v_supp in v_supports:
                # Determine W support from u_supp and v_supp
                possible_w_indices = set()
                for u_idx in u_supp:
                    i = u_idx // m
                    j = u_idx % m
                    for v_idx in v_supp:
                        jj = v_idx // p
                        k = v_idx % p
                        if jj == j:
                            possible_w_indices.add(k * n + i)
                
                if not possible_w_indices:
                    continue
                
                w_list = list(possible_w_indices)
                for w_size in [1, 2, 3]:
                    if w_size > len(w_list):
                        continue
                    from itertools import combinations
                    for w_combo in combinations(w_list, w_size):
                        w_supp = set(w_combo)
                        
                        # Check if this pattern gives positive correlation with remaining
                        # Build the contribution tensor
                        contrib = np.zeros((n, m, p))
                        for u_idx in u_supp:
                            i = u_idx // m
                            j = u_idx % m
                            u_val = u_vals.get(u_idx, 0)
                            if u_val == 0:
                                continue
                            for v_idx in v_supp:
                                jj = v_idx // p
                                k = v_idx % p
                                if jj != j:
                                    continue
                                v_val = 1.0  # Assume positive for now
                                w_idx = k * n + i
                                if w_idx in w_supp:
                                    w_val = 1.0
                                    contrib[i, j, k] += u_val * v_val * w_val
                        
                        score = np.dot(contrib.flatten(), remaining.flatten())
                        if score > best_score:
                            best_score = score
                            best_pattern = (u_supp, v_supp, w_supp)
        
        if best_pattern is None:
            continue
        
        u_supp, v_supp, w_supp = best_pattern
        
        # Set the values (all 1.0 for now)
        for idx in u_supp:
            U[idx, c] = 1.0
        for idx in v_supp:
            V[idx, c] = 1.0
        for idx in w_supp:
            W[idx, c] = 1.0
        
        # Update T_current
        for u_idx in u_supp:
            i = u_idx // m
            j = u_idx % m
            for v_idx in v_supp:
                jj = v_idx // p
                k = v_idx % p
                if jj == j:
                    w_idx = k * n + i
                    if w_idx in w_supp:
                        T_current[i, j, k] += 1.0
    
    # Check if we achieved the target
    if _verify(U, V, W, n, m, p, r) and _all_half_integers(U, V, W):
        return U, V, W
    
    return None


def _greedy_construct(n, m, p, r, deadline):
    """Greedy column-by-column construction."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    T_current = np.zeros((n, m, p))
    T_target = np.ones((n, m, p))
    
    for c in range(r):
        if time.time() > deadline:
            return None
        
        remaining = T_target - T_current
        
        # Try to find a good rank-1 pattern for this column
        # Use random sampling with local optimization
        
        best_U = None
        best_V = None
        best_W = None
        best_score = -1e9
        
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            # Random initialization with sparsity
            u_sparse = np.random.randint(0, 2, size=nm)
            v_sparse = np.random.randint(0, 2, size=mp)
            w_sparse = np.random.randint(0, 2, size=pn)
            
            # Ensure some connectivity
            if np.sum(u_sparse) == 0 or np.sum(v_sparse) == 0 or np.sum(w_sparse) == 0:
                continue
            
            # Assign signs
            u_vals = u_sparse * np.random.choice([-1.0, 1.0], size=nm)
            v_vals = v_sparse * np.random.choice([-1.0, 1.0], size=mp)
            w_vals = w_sparse * np.random.choice([-1.0, 1.0], size=pn)
            
            # Compute contribution
            contrib = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    u_val = u_vals[i * m + j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = v_vals[j * p + k]
                        if v_val == 0:
                            continue
                        w_val = w_vals[k * n + i]
                        if w_val == 0:
                            continue
                        contrib[i, j, k] += u_val * v_val * w_val
            
            score = np.dot(contrib.flatten(), remaining.flatten())
            if score > best_score:
                best_score = score
                best_U = u_vals.copy()
                best_V = v_vals.copy()
                best_W = w_vals.copy()
        
        if best_U is None:
            continue
        
        U[:, c] = best_U
        V[:, c] = best_V
        W[:, c] = best_W
        
        # Update T_current
        for i in range(n):
            for j in range(m):
                u_val = best_U[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = best_V[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = best_W[k * n + i]
                    if w_val == 0:
                        continue
                    T_current[i, j, k] += u_val * v_val * w_val
    
    if _verify(U, V, W, n, m, p, r) and _all_half_integers(U, V, W):
        return U, V, W
    
    return None


def _refine(U, V, W, n, m, p, r, deadline):
    """Refine a near-miss decomposition by flipping entries."""
    if not _verify(U, V, W, n, m, p, r):
        # Check how close we are
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        P = np.einsum('jkc,kic->jkic', V_r, W_r)
        T = np.einsum('ijc,jkic->ijk', U_r, P)
        errors = T - 1
        
        if np.max(np.abs(errors)) > 0.5:
            return None  # Too far from valid
        
        # Try to fix by flipping entries
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            P = np.einsum('jkc,kic->jkic', V_r, W_r)
            T = np.einsum('ijc,jkic->ijk', U_r, P)
            errors = T - 1
            
            if np.all(np.abs(errors) < 1e-9):
                if _verify(U, V, W, n, m, p, r) and _all_half_integers(U, V, W):
                    return U, V, W
                break
            
            # Find the largest error
            max_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
            i, j, k = max_idx
            err = errors[i, j, k]
            
            if abs(err) < 1e-12:
                continue
            
            # Try flipping a random entry in a random column
            c = np.random.randint(r)
            
            # Try flipping U[i*m+j, c]
            old_u = U[i*m+j, c]
            if old_u != 0:
                U[i*m+j, c] = -old_u
                # Check if error improved
                U_r2 = U.reshape(n, m, r)
                V_r2 = V.reshape(m, p, r)
                W_r2 = W.reshape(p, n, r)
                P2 = np.einsum('jkc,kic->jkic', V_r2, W_r2)
                T2 = np.einsum('ijc,jkic->ijk', U_r2, P2)
                err2 = T2[i, j, k] - 1
                if abs(err2) < abs(err):
                    # Keep the flip
                    pass
                else:
                    # Revert
                    U[i*m+j, c] = old_u
            
            # Try flipping V[j*p+k, c]
            old_v = V[j*p+k, c]
            if old_v != 0:
                V[j*p+k, c] = -old_v
                U_r2 = U.reshape(n, m, r)
                V_r2 = V.reshape(m, p, r)
                W_r2 = W.reshape(p, n, r)
                P2 = np.einsum('jkc,kic->jkic', V_r2, W_r2)
                T2 = np.einsum('ijc,jkic->ijk', U_r2, P2)
                err2 = T2[i, j, k] - 1
                if abs(err2) < abs(err):
                    pass
                else:
                    V[j*p+k, c] = old_v
            
            # Try flipping W[k*n+i, c]
            old_w = W[k*n+i, c]
            if old_w != 0:
                W[k*n+i, c] = -old_w
                U_r2 = U.reshape(n, m, r)
                V_r2 = V.reshape(m, p, r)
                W_r2 = W.reshape(p, n, r)
                P2 = np.einsum('jkc,kic->jkic', V_r2, W_r2)
                T2 = np.einsum('ijc,jkic->ijk', U_r2, P2)
                err2 = T2[i, j, k] - 1
                if abs(err2) < abs(err):
                    pass
                else:
                    W[k*n+i, c] = old_w
        
        # Final check
        if _verify(U, V, W, n, m, p, r) and _all_half_integers(U, V, W):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    start_time = time.time()
    deadline = start_time + 135  # Leave margin for 150s limit
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    best_rank = rank_sb
    
    # Try ranks from low to high
    # Theoretical: rank must be at least max(n*p, m*p, n*m) / something
    # For 2x3x6: schoolbook is 36
    # Try: 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36
    
    candidate_ranks = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
                       27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        if r >= best_rank:
            continue
        
        # Try greedy construction
        result = _greedy_construct(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, r)
            best_rank = r
            continue
        
        # Try refinement from random initialization
        if time.time() < deadline - 10:
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                U = np.random.choice([-1.0, 0.0, 1.0], size=(n*m, r))
                V = np.random.choice([-1.0, 0.0, 1.0], size=(m*p, r))
                W = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, r))
                
                refined = _refine(U, V, W, n, m, p, r, deadline)
                if refined is not None:
                    U, V = refined
                    best_result = (U, V, W, r)
                    best_rank = r
                    break
    
    U, V, W, rank = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


# EVOLVE-BLOCK-END
