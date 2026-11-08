import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def _verify_fast(U, V, W, n, m, p):
    """
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Returns True if decomposition is exact.
    """
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    target = np.ones((n, m, p))
    return np.array_equal(T_prime, target)

def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors."""
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    target = np.ones((n, m, p))
    return np.sum((T_prime - target) ** 2)

def _local_search(U, V, W, n, m, p, max_steps=1000, rng=None):
    """
    Perform local search to minimize error.
    Returns (U, V, W, success)
    """
    if rng is None:
        rng = np.random.RandomState()
        
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    target = np.ones((n, m, p))
    
    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    error = np.sum((T_prime - target) ** 2)
    
    if error == 0:
        return U, V, W, True

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = error

    for step in range(max_steps):
        c = rng.randint(r)
        mat_idx = rng.randint(3)
        
        if mat_idx == 0:
            idx = rng.randint(n * m)
            i, j = divmod(idx, m)
            old_val = U_3d[i, j, c]
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                U_3d[i, j, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
            U_3d[i, j, c] = best_new_val
            error = best_new_error
            
        elif mat_idx == 1:
            idx = rng.randint(m * p)
            j, k = divmod(idx, p)
            old_val = V_3d[j, k, c]
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                V_3d[j, k, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
                V_3d[j, k, c] = old_val
            V_3d[j, k, c] = best_new_val
            error = best_new_error
            
        else:
            idx = rng.randint(p * n)
            k, i = divmod(idx, n)
            old_val = W_3d[k, i, c]
            candidates = [old_val + 1, old_val - 1, 0, 1, -1]
            candidates = list(set(candidates))
            best_new_val = old_val
            best_new_error = error
            for new_val in candidates:
                W_3d[k, i, c] = new_val
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                new_error = np.sum((T_prime - target) ** 2)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = new_val
                W_3d[k, i, c] = old_val
            W_3d[k, i, c] = best_new_val
            error = best_new_error

        if error == 0:
            return U_3d.reshape(n*m, r), V_3d.reshape(m*p, r), W_3d.reshape(p*n, r), True

        if error < best_error:
            best_error = error
            best_U = U_3d.copy()
            best_V = V_3d.copy()
            best_W = W_3d.copy()
            
    return best_U, best_V, best_W, False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _structured_init(n, m, p, r, rng):
    """
    Structured initialization: try to cover triples in groups.
    Each column is assigned to cover a subset of (i,j,k) triples.
    """
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Generate all triples
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    rng.shuffle(all_triples)
    
    # Assign triples to columns round-robin
    for c in range(r):
        # Each column gets ~n*m*p/r triples
        start = c * (len(all_triples) // r)
        end = start + (len(all_triples) // r) + (1 if c < len(all_triples) % r else 0)
        triples_for_col = all_triples[start:end]
        
        # For each triple in this column, set the corresponding entries
        # But we need to be careful: setting U[i,j,c]=1 for multiple (i,j) means
        # this column will contribute to multiple (i,j,k) combinations.
        
        # Strategy: for each triple (i,j,k), set U[i,j,c], V[j,k,c], W[k,i,c] to 1
        # But if multiple triples share the same (i,j), we set U[i,j,c]=1 once.
        
        used_U = set()
        used_V = set()
        used_W = set()
        
        for (i, j, k) in triples_for_col:
            U_idx = i * m + j
            V_idx = j * p + k
            W_idx = k * n + i
            
            if U_idx not in used_U:
                U[U_idx, c] = 1
                used_U.add(U_idx)
            if V_idx not in used_V:
                V[V_idx, c] = 1
                used_V.add(V_idx)
            if W_idx not in used_W:
                W[W_idx, c] = 1
                used_W.add(W_idx)
    
    return U, V, W

def _random_sparse_init(n, m, p, r, rng, density=0.15):
    """Random sparse initialization."""
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for c in range(r):
        # U: ~density fraction non-zero
        for idx in range(n*m):
            if rng.rand() < density:
                U[idx, c] = rng.choice([-1, 1])
        # V
        for idx in range(m*p):
            if rng.rand() < density:
                V[idx, c] = rng.choice([-1, 1])
        # W
        for idx in range(p*n):
            if rng.rand() < density:
                W[idx, c] = rng.choice([-1, 1])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115s safety margin
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try ranks from 55 down to 30
    ranks_to_try = list(range(55, 29, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(restart * 1000 + r)
            
            # Alternate between structured and random initialization
            if restart % 2 == 0:
                U, V, W = _structured_init(n, m, p, r, rng)
            else:
                U, V, W = _random_sparse_init(n, m, p, r, rng, density=0.12)
            
            # Quick check
            if _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
            
            # Local search
            U_ls, V_ls, W_ls, success = _local_search(U, V, W, n, m, p, max_steps=300, rng=rng)
            if success:
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                best_rank = r
                found = True
                break
                
        if found:
            # Found at rank r, continue to try lower ranks
            pass

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
