import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_rank21_strassen(n, m, p):
    """Build the rank-21 Strassen decomposition."""
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 0 * p + 2 * b
                elif l == 1: v_idx = 0 * p + 2 * b + 1
                elif l == 2: v_idx = 1 * p + 2 * b
                elif l == 3: v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _construct_target_tensor(n, m, p):
    """Return list of non-zero tensor entries."""
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c, 1.0))
    return non_zeros

def _als_step(U, V, W, target, n, m, p, rank):
    """One ALS iteration. Returns new U, V, W."""
    nm, mp, pn = n * m, m * p, p * n
    new_U = np.zeros_like(U)
    new_V = np.zeros_like(V)
    new_W = np.zeros_like(W)
    
    # Precompute target lookup for speed
    target_by_a = {}
    target_by_b = {}
    target_by_c = {}
    for (a, b, c, val) in target:
        target_by_a.setdefault(a, {})[(b, c)] = val
        target_by_b.setdefault(b, {})[(a, c)] = val
        target_by_c.setdefault(c, {})[(a, b)] = val
    
    # Update U
    for a in range(nm):
        A = np.zeros((mp * pn, rank))
        t = np.zeros(mp * pn)
        for (b, c), val in target_by_a.get(a, {}).items():
            t[b * pn + c] = val
        for r in range(rank):
            A[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        new_U[a, :] = np.linalg.lstsq(A, t, rcond=None)[0]
    
    # Update V
    for b in range(mp):
        A = np.zeros((nm * pn, rank))
        t = np.zeros(nm * pn)
        for (a, c), val in target_by_b.get(b, {}).items():
            t[a * pn + c] = val
        for r in range(rank):
            A[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        new_V[b, :] = np.linalg.lstsq(A, t, rcond=None)[0]
    
    # Update W
    for c in range(pn):
        A = np.zeros((nm * mp, rank))
        t = np.zeros(nm * mp)
        for (a, b), val in target_by_c.get(c, {}).items():
            t[a * mp + b] = val
        for r in range(rank):
            A[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        new_W[c, :] = np.linalg.lstsq(A, t, rcond=None)[0]
    
    return new_U, new_V, new_W

def _calculate_error(U, V, W, target):
    """Calculate RMS error."""
    error = 0.0
    for (a, b, c, val) in target:
        computed = 0.0
        for r in range(U.shape[1]):
            computed += U[a, r] * V[b, r] * W[c, r]
        error += (computed - val) ** 2
    return np.sqrt(error)

def _verify_exact(U, V, W, target, tol=1e-10):
    """Check if decomposition is exact."""
    for (a, b, c, val) in target:
        computed = 0.0
        for r in range(U.shape[1]):
            computed += U[a, r] * V[b, r] * W[c, r]
        if abs(computed - val) > tol:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target = _construct_target_tensor(n, m, p)
    
    # Fallback: Rank 21
    U21, V21, W21 = _build_rank21_strassen(n, m, p)
    best_result = {
        "U": U21.tolist(),
        "V": V21.tolist(),
        "W": W21.tolist()
    }
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try rank 20 with structured initialization from rank-21
    target_rank = 20
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        np.random.seed(12345 + restart * 7)
        
        # Strategy: take rank-21 solution, merge two columns by averaging
        # with a random mixing, then optimize
        if restart < 15:
            # Structured: take first 19 columns + 1 merged column
            merge_i = restart % 21
            merge_j = (merge_i + 7) % 21
            if merge_i == merge_j:
                merge_j = (merge_j + 1) % 21
            
            U = np.column_stack([
                U21[:, :merge_i],
                U21[:, merge_i+1:merge_j],
                0.5 * (U21[:, merge_i] + U21[:, merge_j]),
                U21[:, merge_j+1:]
            ])
            V = np.column_stack([
                V21[:, :merge_i],
                V21[:, merge_i+1:merge_j],
                0.5 * (V21[:, merge_i] + V21[:, merge_j]),
                V21[:, merge_j+1:]
            ])
            W = np.column_stack([
                W21[:, :merge_i],
                W21[:, merge_i+1:merge_j],
                0.5 * (W21[:, merge_i] + W21[:, merge_j]),
                W21[:, merge_j+1:]
            ])
        else:
            # Random perturbation of rank-21 (drop one column, add random)
            drop_idx = restart % 21
            U = np.delete(U21, drop_idx, axis=1)
            V = np.delete(V21, drop_idx, axis=1)
            W = np.delete(W21, drop_idx, axis=1)
            # Add a small random column
            U = np.column_stack([U, np.random.randn(nm) * 0.1])
            V = np.column_stack([V, np.random.randn(mp) * 0.1])
            W = np.column_stack([W, np.random.randn(pn) * 0.1])
        
        # Run ALS with periodic normalization
        best_err = float('inf')
        for it in range(200):
            if time.time() > deadline:
                break
            U, V, W = _als_step(U, V, W, target, n, m, p, target_rank)
            
            # Normalize columns to prevent scaling issues
            for r in range(target_rank):
                nu = np.linalg.norm(U[:, r])
                nv = np.linalg.norm(V[:, r])
                nw = np.linalg.norm(W[:, r])
                if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                    # Keep product of norms constant
                    prod = nu * nv * nw
                    if prod > 1e-12:
                        scale = (1.0 / prod) ** (1.0/3.0)
                        U[:, r] *= scale
                        V[:, r] *= scale
                        W[:, r] *= scale
            
            if it % 10 == 9:
                err = _calculate_error(U, V, W, target)
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        break
        
        # Final check
        err = _calculate_error(U, V, W, target)
        if err < 1e-8:
            # Polish with tighter tolerance
            for polish_it in range(50):
                if time.time() > deadline:
                    break
                U, V, W = _als_step(U, V, W, target, n, m, p, target_rank)
                for r in range(target_rank):
                    nu = np.linalg.norm(U[:, r])
                    nv = np.linalg.norm(V[:, r])
                    nw = np.linalg.norm(W[:, r])
                    if nu > 1e-12 and nv > 1e-12 and nw > 1e-12:
                        prod = nu * nv * nw
                        if prod > 1e-12:
                            scale = (1.0 / prod) ** (1.0/3.0)
                            U[:, r] *= scale
                            V[:, r] *= scale
                            W[:, r] *= scale
                if _calculate_error(U, V, W, target) < 1e-12:
                    break
            
            if _verify_exact(U, V, W, target, tol=1e-9):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # If we didn't find rank 20, return rank 21
    return best_result
