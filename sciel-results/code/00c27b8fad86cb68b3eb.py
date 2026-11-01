import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the Strassen rank-28 solution
    U, V, W = _build_strassen_28()
    
    # Try to reduce rank via systematic search
    result = _try_reduce_rank(U, V, W, deadline)
    if result is not None:
        return result
    
    # Fallback to rank 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    # Compute T_hat[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (4, 16, rank)
    T_hat = np.einsum('abr,cr->abc', UV, W)  # (4, 16, 16)
    
    # Check non-zero entries: T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                if abs(T_hat[a, b, c] - 1) > tol:
                    return False
    
    # Check zero entries
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                i = a // 2
                j = a % 2
                j2 = b // 8
                k = b % 8
                k2 = c // 2
                i2 = c % 2
                if not (i == i2 and j == j2 and k == k2):
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True

def _try_reduce_rank(U, V, W, deadline):
    """Try to find a lower-rank decomposition using structured search."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    rank = U.shape[1]
    
    # Flatten each rank-1 term to a vector
    terms = []
    for r in range(rank):
        u_r = U[:, r]  # (4,)
        v_r = V[:, r]  # (16,)
        w_r = W[:, r]  # (16,)
        tensor_r = np.einsum('a,b,c->abc', u_r, v_r, w_r)  # (4,16,16)
        terms.append(tensor_r.flatten())
    
    terms = np.array(terms)  # (rank, 1024)
    
    # Check if the 28 terms are linearly independent
    term_rank = np.linalg.matrix_rank(terms, tol=1e-10)
    
    if term_rank < rank:
        # There's a linear dependency - try to exploit it
        _, S, Vt = np.linalg.svd(terms, full_matrices=False)
        null_mask = S < 1e-10
        if np.any(null_mask):
            null_vecs = Vt[null_mask, :]
            for nv in null_vecs:
                j = np.argmax(np.abs(nv))
                if abs(nv[j]) > 1e-10:
                    # Try removing term j and solving for replacement
                    result = _try_remove_term(U, V, W, j, terms, deadline)
                    if result is not None:
                        return result
    
    # Structured search: try all pairs of terms to see if one can be eliminated
    # by expressing it as a combination of others with integer/half-integer coefficients
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 10:
            break
        result = _structured_search(target_rank, U, V, W, deadline)
        if result is not None:
            return result
    
    return None

def _try_remove_term(U, V, W, j, terms, deadline):
    """Try to remove term j by finding if it's a linear combination of others."""
    # This is complex; skip for now
    return None

def _structured_search(target_rank, U, V, W, deadline):
    """Search for a valid decomposition of given rank using structured perturbations."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    original_rank = U.shape[1]
    
    # Non-zero indices
    nonzero_set = set()
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_set.add((a, b, c))
                nonzero_indices.append((a, b, c))
    
    # Try removing one term at a time and see if the remaining terms still span
    # the multiplication tensor
    for remove_idx in range(original_rank):
        if time.time() - start > deadline - 5:
            break
        
        # Remove term remove_idx
        mask = np.ones(original_rank, dtype=bool)
        mask[remove_idx] = False
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if the remaining terms still decompose the tensor
        if _verify(U_sub, V_sub, W_sub, tol=1e-9):
            return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    
    # Try more aggressive: check if any 27-term subset works
    # This is combinatorial, so sample
    rng = np.random.default_rng(seed=42)
    for trial in range(50):
        if time.time() - start > deadline - 3:
            break
        indices = list(range(original_rank))
        rng.shuffle(indices)
        keep = indices[:target_rank]
        keep.sort()
        U_sub = U[:, keep]
        V_sub = V[:, keep]
        W_sub = W[:, keep]
        
        if _verify(U_sub, V_sub, W_sub, tol=1e-9):
            return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    
    return None

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts with half-integers."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    # Non-zero indices
    nonzero_indices = []
    nonzero_set = set()
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
                nonzero_set.add((a, b, c))
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart)
        # Use half-integers: values in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
        U = (rng.integers(-3, 4, size=(num_a, target_rank)) * 0.5)
        V = (rng.integers(-3, 4, size=(num_b, target_rank)) * 0.5)
        W = (rng.integers(-3, 4, size=(num_c, target_rank)) * 0.5)
        
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)
        
        # Check non-zero entries
        max_err = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat[a, b, c] - 1)
            max_err = max(max_err, err)
        
        if max_err < 1e-6:
            # Check zero entries (sample)
            all_zero_ok = True
            for a in range(num_a):
                for b in range(num_b):
                    for c in range(num_c):
                        if (a, b, c) not in nonzero_set:
                            if abs(T_hat[a, b, c]) > 1e-6:
                                all_zero_ok = False
                                break
                    if not all_zero_ok:
                        break
                if not all_zero_ok:
                    break
            
            if all_zero_ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
