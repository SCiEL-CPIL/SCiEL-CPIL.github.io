import time
import sys
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rm, rank = U.shape
    sm, rank2 = V.shape
    tm, rank3 = W.shape
    
    if rank != rank2 or rank != rank3:
        return False
    if rm != n * m:
        return False
    if sm != m * p:
        return False
    if tm != p * n:
        return False
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d] for all entries
    # T = np.einsum('ad,bd,cd->abc', U, V, W)
    T = np.einsum('ad,bd,cd->abc', U, V, W)
    
    # Build target: T_target[a,b,c] = 1 if consistent (i,j,k), else 0
    # a = i*m+j, b = j*p+k, c = k*n+i
    # Check on-diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal
    # Build mask of on-diagonal positions
    on_diag = np.zeros((nm, mp, pn), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on_diag[a, b, c] = True
    
    # Check all off-diagonal entries are ~0
    off_mask = ~on_diag
    if np.any(np.abs(T[off_mask]) > 1e-9):
        return False
    
    return True


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    best_rank = n * m * p  # Schoolbook: 54
    best_result = None
    
    # Start with schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if _verify(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    import random
    
    # Precompute on-diagonal indices for fast error computation
    on_diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                on_diag_indices.append((i * m + j, j * p + k, k * n + i))
    
    # Precompute off-diagonal indices (all except on-diag)
    on_diag_set = set(on_diag_indices)
    off_diag_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in on_diag_set:
                    off_diag_indices.append((a, b, c))
    
    # Precompute as arrays for vectorized operations
    on_diag_a = np.array([x[0] for x in on_diag_indices])
    on_diag_b = np.array([x[1] for x in on_diag_indices])
    on_diag_c = np.array([x[2] for x in on_diag_indices])
    
    off_diag_a = np.array([x[0] for x in off_diag_indices])
    off_diag_b = np.array([x[1] for x in off_diag_indices])
    off_diag_c = np.array([x[2] for x in off_diag_indices])
    
    def compute_error_vec(U_arr, V_arr, W_arr):
        """Vectorized error computation using numpy."""
        # On-diagonal: sum_d U[a,d]*V[b,d]*W[c,d] should be 1
        # Use einsum for batch computation
        U_od = U_arr[on_diag_a, :]  # shape (54, r)
        V_od = V_arr[on_diag_b, :]
        W_od = W_arr[on_diag_c, :]
        on_vals = np.sum(U_od * V_od * W_od, axis=1)
        on_err = np.sum((on_vals - 1.0) ** 2)
        
        # Off-diagonal: should be 0
        # Sample if too many
        if len(off_diag_a) > 500:
            idx = np.random.choice(len(off_diag_a), 500, replace=False)
            od_a = off_diag_a[idx]
            od_b = off_diag_b[idx]
            od_c = off_diag_c[idx]
        else:
            od_a = off_diag_a
            od_b = off_diag_b
            od_c = off_diag_c
        
        U_od2 = U_arr[od_a, :]
        V_od2 = V_arr[od_b, :]
        W_od2 = W_arr[od_c, :]
        off_vals = np.sum(U_od2 * V_od2 * W_od2, axis=1)
        off_err = np.sum(off_vals ** 2)
        
        return on_err + off_err
    
    def try_rank(r, max_iter=100, seed=42, n_attempts=20):
        """Try to find a rank-r decomposition using vectorized random search and refinement."""
        rng = random.Random(seed)
        
        def random_entry():
            return rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2])
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            U_cand = np.array([[random_entry() for _ in range(r)] for _ in range(nm)])
            V_cand = np.array([[random_entry() for _ in range(r)] for _ in range(mp)])
            W_cand = np.array([[random_entry() for _ in range(r)] for _ in range(pn)])
            
            err0 = compute_error_vec(U_cand, V_cand, W_cand)
            if err0 < 1e-9:
                return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
            
            # Local search: perturb one entry at a time
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for iter_i in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    mat_idx = rng.choice([0, 1, 2])
                    if mat_idx == 0:
                        mat = U_cand
                        nrows = nm
                    elif mat_idx == 1:
                        mat = V_cand
                        nrows = mp
                    else:
                        mat = W_cand
                        nrows = pn
                    
                    row = rng.randint(0, nrows - 1)
                    col = rng.randint(0, r - 1)
                    old_val = mat[row, col]
                    
                    # Try a few neighbor values
                    candidates = [old_val + d for d in [-2, -1, -0.5, 0.5, 1, 2]]
                    candidates += [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
                    
                    best_err = err0
                    best_val = old_val
                    for cand in candidates:
                        mat[row, col] = cand
                        err = compute_error_vec(U_cand, V_cand, W_cand)
                        if err < best_err:
                            best_err = err
                            best_val = cand
                    
                    mat[row, col] = best_val
                    if best_err < err0:
                        err0 = best_err
                        improved = True
                        if err0 < 1e-9:
                            return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
            
            if err0 < 1e-9:
                return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
        
        return None
    
    # Try structured approach first: split into two 3x3x3 problems
    # C = A * B where B = [B1 | B2], each 3x3
    # If we can find rank-r1 for 3x3x3, total rank is 2*r1
    # But we can share U factors: use same U for both halves
    # Actually the U factor depends on (i,j) which is the same for both
    # So we can share U columns!
    
    # Let's try: for each half, we need U[i*m+j, d] * V1[j*p+k, d] * W1[k*n+i, d] for first half
    # and U[i*m+j, d'] * V2[j*p+3+k, d'] * W2[(3+k)*n+i, d'] for second half
    # If we use the same U for both: rank = r1 + r1 but U is shared
    # Actually no, U is the same matrix, so we can have:
    # First r1 columns: handle B1
    # Next r1 columns: handle B2
    # Total rank = 2 * r1
    
    # For 3x3x3, best known is 19, so total would be 38
    # But maybe we can do better with a joint decomposition
    
    # Try ranks
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for r in sorted(ranks_to_try):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
        
        print(f"Trying rank {r}...", file=sys.stderr)
        result = try_rank(r, max_iter=50, seed=r, n_attempts=15)
        if result is not None:
            U_cand, V_cand, W_cand, r_found = result
            if _verify(U_cand, V_cand, W_cand, n, m, p):
                print(f"Found valid rank {r_found}!", file=sys.stderr)
                best_rank = r_found
                best_result = {"U": U_cand, "V": V_cand, "W": W_cand}
                break
        else:
            print(f"Rank {r}: no solution found in time", file=sys.stderr)
    
    # If we didn't find anything better, return the schoolbook
    if best_result is None:
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
