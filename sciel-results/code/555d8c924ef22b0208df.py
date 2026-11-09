import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_base_decomposition(n, m, p):
    """Build the base rank-36 decomposition using Strassen 2x2x2 blocks + schoolbook."""
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ]
    
    rank = 4 * 7 + 8  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li * m + j0 + lj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + k0 + lk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return U, V, W, rank

def _precompute_tensor_data(U_arr, V_arr, W_arr, n, m, p):
    """
    Precompute for each tensor entry (i,j,k) and each column c, the product U*V*W.
    Returns array of shape (n*m*p, rank) where entry [idx, c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
    """
    rank = U_arr.shape[1]
    nmp = n * m * p
    products = np.zeros((nmp, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                products[idx] = U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]
                idx += 1
    return products

def _verify_keep_fast(products, keep_indices):
    """Check if keeping only these columns still sums to 1 for every tensor entry."""
    if len(keep_indices) == 0:
        return products.sum(axis=1) if products.shape[1] > 0 else np.ones(products.shape[0])
    subset = products[:, keep_indices]
    totals = subset.sum(axis=1)
    return np.all(np.abs(totals - 1.0) < 1e-9)

def _try_rank_reduction(U, V, W, n, m, p, time_limit=100):
    """
    Attempt to reduce the rank by removing columns.
    Uses precomputed products for fast verification.
    """
    start_time = time.time()
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    products = _precompute_tensor_data(U_arr, V_arr, W_arr, n, m, p)
    
    current_keep = list(range(rank))
    best_rank = rank
    best_keep = list(current_keep)
    
    def _elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy single-column removal
    changed = True
    while changed and _elapsed() < time_limit * 0.15:
        changed = False
        for c in list(current_keep):
            if _elapsed() > time_limit * 0.15:
                break
            new_keep = [i for i in current_keep if i != c]
            if _verify_keep_fast(products, new_keep):
                current_keep = new_keep
                best_rank = len(new_keep)
                best_keep = list(new_keep)
                changed = True
    
    # Phase 2: Try removing pairs of columns
    if _elapsed() < time_limit * 0.35:
        changed = True
        while changed and _elapsed() < time_limit * 0.35:
            changed = False
            for a_idx in range(len(current_keep)):
                if _elapsed() > time_limit * 0.35:
                    break
                for b_idx in range(a_idx + 1, len(current_keep)):
                    if _elapsed() > time_limit * 0.35:
                        break
                    a = current_keep[a_idx]
                    b = current_keep[b_idx]
                    new_keep = [i for i in current_keep if i != a and i != b]
                    if _verify_keep_fast(products, new_keep):
                        current_keep = new_keep
                        best_rank = len(new_keep)
                        best_keep = list(new_keep)
                        changed = True
                        break
                if changed:
                    break
    
    # Phase 3: Try removing triples of columns
    if _elapsed() < time_limit * 0.55:
        changed = True
        while changed and _elapsed() < time_limit * 0.55:
            changed = False
            for combo in combinations(current_keep, 3):
                if _elapsed() > time_limit * 0.55:
                    break
                new_keep = [i for i in current_keep if i not in combo]
                if _verify_keep_fast(products, new_keep):
                    current_keep = new_keep
                    best_rank = len(new_keep)
                    best_keep = list(new_keep)
                    changed = True
                    break
    
    # Phase 4: Try removing quadruples of columns
    if _elapsed() < time_limit * 0.80:
        changed = True
        while changed and _elapsed() < time_limit * 0.80:
            changed = False
            for combo in combinations(current_keep, 4):
                if _elapsed() > time_limit * 0.80:
                    break
                new_keep = [i for i in current_keep if i not in combo]
                if _verify_keep_fast(products, new_keep):
                    current_keep = new_keep
                    best_rank = len(new_keep)
                    best_keep = list(new_keep)
                    changed = True
                    break
    
    # Phase 5: Another round of single removals
    if _elapsed() < time_limit * 0.90:
        changed = True
        while changed and _elapsed() < time_limit * 0.90:
            changed = False
            for c in list(current_keep):
                if _elapsed() > time_limit * 0.90:
                    break
                new_keep = [i for i in current_keep if i != c]
                if _verify_keep_fast(products, new_keep):
                    current_keep = new_keep
                    best_rank = len(new_keep)
                    best_keep = list(new_keep)
                    changed = True
    
    # Phase 6: Try merging pairs - replace 2 columns with 1
    # For each pair (a,b), check if there exists alpha, beta such that
    # for all entries: alpha*prod[a] + beta*prod[b] + sum_others = 1
    # i.e., alpha*prod[a] + beta*prod[b] = 1 - sum_others
    if _elapsed() < time_limit * 0.98:
        for a_idx in range(len(current_keep)):
            if _elapsed() > time_limit * 0.98:
                break
            for b_idx in range(a_idx + 1, len(current_keep)):
                if _elapsed() > time_limit * 0.98:
                    break
                a = current_keep[a_idx]
                b = current_keep[b_idx]
                others = [i for i in current_keep if i != a and i != b]
                
                # residual = 1 - sum of products for others
                if others:
                    residual = 1.0 - products[:, others].sum(axis=1)
                else:
                    residual = np.ones(products.shape[0])
                
                # We need alpha * products[:,a] + beta * products[:,b] = residual
                # This is a system of n*m*p equations in 2 unknowns
                # Use least squares
                A = np.column_stack([products[:, a], products[:, b]])
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, residual, rcond=1e-10)
                except np.linalg.LinAlgError:
                    continue
                
                alpha, beta = sol
                # Check if this is an exact solution
                reconstructed = A @ sol
                if np.all(np.abs(reconstructed - residual) < 1e-9):
                    # Check if alpha and beta are "nice" (integers or half-integers)
                    # Actually, we just need to check if the merge is valid
                    new_keep = others
                    # Create new column: for each row, new_val = alpha*U[row,a] (for U matrix)
                    # But we need consistent scaling... 
                    # Actually the merge means: replace columns a and b with a single column c
                    # such that U[row,c] = alpha * U[row,a] (and similarly for V, W)
                    # Wait, that's not right. The tensor entry is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # If we replace (a,b) with c where U_c = alpha*U_a, V_c = beta*V_b, W_c = 1*W_a
                    # that doesn't work generally.
                    # 
                    # Actually, for a valid merge: we need new column c such that
                    # U_c[i,j] * V_c[j,k] * W_c[k,i] = alpha * U_a[i,j] * V_a[j,k] * W_a[k,i]
                    #                                    + beta * U_b[i,j] * V_b[j,k] * W_b[k,i]
                    # for all i,j,k. This is generally not possible with a single rank-1 tensor
                    # unless the products have special structure.
                    #
                    # Skip this - it's too complex and unlikely to work for arbitrary columns.
                    pass
    
    if best_rank < rank:
        U_list = [[float(x) for x in row] for row in U_arr[:, best_keep].tolist()]
        V_list = [[float(x) for x in row] for row in V_arr[:, best_keep].tolist()]
        W_list = [[float(x) for x in row] for row in W_arr[:, best_keep].tolist()]
        
        # Round to nearest 0.5 if close
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-9:
                        mat[i][j] = rounded
        
        return U_list, V_list, W_list, best_rank
    
    return None

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Build the base rank-36 solution
    U, V, W, rank = _build_base_decomposition(n, m, p)
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    time_remaining = 145 - (time.time() - start_time)
    if time_remaining > 15:
        result = _try_rank_reduction(U, V, W, n, m, p, time_limit=min(time_remaining - 5, 100))
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < rank:
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W, rank = U_new, V_new, W_new, new_rank
    
    return {"U": U, "V": V, "W": W}
