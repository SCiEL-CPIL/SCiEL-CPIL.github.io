"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _solve_for_W(U, V, n, m, p):
    """Given U (nm x r) and V (mp x r), solve for W (pn x r).
    
    The equation is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
    Total variables: p * n * r
    Equations: n * m * p
    """
    r = U.shape[1]
    num_vars = p * n * r
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    col = (k * n + i) * r + c
                    A[eq_idx, col] = U[i * m + j, c] * V[j * p + k, c]
                eq_idx += 1
    
    # Solve the linear system
    try:
        W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        # Check if exact solution
        if np.allclose(A @ W_flat, b, atol=1e-8):
            W = W_flat.reshape((p * n, r))
            # Check if entries are nice (integer or half-integer)
            W_rounded = np.round(W * 2) / 2
            if np.allclose(W_rounded, W, atol=1e-8):
                return W_rounded
    except:
        pass
    
    return None


def _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout=60):
    """Search for rank-minimal decomposition of n x m x p tensor."""
    import time
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return None
        
        # Systematic enumeration: try different patterns for U and V
        # For small ranks, we can be more structured
        
        # Strategy: Use structured matrices based on the dimensions
        # For rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
        
        # Try random search with verification
        num_trials = 3000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Generate random U and V with small entries
            U_rand = np.zeros((nm, r))
            V_rand = np.zeros((mp, r))
            
            for i in range(nm):
                for c in range(r):
                    U_rand[i, c] = np.random.choice(coeff_set)
            
            for i in range(mp):
                for c in range(r):
                    V_rand[i, c] = np.random.choice(coeff_set)
            
            # Solve for W
            W = _solve_for_W(U_rand, V_rand, n, m, p)
            if W is not None:
                # Verify the full decomposition
                if _check_decomposition(U_rand, V_rand, W, n, m, p):
                    return (r, U_rand.tolist(), V_rand.tolist(), W.tolist())
        
        # Also try structured approaches
        # For specific ranks, try known patterns
        
        if r <= 8:
            # Try rank-r with all-ones patterns and variations
            for pattern in range(4):
                U_struct = np.zeros((nm, r))
                V_struct = np.zeros((mp, r))
                
                if pattern == 0:
                    # All ones in first column
                    U_struct[:, 0] = 1
                    V_struct[:, 0] = 1
                elif pattern == 1:
                    # Block diagonal
                    for c in range(r):
                        if c < n:
                            U_struct[c * m:(c+1) * m, c] = 1
                        if c < m:
                            V_struct[c * p:(c+1) * p, c] = 1
                elif pattern == 2:
                    # Alternating signs
                    for i in range(nm):
                        for c in range(r):
                            U_struct[i, c] = 1 if (i + c) % 2 == 0 else -1
                    for i in range(mp):
                        for c in range(r):
                            V_struct[i, c] = 1 if (i + c) % 2 == 0 else -1
                else:
                    # Random structured
                    U_struct = np.random.choice(coeff_set, size=(nm, r))
                    V_struct = np.random.choice(coeff_set, size=(mp, r))
                
                W = _solve_for_W(U_struct, V_struct, n, m, p)
                if W is not None:
                    if _check_decomposition(U_struct, V_struct, W, n, m, p):
                        return (r, U_struct.tolist(), V_struct.tolist(), W.tolist())
    
    return None


def _construct_sub_3x4x3():
    """Construct optimal decomposition for 3x4x3."""
    n, m, p = 3, 4, 3
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_sub_decomposition(n, m, p, max_rank=18, coeff_set=coeff_set, timeout=35)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def _construct_sub_3x4x2():
    """Construct optimal decomposition for 3x4x2."""
    n, m, p = 3, 4, 2
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_sub_decomposition(n, m, p, max_rank=14, coeff_set=coeff_set, timeout=35)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    # Find sub-decompositions
    sub_decomps = []
    # Block 0: k in [0, 3) -> 3x4x3
    r0, U0, V0, W0 = _construct_sub_3x4x3()
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], blocks[0]))
    # Block 1: k in [3, 5) -> 3x4x2
    r1, U1, V1, W1 = _construct_sub_3x4x2()
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], blocks[1]))
    # Block 2: k in [5, 7) -> 3x4x2
    r2, U2, V2, W2 = _construct_sub_3x4x2()
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], blocks[2]))

    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)

    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        # U is shared across blocks (same input indices)
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        # V: for block starting at k_off with size b
        for j in range(m):
            for kk in range(b):
                k_local = kk
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + k_local][c]

        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
