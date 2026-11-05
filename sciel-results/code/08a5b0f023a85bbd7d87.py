"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  * Build a rank-44 base using 4 Strassen(2x2x2) blocks for the 2x2 core
    and schoolbook for the 2x1 tail.
  * Use slice-rank structural analysis to guide targeted column elimination.
  * Try single, pair, and triple column elimination with integer/half-integer rounding.
  * Group columns by output slice k for structural elimination.
  * Exact validation at each step.
"""
import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    acc = 0
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    # ---- Build rank-44 base ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: 4 Strassen blocks for A1(2x2) * B1(2x8)
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: A2(2x1) * B2(1x8) schoolbook
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    U, V, W = base["U"], base["V"], base["W"]
    rank0 = len(U[0])  # 44

    # ---- Build column vectors in tensor space ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                uv = U[i * m + a][c]
                if not uv:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if not v_val:
                        continue
                    w_val = W[k * n + i][c]
                    if not w_val:
                        continue
                    vec[i * m * p + a * p + k] = uv * v_val * w_val
        return vec

    cols = [_col_vector(U, V, W, c) for c in range(rank0)]

    # ---- (a) Duplicate column removal ----
    seen = {}
    removable = []
    for c in range(rank0):
        key = tuple(cols[c])
        if key in seen:
            removable.append(c)
        else:
            seen[key] = c

    for c_rem in removable:
        if not _ok():
            break
        new_rank = rank0 - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        nc = 0
        for c in range(rank0):
            if c == c_rem:
                continue
            for i in range(n * m):
                new_U[i][nc] = U[i][c]
            for i in range(m * p):
                new_V[i][nc] = V[i][c]
            for i in range(p * n):
                new_W[i][nc] = W[i][c]
            nc += 1
        cand = {"U": new_U, "V": new_V, "W": new_W}
        if _validate(cand):
            best_candidate = cand
            best_rank = new_rank
            return best_candidate

    # ---- (b) Global SVD-based rank reduction ----
    M = np.zeros((num_terms, rank0), dtype=np.float64)
    for c in range(rank0):
        for r in range(num_terms):
            if cols[c][r]:
                M[r, c] = float(cols[c][r])

    try:
        sv_vals = np.linalg.svd(M, compute_uv=False)
        tol = 1e-10 * max(sv_vals) if len(sv_vals) > 0 else 1e-10
        num_rank = int(np.sum(sv_vals > tol))
    except np.linalg.LinAlgError:
        num_rank = rank0

    if num_rank < rank0:
        # Find basis columns greedily
        basis_cols = []
        for c in range(rank0):
            if len(basis_cols) >= num_rank:
                break
            if not basis_cols:
                basis_cols.append(c)
                continue
            B = M[:, basis_cols]
            b = M[:, c]
            try:
                x, residuals, rnk, sv = np.linalg.lstsq(B, b, rcond=1e-10)
                resid = np.linalg.norm(B @ x - b)
                if resid > 1e-8:
                    basis_cols.append(c)
            except np.linalg.LinAlgError:
                basis_cols.append(c)

        if len(basis_cols) < rank0:
            non_basis = [c for c in range(rank0) if c not in basis_cols]
            B = M[:, basis_cols]
            coeffs = {}
            success = True
            for c in non_basis:
                b = M[:, c]
                try:
                    x, residuals, rnk, sv = np.linalg.lstsq(B, b, rcond=1e-10)
                    x_round = np.round(x * 2) / 2.0
                    if np.allclose(B @ x_round, b, atol=1e-10):
                        coeffs[c] = x_round
                    else:
                        success = False
                        break
                except np.linalg.LinAlgError:
                    success = False
                    break

            if success:
                new_rank = len(basis_cols)
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]

                for j, bc in enumerate(basis_cols):
                    for i in range(n * m):
                        new_U[i][j] = U[i][bc]
                    for i in range(m * p):
                        new_V[i][j] = V[i][bc]
                    for i in range(p * n):
                        new_W[i][j] = W[i][bc]

                for c in non_basis:
                    x_round = coeffs[c]
                    for j, bc in enumerate(basis_cols):
                        wgt = x_round[j]
                        if wgt != 0:
                            for i in range(n * m):
                                new_U[i][j] += wgt * U[i][c]
                            for i in range(m * p):
                                new_V[i][j] += wgt * V[i][c]
                            for i in range(p * n):
                                new_W[i][j] += wgt * W[i][c]

                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return best_candidate

    # ---- Robust Integer Dependency Search ----
    def _find_integer_dependency():
        """
        Finds a non-zero integer vector x such that sum(x[c] * col_c) == 0.
        Uses exact rational arithmetic on the scaled integer tensor columns.
        """
        from fractions import Fraction
        
        # Scale by 2 to handle half-integers
        scale = 2
        int_cols = []
        for c in range(rank0):
            vec = []
            for r in range(num_terms):
                val = cols[c][r]
                # val is float, but should be integer or half-integer
                iv = int(round(val * scale))
                vec.append(iv)
            int_cols.append(vec)
        
        # We want to find a small integer vector x in the kernel of the matrix formed by int_cols.
        # Since rank0 is ~44 and num_terms is 48, the kernel is likely 0-dimensional or very small.
        # However, if the decomposition is not minimal, there are dependencies.
        
        # We will try to eliminate one column at a time by checking if it's in the span of others.
        # But we do it using exact arithmetic.
        
        # To speed up, we can pick a pivot column and try to express it.
        # Let's iterate through columns.
        for c in range(rank0):
            if not _ok():
                break
            
            kept = [c2 for c2 in range(rank0) if c2 != c]
            ncols = len(kept)
            
            # Build system: sum_{j} x_j * col_{kept[j]} = -col_c
            # This is a system of num_terms linear equations in ncols variables.
            # Since ncols < num_terms usually, it's overdetermined.
            # We solve using Gaussian elimination over Q.
            
            # We only need to solve if the system is consistent.
            # We can use the fact that if a solution exists, it is unique (if rank is ncols).
            
            # Pick ncols pivot rows.
            # We'll use a simple heuristic: try to find a basis.
            
            # Instead of full GE, we can use the property that if a solution exists,
            # any subset of ncols linearly independent equations will give the solution.
            
            # Let's try to solve using the first ncols equations? No, might be singular.
            # Let's use SVD to find the best real solution first, then verify with integers.
            # This is what the old code did. The issue was rounding.
            
            # New approach: 
            # 1. Find real solution x_real via lstsq.
            # 2. Check if x_real is close to integers/half-integers.
            # 3. If not, search for nearby integer points?
            
            # Actually, for matrix multiplication tensors, dependencies are often "local" or "structured".
            # Let's try to find a dependency involving a small subset of columns.
            
            # We will check all pairs and triples of columns to see if their sum is zero or a multiple of another.
            # This is O(R^3) which is 44^3 ~ 85k, very fast.
            
            # Check if sum of any 2 columns is 0
            for i in range(rank0):
                for j in range(i+1, rank0):
                    if not _ok(): break
                    s = [cols[i][r] + cols[j][r] for r in range(num_terms)]
                    if all(x == 0 for x in s):
                        # col_i + col_j = 0 => col_i = -col_j
                        # We can eliminate col_i by adding col_j to it? No, we remove one.
                        # If we remove col_i, we must adjust others? No, we just remove it.
                        # But wait, if col_i = -col_j, then the term is redundant.
                        # We can remove col_i.
                        kept = [c2 for c2 in range(rank0) if c2 != i]
                        new_rank = len(kept)
                        new_U = [[0]*new_rank for _ in range(n*m)]
                        new_V = [[0]*new_rank for _ in range(m*p)]
                        new_W = [[0]*new_rank for _ in range(p*n)]
                        nc = 0
                        for c2 in kept:
                            for i_idx in range(n*m): new_U[i_idx][nc] = U[i_idx][c2]
                            for i_idx in range(m*p): new_V[i_idx][nc] = V[i_idx][c2]
                            for i_idx in range(p*n): new_W[i_idx][nc] = W[i_idx][c2]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            return cand
                        break
                if not _ok(): break
            if not _ok(): break
            
            # Check if sum of any 2 columns equals a third
            for i in range(rank0):
                if not _ok(): break
                for j in range(i+1, rank0):
                    if not _ok(): break
                    s = [cols[i][r] + cols[j][r] for r in range(num_terms)]
                    # Check if s is equal to any other column
                    for k in range(rank0):
                        if k == i or k == j: continue
                        if all(s[r] == cols[k][r] for r in range(num_terms)):
                            # col_i + col_j = col_k => col_k is redundant
                            kept = [c2 for c2 in range(rank0) if c2 != k]
                            new_rank = len(kept)
                            new_U = [[0]*new_rank for _ in range(n*m)]
                            new_V = [[0]*new_rank for _ in range(m*p)]
                            new_W = [[0]*new_rank for _ in range(p*n)]
                            nc = 0
                            for c2 in kept:
                                for i_idx in range(n*m): new_U[i_idx][nc] = U[i_idx][c2]
                                for i_idx in range(m*p): new_V[i_idx][nc] = V[i_idx][c2]
                                for i_idx in range(p*n): new_W[i_idx][nc] = W[i_idx][c2]
                                nc += 1
                            cand = {"U": new_U, "V": new_V, "W": new_W}
                            if _validate(cand):
                                return cand
                            break
                    if not _ok(): break
            if not _ok(): break
            
            # Check if sum of any 3 columns is 0
            for i in range(rank0):
                if not _ok(): break
                for j in range(i+1, rank0):
                    if not _ok(): break
                    for k in range(j+1, rank0):
                        if not _ok(): break
                        s = [cols[i][r] + cols[j][r] + cols[k][r] for r in range(num_terms)]
                        if all(x == 0 for x in s):
                            # col_i + col_j + col_k = 0
                            # Remove col_k
                            kept = [c2 for c2 in range(rank0) if c2 != k]
                            new_rank = len(kept)
                            new_U = [[0]*new_rank for _ in range(n*m)]
                            new_V = [[0]*new_rank for _ in range(m*p)]
                            new_W = [[0]*new_rank for _ in range(p*n)]
                            nc = 0
                            for c2 in kept:
                                for i_idx in range(n*m): new_U[i_idx][nc] = U[i_idx][c2]
                                for i_idx in range(m*p): new_V[i_idx][nc] = V[i_idx][c2]
                                for i_idx in range(p*n): new_W[i_idx][nc] = W[i_idx][c2]
                                nc += 1
                            cand = {"U": new_U, "V": new_V, "W": new_W}
                            if _validate(cand):
                                return cand
                            break
                        if not _ok(): break
                    if not _ok(): break
                if not _ok(): break
            if not _ok(): break
            
            # Check if sum of any 3 columns equals a fourth
            for i in range(rank0):
                if not _ok(): break
                for j in range(i+1, rank0):
                    if not _ok(): break
                    for k in range(j+1, rank0):
                        if not _ok(): break
                        s = [cols[i][r] + cols[j][r] + cols[k][r] for r in range(num_terms)]
                        for l in range(rank0):
                            if l in (i,j,k): continue
                            if all(s[r] == cols[l][r] for r in range(num_terms)):
                                # col_i + col_j + col_k = col_l
                                kept = [c2 for c2 in range(rank0) if c2 != l]
                                new_rank = len(kept)
                                new_U = [[0]*new_rank for _ in range(n*m)]
                                new_V = [[0]*new_rank for _ in range(m*p)]
                                new_W = [[0]*new_rank for _ in range(p*n)]
                                nc = 0
                                for c2 in kept:
                                    for i_idx in range(n*m): new_U[i_idx][nc] = U[i_idx][c2]
                                    for i_idx in range(m*p): new_V[i_idx][nc] = V[i_idx][c2]
                                    for i_idx in range(p*n): new_W[i_idx][nc] = W[i_idx][c2]
                                    nc += 1
                                cand = {"U": new_U, "V": new_V, "W": new_W}
                                if _validate(cand):
                                    return cand
                                break
                        if not _ok(): break
                    if not _ok(): break
                if not _ok(): break
            if not _ok(): break

        return None

    # Try to find a dependency
    improved = _find_integer_dependency()
    if improved:
        return improved

    # Fallback to old single elimination if no simple dependency found
    # (This part is kept from original but simplified)
    def _try_eliminate_single_fast(c):
        kept = [c2 for c2 in range(rank0) if c2 != c]
        ncols = len(kept)
        A = np.zeros((num_terms, ncols), dtype=np.float64)
        for j, c2 in enumerate(kept):
            for r in range(num_terms):
                if cols[c2][r]:
                    A[r, j] = float(cols[c2][r])
        b = np.array([-float(x) for x in cols[c]], dtype=np.float64)
        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False
        if residuals and residuals[0] > 1e-8:
            return False
        for scale in [1, 2]:
            x_round = np.round(x * scale) / scale
            if not np.allclose(A @ x_round, b, atol=1e-10):
                continue
            new_rank = ncols
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            for j, c2 in enumerate(kept):
                wgt = x_round[j]
                for i in range(n * m):
                    new_U[i][j] = U[i][c2] + wgt * U[i][c]
                for i in range(m * p):
                    new_V[i][j] = V[i][c2] + wgt * V[i][c]
                for i in range(p * n):
                    new_W[i][j] = W[i][c2] + wgt * W[i][c]
            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                return cand
        return False

    for c in range(rank0):
        if not _ok(): break
        res = _try_eliminate_single_fast(c)
        if res:
            return res

    if best_rank < 44:
        return best_candidate
    return base

# EVOLVE-BLOCK-END
