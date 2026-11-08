import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]


def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[kp*n+ip,r] == delta_{i,ip}*delta_{k,kp}"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        expected = 1 if (i == ip and k == kp) else 0
                        actual = 0
                        for r in range(rank):
                            actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        if actual != expected:
                            return False
    return True


def _build_coefficient_matrix(U, V, W, n=2, m=4, p=8):
    """Build the coefficient matrix A where A[eq, r] = U[i*m+j,r]*V[j*p+k,r]*W[kp*n+ip,r]"""
    R = len(U[0])
    num_eq = n * m * p * n * p
    A = np.zeros((num_eq, R), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for r in range(R):
                            A[eq_idx, r] = U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        eq_idx += 1
    return A


def _find_null_vector_exact(U, V, W, n=2, m=4, p=8):
    """Find a non-trivial null space vector using exact rational arithmetic via Gaussian elimination."""
    R = len(U[0])
    num_eq = n * m * p * n * p
    
    # Build coefficient matrix with Fractions
    A = [[Fraction(0) for _ in range(R)] for _ in range(num_eq)]
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for r in range(R):
                            A[eq_idx][r] = Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[kp * n + ip][r])
                        eq_idx += 1
    
    # Gaussian elimination to find null space
    M = [row[:] for row in A]
    rank = 0
    pivots = []
    
    for col in range(R):
        pivot_row = None
        for row in range(rank, num_eq):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        if pivot_row != rank:
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
        # Normalize pivot row
        pivot_val = M[rank][col]
        for c in range(col, R):
            M[rank][c] = M[rank][c] / pivot_val
        pivots.append(col)
        # Eliminate
        for row in range(num_eq):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for c in range(col, R):
                    M[row][c] = M[row][c] - factor * M[rank][c]
        rank += 1
        if rank == R:
            break
    
    # Find free variables
    pivot_cols = set(pivots)
    free_cols = [c for c in range(R) if c not in pivot_cols]
    
    if not free_cols:
        return None  # Full rank, no null space
    
    # For each free variable, find a null space vector
    for free_col in free_cols:
        c = [Fraction(0) for _ in range(R)]
        c[free_col] = Fraction(1)
        for pivot_col in pivots:
            row_idx = pivots.index(pivot_col)
            c[pivot_col] = -M[row_idx][free_col]
        
        # Verify this is actually in the null space
        valid = True
        for eq in range(num_eq):
            dot = Fraction(0)
            for r in range(R):
                dot += A[eq][r] * c[r]
            if dot != 0:
                valid = False
                break
        
        if valid:
            return c
    
    return None


def _try_reduce_rank(U, V, W, n=2, m=4, p=8, deadline=None):
    """Try to find a rank reduction by 1 from the given decomposition."""
    if deadline is not None and time.time() > deadline:
        return None
    
    R = len(U[0])
    
    # First quick check with numpy
    A = _build_coefficient_matrix(U, V, W, n, m, p)
    rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    
    if rank_A >= R:
        return None  # No reduction possible
    
    # Exact verification
    c = _find_null_vector_exact(U, V, W, n, m, p)
    if c is None:
        return None
    
    # Find the index of the largest magnitude coefficient
    max_idx = max(range(R), key=lambda i: abs(c[i]))
    if c[max_idx] == 0:
        return None
    
    # Construct new (R-1) rank decomposition
    new_rank = R - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    # Map old indices to new indices (skipping max_idx)
    new_idx_map = {}
    new_i = 0
    for old_i in range(R):
        if old_i != max_idx:
            new_idx_map[old_i] = new_i
            new_i += 1
    
    # For each remaining term, scale by (1 - c[old_r]/c[max_idx])
    for old_r in range(R):
        if old_r == max_idx:
            continue
        new_r = new_idx_map[old_r]
        factor = 1 - c[old_r] / c[max_idx]
        
        for row in range(n * m):
            new_U[row][new_r] = factor * U[row][old_r]
        for row in range(m * p):
            new_V[row][new_r] = factor * V[row][old_r]
        for row in range(p * n):
            new_W[row][new_r] = factor * W[row][old_r]
    
    # Verify the new decomposition exactly
    if not _verify_decomposition(new_U, new_V, new_W, n, m, p):
        return None
    
    # Check if entries are integer or half-integer
    def is_half_int(x):
        return abs(x * 2 - round(x * 2)) < 1e-12
    
    all_half_int = True
    for row in new_U:
        for val in row:
            if not is_half_int(val):
                all_half_int = False
                break
        if not all_half_int:
            break
    if all_half_int:
        for row in new_V:
            for val in row:
                if not is_half_int(val):
                    all_half_int = False
                    break
            if not all_half_int:
                break
    if all_half_int:
        for row in new_W:
            for val in row:
                if not is_half_int(val):
                    all_half_int = False
                    break
            if not all_half_int:
                break
    
    if not all_half_int:
        return None
    
    # Round to half-integers
    def round_half(x):
        return round(x * 2) / 2.0
    
    for row in new_U:
        for idx in range(len(row)):
            row[idx] = round_half(row[idx])
    for row in new_V:
        for idx in range(len(row)):
            row[idx] = round_half(row[idx])
    for row in new_W:
        for idx in range(len(row)):
            row[idx] = round_half(row[idx])
    
    return {"U": new_U, "V": new_V, "W": new_W}


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Start with the known rank-56 solution
    base = _build_rank56()
    
    # Try to reduce the rank
    if time.time() < deadline:
        reduced = _try_reduce_rank(base["U"], base["V"], base["W"], deadline=deadline)
        if reduced is not None:
            return reduced
    
    # Fallback to rank-56
    return base
