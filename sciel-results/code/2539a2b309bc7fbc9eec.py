"""Matrix multiplication 4x5 by 5x6 — Rank-minimal via linear-algebra peeling."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _matmul_frac(A, B):
    """Exact rational matrix multiplication."""
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    C = [[Fraction(0) for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            if A[i][k] == 0:
                continue
            for j in range(cols_b):
                C[i][j] += A[i][k] * B[k][j]
    return C

def _matvec_frac(A, v):
    """Exact rational matrix-vector multiplication."""
    rows_a, cols_a = len(A), len(A[0])
    C = [Fraction(0) for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_a):
            if A[i][j] != 0:
                C[i] += A[i][j] * v[j]
    return C

def _rref_exact(M):
    """
    Compute rank-revealing form using exact Gaussian elimination with Fraction.
    Returns (rank, pivot_cols, R) where R is the reduced matrix.
    """
    if len(M) == 0:
        return 0, [], []
    rows = len(M)
    cols = len(M[0])
    R = [row[:] for row in M]
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # Find pivot
        pivot_row = -1
        for i in range(r, rows):
            if R[i][c] != 0:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
        # Swap
        R[r], R[pivot_row] = R[pivot_row], R[r]
        # Normalize
        pv = R[r][c]
        for j in range(c, cols):
            R[r][j] /= pv
        # Eliminate
        for i in range(rows):
            if i != r and R[i][c] != 0:
                factor = R[i][c]
                for j in range(c, cols):
                    R[i][j] -= factor * R[r][j]
        pivot_cols.append(c)
        r += 1
    return r, pivot_cols, R

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _peel_rank(U, V, W, n, m, p, deadline):
    """
    Iteratively peel off linearly dependent rank-1 terms.
    
    The multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    We view each rank-1 term r as a vector in a combined space. Two terms r1, r2 are 
    linearly dependent (in the sense that one can be removed by adjusting the other) 
    if their "outer product vectors" are linearly dependent.
    
    More precisely: for each rank index r, define the vector:
      vec_r = [U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k)]
    
    If vec_r is in the span of {vec_s : s != r}, then term r is redundant.
    
    We use a greedy approach: build a matrix where each column is vec_r, compute its
    rank, and identify which columns are linearly dependent.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Build the tensor matrix: columns are rank-1 vectors
    # For each r, vec_r has n*m*p entries
    total_entries = n * m * p
    # M_tensor[t, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] where t = ((i*m+j)*p + k)
    # Actually let's index t = i * (m*p) + j * p + k
    
    M_tensor = [[Fraction(0) for _ in range(rank)] for _ in range(total_entries)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * (m * p) + j * p + k
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(rank):
                    M_tensor[t][r] = Fraction(U[u_row][r]) * Fraction(V[v_row][r]) * Fraction(W[w_row][r])
    
    # Now find which columns are linearly dependent
    # Compute RREF of M_tensor (transposed: we want column rank)
    # Actually, we want to find a subset of columns that spans the column space.
    # Transpose: rows are rank indices, columns are tensor entries
    M_T = [[M_tensor[t][r] for t in range(total_entries)] for r in range(rank)]
    
    # RREF of M_T gives us which rows (original columns) are pivots
    # But we want to keep a minimal set of original columns
    # Let's do RREF on M_tensor directly (rows = tensor entries, cols = rank)
    # The pivot columns in RREF of M_tensor tell us which rank indices are independent
    
    r, pivot_cols, R = _rref_exact(M_tensor)
    
    # pivot_cols are the independent rank indices
    # We keep only those
    if time.time() > deadline - 30:
        return U, V, W  # No time to rebuild
    
    # Rebuild U, V, W with only the pivot columns
    U_new = []
    for i in range(n * m):
        U_new.append([U[i][c] for c in pivot_cols])
    V_new = []
    for j in range(m * p):
        V_new.append([V[j][c] for c in pivot_cols])
    W_new = []
    for k in range(p * n):
        W_new.append([W[k][c] for c in pivot_cols])
    
    return U_new, V_new, W_new

def _greedy_min_rank(U, V, W, n, m, p, deadline):
    """
    Greedy rank minimization: try to remove each rank-1 term one at a time,
    checking if the remaining terms still span the tensor.
    
    This is O(rank * total_entries * rank) which for rank=120, total_entries=120
    is about 120*120*120 = 1.7M fraction ops. Might be slow.
    
    Alternative: use the linear algebra approach directly.
    """
    start_time = time.time()
    rank = len(U[0])
    total_entries = n * m * p
    
    # Build tensor matrix
    M_tensor = [[Fraction(0) for _ in range(rank)] for _ in range(total_entries)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * (m * p) + j * p + k
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(rank):
                    M_tensor[t][r] = Fraction(U[u_row][r]) * Fraction(V[v_row][r]) * Fraction(W[w_row][r])
    
    # Find minimal spanning set of columns
    r, pivot_cols, R = _rref_exact(M_tensor)
    
    if time.time() > deadline - 30:
        return U, V, W
    
    U_new = []
    for i in range(n * m):
        U_new.append([U[i][c] for c in pivot_cols])
    V_new = []
    for j in range(m * p):
        V_new.append([V[j][c] for c in pivot_cols])
    W_new = []
    for k in range(p * n):
        W_new.append([W[k][c] for c in pivot_cols])
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Start with schoolbook decomposition (rank = 4*5*6 = 120)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Use exact linear algebra to find the minimal rank
    # The tensor T[i,j,k] has a specific structure: T[i,j,k] = 1 if the multiplication
    # map sends e_j (input) to e_i (output) via index k, else 0.
    # Actually T[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
    # T[i,j,k] = 1 for all (i,j,k) in the schoolbook, meaning each (i,j,k) triple
    # contributes exactly one rank-1 term.
    
    # The key insight: the multiplication tensor for n x m * m x p has rank at most
    # n*m*p, but may have lower rank due to linear dependencies.
    
    # Let's compute the exact rank by finding the minimal number of rank-1 tensors
    # that sum to the multiplication tensor.
    
    # Build the full tensor as a matrix (tensor unfolding)
    # M[i*(m*p) + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The rank of M gives the minimal number of rank-1 terms needed.
    
    if time.time() < deadline - 60:
        U, V, W = _greedy_min_rank(U, V, W, n, m, p, deadline)
    
    # Verify
    if time.time() < deadline - 10:
        if not _verify_decomp(U, V, W, n, m, p):
            # Fallback
            U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
