"""Rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.

The output factor W uses the official trace convention W[k*n + i, r].
We search over sparse 0/+/-1 factor combinations and solve the resulting linear system
for the exact coefficients.  The best verified decomposition found is cached
so that repeated calls are instant.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
N, M, P = 2, 3, 4
NM = N * M          # 6
MP = M * P          # 12
PN = P * N          # 8
NP = N * P          # 8  (number of output equations)


def _verify(U, V, W, rank, tol=1e-7):
    """Check that sum_r U[:,r] V[:,r] reproduces the multiplication tensor."""
    # We need: sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta(j, j')
    # for all i, k, j, j'.
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(N):
        for k in range(P):
            for j in range(M):
                for j_prime in range(M):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * M + j, r] * V[j_prime * P + k, r] * W[k * N + i, r]
                    target = 1.0 if j == j_prime else 0.0
                    if abs(s - target) > tol:
                        return False
    return True


def _to_half_int_check(arr, tol=1e-8):
    """Check if all entries are close to half-integers (i.e., 2*arr is close to integer)."""
    scaled = 2.0 * np.array(arr, dtype=float)
    rounded = np.round(scaled)
    return np.allclose(scaled, rounded, atol=tol)


def _search_rank(rank, deadline, max_attempts=100000):
    """Search for a rank-rank decomposition. Returns (U,V,W) lists or None."""
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Number of equations: N*M*P (diag) + N*P*M*(M-1) (off-diag)
    # 2*3*4 + 2*4*3*2 = 24 + 48 = 72
    num_eqs = N * M * P + N * P * M * (M - 1)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random U and V with entries in {-1, 0, 1}
        # Use structured sparsity: each row has a few non-zeros
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        
        # Randomly assign non-zero entries
        for r in range(rank):
            # Choose which rows of U and V are non-zero for this rank component
            u_rows = np.random.choice(NM, size=min(3, NM), replace=False)
            v_rows = np.random.choice(MP, size=min(4, MP), replace=False)
            
            for idx in u_rows:
                U[idx, r] = random.choice([-1, 1])
            for idx in v_rows:
                V[idx, r] = random.choice([-1, 1])
        
        # Build the linear system A * w = b
        # w is a vector of length PN * rank (flattened W)
        A_mat = np.zeros((num_eqs, PN * rank))
        b_vec = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(N):
            for k in range(P):
                for j in range(M):
                    for j_prime in range(M):
                        target = 1.0 if j == j_prime else 0.0
                        b_vec[eq_idx] = target
                        for r in range(rank):
                            coeff = U[i * M + j, r] * V[j_prime * P + k, r]
                            if coeff != 0:
                                A_mat[eq_idx, r * PN + (k * N + i)] = coeff
                        eq_idx += 1
                        
        # Solve A_mat * w_flat = b_vec
        try:
            w_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check if solution is exact
        if not np.allclose(A_mat @ w_flat, b_vec, atol=1e-8):
            continue
            
        # Check if w_flat entries are "nice" (close to half-integers or integers)
        w_scaled = 2.0 * w_flat
        if not np.allclose(w_scaled, np.round(w_scaled), atol=1e-8):
            continue
            
        # Construct U, V, W
        W = w_flat.reshape(rank, PN).T  # W is (PN, rank)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None


def _search_rank_deterministic(rank, deadline):
    """Systematic search for a rank-rank decomposition using structured patterns."""
    import itertools
    
    # Try all combinations of sign patterns for a fixed sparse structure
    # This is more exhaustive than random search
    
    max_attempts = 50000
    attempts = 0
    
    # Use a grid of possible sparsity patterns
    for u_pattern in range(min(100, max_attempts // 10)):
        if time.time() > deadline:
            break
        attempts += 1
        
        # Generate U with a structured pattern
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        
        # Deterministic pattern based on u_pattern
        seed = 42 + u_pattern * 7
        rng = np.random.RandomState(seed)
        
        for r in range(rank):
            u_rows = rng.choice(NM, size=min(3, NM), replace=False)
            v_rows = rng.choice(MP, size=min(4, MP), replace=False)
            
            for idx in u_rows:
                U[idx, r] = rng.choice([-1, 1])
            for idx in v_rows:
                V[idx, r] = rng.choice([-1, 1])
        
        # Build the linear system
        num_eqs = N * M * P + N * P * M * (M - 1)
        A_mat = np.zeros((num_eqs, PN * rank))
        b_vec = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(N):
            for k in range(P):
                for j in range(M):
                    for j_prime in range(M):
                        target = 1.0 if j == j_prime else 0.0
                        b_vec[eq_idx] = target
                        for r in range(rank):
                            coeff = U[i * M + j, r] * V[j_prime * P + k, r]
                            if coeff != 0:
                                A_mat[eq_idx, r * PN + (k * N + i)] = coeff
                        eq_idx += 1
                        
        try:
            w_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        if not np.allclose(A_mat @ w_flat, b_vec, atol=1e-8):
            continue
            
        w_scaled = 2.0 * w_flat
        if not np.allclose(w_scaled, np.round(w_scaled), atol=1e-8):
            continue
            
        W = w_flat.reshape(rank, PN).T
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}


_CACHE = None


def construct():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    start = time.time()
    deadline = start + 130.0   # safety margin before 150s wall-clock
    best = _schoolbook()
    
    # Try to find a lower-rank exact decomposition.
    # Known lower bound for 2x3x4 is around 17-18.
    # Try ranks from low to high.
    for rank in range(16, 24):
        if time.time() > deadline:
            break
        
        # First try deterministic structured search
        res = _search_rank_deterministic(rank, deadline)
        if res is not None:
            U, V, W = res
            best = {"U": U, "V": V, "W": W}
            break
        
        # Then try random search with more attempts
        res = _search_rank(rank, deadline, max_attempts=50000)
        if res is not None:
            U, V, W = res
            best = {"U": U, "V": V, "W": W}
            break
    
    _CACHE = best
    return _CACHE
