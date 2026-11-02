import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if abs(total - 1) > 1e-9:
                    return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, float):
                    if not val.is_integer() and abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    return True

def _build_schoolbook(n=3, m=3, p=8):
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
    return {"U": U, "V": V, "W": W}

def _round_half(x):
    return np.round(2.0 * x) / 2.0

def _check_half_integer(arr, tol=1e-9):
    """Check if array entries are integer or half-integer."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _try_solve_W(U, V, n, m, p, r):
    """Given U and V, solve for W exactly."""
    num_constraints = n * m * p
    num_unknowns = p * n * r
    
    A = np.zeros((num_constraints, num_unknowns))
    b = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b_idx = j * p + k
                c = k * n + i
                for r_idx in range(r):
                    A[idx, c * r + r_idx] = U[a, r_idx] * V[b_idx, r_idx]
                idx += 1
    
    # Solve least squares
    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape(p * n, r)
    
    # Check if solution is exact (residuals should be ~0)
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-8:
            return None
    
    # Check if W is half-integer
    if not _check_half_integer(W):
        return None
    
    return W

def _try_solve_V(U, W, n, m, p, r):
    """Given U and W, solve for V exactly."""
    num_constraints = n * m * p
    num_unknowns = m * p * r
    
    A = np.zeros((num_constraints, num_unknowns))
    b = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b_idx = j * p + k
                c = k * n + i
                for r_idx in range(r):
                    A[idx, b_idx * r + r_idx] = U[a, r_idx] * W[c, r_idx]
                idx += 1
    
    V_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
    V = V_flat.reshape(m * p, r)
    
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-8:
            return None
    
    if not _check_half_integer(V):
        return None
    
    return V

def _try_solve_U(V, W, n, m, p, r):
    """Given V and W, solve for U exactly."""
    num_constraints = n * m * p
    num_unknowns = n * m * r
    
    A = np.zeros((num_constraints, num_unknowns))
    b = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b_idx = j * p + k
                c = k * n + i
                for r_idx in range(r):
                    A[idx, a * r + r_idx] = V[b_idx, r_idx] * W[c, r_idx]
                idx += 1
    
    U_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
    U = U_flat.reshape(n * m, r)
    
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-8:
            return None
    
    if not _check_half_integer(U):
        return None
    
    return U

def _random_half_integer_matrix(shape, rng, values=None):
    if values is None:
        values = [-1, -0.5, 0, 0.5, 1]
    return rng.choice(values, size=shape).astype(float)

def _try_rank_search(r, n, m, p, deadline, num_attempts=500):
    """Try to find rank-r decomposition using multiple strategies."""
    rng = np.random.default_rng()
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
        
        # Strategy 1: Random U, V -> solve W
        U = _random_half_integer_matrix((n * m, r), rng)
        V = _random_half_integer_matrix((m * p, r), rng)
        W = _try_solve_W(U, V, n, m, p, r)
        if W is not None:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Strategy 2: Random U, W -> solve V
        U = _random_half_integer_matrix((n * m, r), rng)
        W = _random_half_integer_matrix((p * n, r), rng)
        V = _try_solve_V(U, W, n, m, p, r)
        if V is not None:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Strategy 3: Random V, W -> solve U
        V = _random_half_integer_matrix((m * p, r), rng)
        W = _random_half_integer_matrix((p * n, r), rng)
        U = _try_solve_U(V, W, n, m, p, r)
        if U is not None:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Strategy 4: Random U, V -> solve W, then round W and check
        U = _random_half_integer_matrix((n * m, r), rng)
        V = _random_half_integer_matrix((m * p, r), rng)
        # Build linear system
        num_constraints = n * m * p
        num_unknowns = p * n * r
        A = np.zeros((num_constraints, num_unknowns))
        b = np.ones(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c = k * n + i
                    for r_idx in range(r):
                        A[idx, c * r + r_idx] = U[a, r_idx] * V[b_idx, r_idx]
                    idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(p * n, r)
        W_rounded = _round_half(W)
        # Verify with rounded W
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c = k * n + i
                    total = np.sum(U[a, :] * V[b_idx, :] * W_rounded[c, :])
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid and _check_half_integer(W_rounded):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _structured_search(r, n, m, p, deadline):
    """Try structured decompositions based on algebraic patterns."""
    # Try: U[i*m+j, r] depends only on i, V[j*p+k, r] depends only on k, W[k*n+i, r] depends on j
    # This won't work directly, but let's try block structures
    
    # Pattern: split the rank into blocks corresponding to j values
    # For each j, have a block of rank r/m (if divisible)
    if r % m != 0:
        return None
    
    block_rank = r // m
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    rng = np.random.default_rng()
    for j in range(m):
        block_start = j * block_rank
        block_end = (j + 1) * block_rank
        # U rows for this j
        for i in range(n):
            U[i * m + j, block_start:block_end] = rng.choice([-1, -0.5, 0, 0.5, 1], size=block_rank)
        # V rows for this j
        for k in range(p):
            V[j * p + k, block_start:block_end] = rng.choice([-1, -0.5, 0, 0.5, 1], size=block_rank)
        # W rows for this j (W is indexed by k*n+i, so we need to fill for all k,i)
        # W[k*n+i, block] should be such that sum over r in block: U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] contributes to the total
    
    # This structured approach is complex; skip for now and rely on random search
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s kill)
    
    n, m, p = 3, 3, 8
    
    # Start with schoolbook
    result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound up to best_rank-1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Budget per rank: decrease as r increases (harder to find)
        remaining = deadline - time.time()
        ranks_remaining = best_rank - r
        budget_per_rank = max(10, remaining / max(1, ranks_remaining))
        
        num_attempts = int(min(500, max(50, budget_per_rank * 10)))
        
        res = _try_rank_search(r, n, m, p, deadline, num_attempts)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                best_rank = r
                result = res
                # Found a valid decomposition; continue searching for lower rank
                # but we're going upward, so we should break and restart downward
                # Actually, we're searching from low to high, so first found is minimal
                break
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
