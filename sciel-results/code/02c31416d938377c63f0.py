"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
N = n * m  # 12
M = m * p  # 20
P = p * n  # 15

def build_tensor():
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def get_nonzero_indices():
    """Get all non-zero (a,b,c) indices of the tensor."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def get_zero_indices(sample_size=200, seed=42):
    """Get a sample of zero (a,b,c) indices."""
    rng = np.random.RandomState(seed)
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_set.add((a, b, c))
    
    zeros = []
    attempts = 0
    while len(zeros) < sample_size and attempts < sample_size * 10:
        a = rng.randint(0, N)
        b = rng.randint(0, M)
        c = rng.randint(0, P)
        if (a, b, c) not in nonzero_set:
            zeros.append((a, b, c))
        attempts += 1
    return zeros

def verify_decomposition(U, V, W, T=None, nonzero_indices=None, zero_indices=None):
    """Verify that U, V, W exactly decompose the tensor."""
    if T is None:
        T = build_tensor()
    if nonzero_indices is None:
        nonzero_indices = get_nonzero_indices()
    if zero_indices is None:
        zero_indices = get_zero_indices(200)
    
    rank = U.shape[1]
    
    # Check all non-zero entries exactly
    for (a, b, c) in nonzero_indices:
        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(val - 1.0) > 1e-10:
            return False
    
    # Check sampled zero entries
    for (a, b, c) in zero_indices:
        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
        if abs(val) > 1e-10:
            return False
    
    # Check half-integer constraint
    for mat in [U, V, W]:
        for val in mat:
            if not (abs(val * 2 - round(val * 2)) < 1e-10):
                return False
    
    return True

def schoolbook_decomposition():
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((N, rank), dtype=np.float64)
    V = np.zeros((M, rank), dtype=np.float64)
    W = np.zeros((P, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def snap_to_halfint(X, tol=1e-8):
    """Snap values to nearest half-integer if within tolerance."""
    Y = np.round(X * 2) / 2
    if np.allclose(X, Y, atol=tol):
        return Y
    return None

def structured_als(U, V, W, T, nonzero_indices, zero_indices, max_iters=50, tol=1e-14, timeout=30.0, start_time=None):
    """
    Structured ALS that exploits the tensor's sparsity.
    Only updates based on non-zero constraints plus sampled zero constraints.
    """
    if start_time is None:
        start_time = time.time()
    
    r = U.shape[1]
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    all_constraints = [(a, b, c, 1.0) for (a, b, c) in nonzero_indices] + \
                      [(a, b, c, 0.0) for (a, b, c) in zero_indices]
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Update U: for each row a, solve least squares
        for a in range(N):
            # Collect all constraints involving row a
            A_mat = np.zeros((len(all_constraints), r * r))
            target = np.zeros(len(all_constraints))
            valid = []
            for idx, (ca, cb, cc, cv) in enumerate(all_constraints):
                if ca == a:
                    # U[a,:] * V[cb,:] * W[cc,:] should equal cv
                    # This is bilinear in V and W given U
                    # For updating U[a,:], treat V[cb,:] and W[cc,:] as fixed
                    vw = np.outer(V[cb, :], W[cc, :])  # r x r
                    # U[a,:] @ vw = cv is not linear in U[a,:]
                    # Actually: sum_r U[a,r] * V[cb,r] * W[cc,r] = cv
                    # This is linear in U[a,:] with coefficient V[cb,r]*W[cc,r]
                    coeff = V[cb, :] * W[cc, :]
                    valid.append((idx, coeff, cv))
            
            if not valid:
                continue
            
            # Build system: sum over constraints of U[a,:] * coeff = cv
            # This is: sum_k U[a,k] * sum_j coeff[j,k] = sum_j cv[j]
            # Wait, that's not right. Each constraint gives one equation:
            # sum_k U[a,k] * coeff_k = cv
            # So we have multiple equations in r unknowns
            
            A_sys = np.zeros((len(valid), r))
            b_sys = np.zeros(len(valid))
            for vi, (idx, coeff, cv) in enumerate(valid):
                A_sys[vi, :] = coeff
                b_sys[vi] = cv
            
            if len(valid) >= r:
                try:
                    U[a, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                except:
                    pass
        
        # Update V similarly
        for b in range(M):
            valid = []
            for idx, (ca, cb, cc, cv) in enumerate(all_constraints):
                if cb == b:
                    coeff = U[ca, :] * W[cc, :]
                    valid.append((idx, coeff, cv))
            
            if not valid:
                continue
            
            A_sys = np.zeros((len(valid), r))
            b_sys = np.zeros(len(valid))
            for vi, (idx, coeff, cv) in enumerate(valid):
                A_sys[vi, :] = coeff
                b_sys[vi] = cv
            
            if len(valid) >= r:
                try:
                    V[b, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                except:
                    pass
        
        # Update W similarly
        for c in range(P):
            valid = []
            for idx, (ca, cb, cc, cv) in enumerate(all_constraints):
                if cc == c:
                    coeff = U[ca, :] * V[cb, :]
                    valid.append((idx, coeff, cv))
            
            if not valid:
                continue
            
            A_sys = np.zeros((len(valid), r))
            b_sys = np.zeros(len(valid))
            for vi, (idx, coeff, cv) in enumerate(valid):
                A_sys[vi, :] = coeff
                b_sys[vi] = cv
            
            if len(valid) >= r:
                try:
                    W[c, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                except:
                    pass
        
        # Compute error on all constraints
        error = 0.0
        for (ca, cb, cc, cv) in all_constraints:
            val = np.dot(np.dot(U[ca, :], V[cb, :]), W[cc, :])
            error += (val - cv) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if error < tol:
                break
    
    if best_error < 1e-10:
        sU = snap_to_halfint(best_U, tol=1e-8)
        sV = snap_to_halfint(best_V, tol=1e-8)
        sW = snap_to_halfint(best_W, tol=1e-8)
        if sU is not None and sV is not None and sW is not None:
            if verify_decomposition(sU, sV, sW, T, nonzero_indices, zero_indices):
                return sU, sV, sW
    
    return None

def try_rank_structured(r, T, nonzero_indices, zero_indices, max_iters=80, timeout=25.0, seed=42):
    """
    Try to find a rank-r decomposition using structured ALS with multiple initializations.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_error = np.inf
    best_result = None
    
    # Multiple diverse initializations exploiting tensor structure
    init_candidates = []
    
    # Init 0: Gaussian
    init_candidates.append(('gauss', rng.randn(N, r) * 0.5, rng.randn(M, r) * 0.5, rng.randn(P, r) * 0.5))
    
    # Init 1: Sparse random with structure
    U_s = np.zeros((N, r))
    V_s = np.zeros((M, r))
    W_s = np.zeros((P, r))
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for col in range(r):
        idx = rng.randint(0, len(triples) - 1)
        i, j, k = triples[idx]
        sign = rng.choice([-1, 1])
        U_s[i * m + j, col] = sign
        V_s[j * p + k, col] = sign
        W_s[k * n + i, col] = sign
    init_candidates.append(('sparse_struct', U_s, V_s, W_s))
    
    # Init 2: Block-structured - assign multiple triples per column
    U_b = np.zeros((N, r))
    V_b = np.zeros((M, r))
    W_b = np.zeros((P, r))
    for col in range(r):
        # Assign 1-3 triples to each column
        num_triples = rng.randint(1, min(3, r))
        for _ in range(num_triples):
            idx = rng.randint(0, len(triples) - 1)
            i, j, k = triples[idx]
            sign = rng.choice([-1, 1])
            U_b[i * m + j, col] += sign
            V_b[j * p + k, col] += sign
            W_b[k * n + i, col] += sign
    init_candidates.append(('block_multi', U_b, V_b, W_b))
    
    # Init 3: Perturbed schoolbook (if r close to 60)
    if r >= 50:
        U_p = np.zeros((N, r))
        V_p = np.zeros((M, r))
        W_p = np.zeros((P, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U_p[i * m + j, col] = 1.0 + rng.randn() * 0.1
                        V_p[j * p + k, col] = 1.0 + rng.randn() * 0.1
                        W_p[k * n + i, col] = 1.0 + rng.randn() * 0.1
                        col += 1
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        init_candidates.append(('perturbed_school', U_p, V_p, W_p))
    
    # Init 4: Small Gaussian
    init_candidates.append(('small_gauss', rng.randn(N, r) * 0.2, 
                            rng.randn(M, r) * 0.2, 
                            rng.randn(P, r) * 0.2))
    
    for init_name, U_init, V_init, W_init in init_candidates:
        if time.time() - start_time > timeout - 3:
            break
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        result = structured_als(U, V, W, T, nonzero_indices, zero_indices, 
                                max_iters=max_iters, timeout=timeout, start_time=start_time)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    T = build_tensor()
    nonzero_indices = get_nonzero_indices()
    zero_indices = get_zero_indices(300, seed=42)
    
    # Schoolbook fallback
    U, V, W = schoolbook_decomposition()
    best_rank = n * m * p
    best_result = (U, V, W)
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    lower_bound = max(N, M, P)  # 20
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time() - 20
        if remaining < 10:
            break
        
        found = None
        
        # Allocate time based on rank
        if r <= 25:
            max_trials = 4
            per_trial_time = min(remaining / max_trials, 25)
        elif r <= 40:
            max_trials = 3
            per_trial_time = min(remaining / max_trials, 20)
        else:
            max_trials = 2
            per_trial_time = min(remaining / max_trials, 15)
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            result = try_rank_structured(r, T, nonzero_indices, zero_indices,
                                         max_iters=60, timeout=per_trial_time, seed=42 + trial * 13)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_rank = r
            best_result = found
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
