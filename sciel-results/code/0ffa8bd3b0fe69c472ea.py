import numpy as np
import time
import random
from fractions import Fraction
from itertools import product
from scipy.optimize import linprog

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                i = a // m; j = a % m
                jp = b // p; k = b % p
                kn = g // n; ip = g % n
                if i != ip or j != jp or k != kn:
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[g, r]
                    if abs(val) > tol:
                        return False
    return True

def _laser_support_pattern(n, m, p, target_rank, seed=42):
    """
    Use the laser method: find a sparse support pattern for U, V, W
    that can achieve the given rank via tropical LP relaxation.
    
    The idea: each rank-1 term (r) contributes to specific (i,j,k) entries.
    We want to assign which (i,j) pairs use which rank indices in U,
    which (j,k) pairs use which in V, and which (k,i) pairs use which in W.
    
    This is formulated as a covering problem: for each (i,j,k), there must
    be at least one rank r such that (i,j) uses r in U, (j,k) uses r in V,
    and (k,i) uses r in W.
    
    We use a greedy randomized approach to find feasible patterns.
    """
    rng = random.Random(seed)
    
    # For each factor, determine how many rank indices each entry uses
    # Start with minimal: 1 per entry, increase if needed
    
    def _try_pattern(nnz_u, nnz_v, nnz_w):
        """Try a pattern where each U entry uses nnz_u rank indices, etc."""
        
        # Assign rank indices to each entry
        # U: (i,j) -> set of rank indices
        U_assign = {}
        for i in range(n):
            for j in range(m):
                cols = set(rng.sample(range(target_rank), min(nnz_u, target_rank)))
                U_assign[(i, j)] = cols
        
        V_assign = {}
        for j in range(m):
            for k in range(p):
                cols = set(rng.sample(range(target_rank), min(nnz_v, target_rank)))
                V_assign[(j, k)] = cols
        
        W_assign = {}
        for k in range(p):
            for i in range(n):
                cols = set(rng.sample(range(target_rank), min(nnz_w, target_rank)))
                W_assign[(k, i)] = cols
        
        # Check coverage: for each (i,j,k), is there a common rank index?
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    common = U_assign[(i,j)] & V_assign[(j,k)] & W_assign[(k,i)]
                    if not common:
                        return None
        
        # Build the support pattern
        U_support = np.zeros((n * m, target_rank), dtype=bool)
        for i in range(n):
            for j in range(m):
                for r in U_assign[(i,j)]:
                    U_support[i*m+j, r] = True
        
        V_support = np.zeros((m * p, target_rank), dtype=bool)
        for j in range(m):
            for k in range(p):
                for r in V_assign[(j,k)]:
                    V_support[j*p+k, r] = True
        
        W_support = np.zeros((p * n, target_rank), dtype=bool)
        for k in range(p):
            for i in range(n):
                for r in W_assign[(k,i)]:
                    W_support[k*n+i, r] = True
        
        return U_support, V_support, W_support
    
    # Try increasing sparsity
    for nnz in [1, 2, 3, 4]:
        for _ in range(20):
            result = _try_pattern(nnz, nnz, nnz)
            if result is not None:
                return result
    
    return None

def _solve_exact_from_support(U_support, V_support, W_support, n, m, p):
    """
    Given a support pattern, solve for exact rational values.
    
    We need: for each (i,j,k), sum_{r in common(i,j,k)} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    And: for non-matching triples, the sum = 0.
    
    This is a system of polynomial equations. We try a structured approach:
    set most values to 1 or -1, and solve for the rest.
    """
    target_rank = U_support.shape[1]
    
    # Initialize with 1s on support
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    for a in range(n * m):
        for r in range(target_rank):
            if U_support[a, r]:
                U[a, r] = 1.0
    
    for b in range(m * p):
        for r in range(target_rank):
            if V_support[b, r]:
                V[b, r] = 1.0
    
    for g in range(p * n):
        for r in range(target_rank):
            if W_support[g, r]:
                W[g, r] = 1.0
    
    # Check if this simple assignment works
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # Try sign patterns
    for trial in range(100):
        rng = random.Random(trial)
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for a in range(n * m):
            for r in range(target_rank):
                if U_support[a, r]:
                    U[a, r] = rng.choice([1.0, -1.0])
        
        for b in range(m * p):
            for r in range(target_rank):
                if V_support[b, r]:
                    V[b, r] = rng.choice([1.0, -1.0])
        
        for g in range(p * n):
            for r in range(target_rank):
                if W_support[g, r]:
                    W[g, r] = rng.choice([1.0, -1.0])
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    # Try solving linear system for W given U, V
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    rng = random.Random(123)
    for a in range(n * m):
        for r in range(target_rank):
            if U_support[a, r]:
                U[a, r] = rng.choice([1.0, -1.0])
    
    for b in range(m * p):
        for r in range(target_rank):
            if V_support[b, r]:
                V[b, r] = rng.choice([1.0, -1.0])
    
    # Solve for W: for each g=(k,i), we need
    # sum_r U[a,r]*V[b,r]*W[g,r] = T[a,b,g] for all (a,b)
    # where a=(i',j), b=(j',k), g=(k,i)
    # T = 1 if i'=i, j'=j, k'=k; else 0
    
    solved = True
    for g in range(p * n):
        k = g // n
        i = g % n
        
        # Build linear system for W[g,:]
        # Variables: W[g, r] for r in support of W[g]
        w_support = [r for r in range(target_rank) if W_support[g, r]]
        num_vars = len(w_support)
        
        if num_vars == 0:
            solved = False
            break
        
        # Equations: for each (a, b) where a=(i',j), b=(j',k)
        # sum_{r in w_support} U[a,r]*V[b,r]*W[g,r] = T[a,b,g]
        num_eqs = n * m * m * p
        A = np.zeros((num_eqs, num_vars))
        t = np.zeros(num_eqs)
        
        idx = 0
        for i_prime in range(n):
            for j in range(m):
                a = i_prime * m + j
                for j_prime in range(m):
                    for k_prime in range(p):
                        b = j_prime * p + k_prime
                        for vi, r in enumerate(w_support):
                            A[idx, vi] = U[a, r] * V[b, r]
                        if i_prime == i and j == j_prime and k_prime == k:
                            t[idx] = 1.0
                        else:
                            t[idx] = 0.0
                        idx += 1
        
        # Least squares
        w_vals, residuals, rank, sv = np.linalg.lstsq(A, t, rcond=None)
        
        # Check if solution is exact
        if np.max(np.abs(A @ w_vals - t)) > 1e-8:
            # Try rounding
            w_rounded = np.round(w_vals * 2) / 2
            if np.max(np.abs(A @ w_rounded - t)) > 1e-8:
                solved = False
                break
        
        for vi, r in enumerate(w_support):
            W[g, r] = w_vals[vi]
    
    if solved:
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _try_laser_decomposition(n, m, p, target_rank, seed=42):
    """
    Try the laser method: find a sparse support pattern and solve for exact values.
    """
    start = time.time()
    
    for trial_seed in range(seed, seed + 50):
        if time.time() - start > 20:
            break
        
        support = _laser_support_pattern(n, m, p, target_rank, seed=trial_seed)
        if support is None:
            continue
        
        U_support, V_support, W_support = support
        result = _solve_exact_from_support(U_support, V_support, W_support, n, m, p)
        if result is not None:
            return result
    
    return None

def _try_structured_decomposition(n, m, p, target_rank, seed=42):
    """
    Try structured sparse patterns for U, V, then solve for W.
    """
    # First try the laser method
    result = _try_laser_decomposition(n, m, p, target_rank, seed=seed)
    if result is not None:
        return result
    
    # Fallback: original approach
    rng = random.Random(seed)
    
    for nnz_u in [1, 2, 3]:
        for nnz_v in [1, 2, 3]:
            if target_rank < n * m * p // (nnz_u * nnz_v):
                continue
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for a in range(n * m):
                cols = rng.sample(range(target_rank), min(nnz_u, target_rank))
                for c in cols:
                    U[a, c] = rng.choice([1, -1, 0.5, -0.5])
            
            for b in range(m * p):
                cols = rng.sample(range(target_rank), min(nnz_v, target_rank))
                for c in cols:
                    V[b, c] = rng.choice([1, -1, 0.5, -0.5])
            
            W = np.zeros((p * n, target_rank))
            solved = True
            
            for g in range(p * n):
                k = g // n
                i = g % n
                
                num_ab = n * m * m * p
                A = np.zeros((num_ab, target_rank))
                t = np.zeros(num_ab)
                
                idx = 0
                for a in range(n * m):
                    ai = a // m; aj = a % m
                    for b in range(m * p):
                        bj = b // p; bk = b % p
                        A[idx, :] = U[a, :] * V[b, :]
                        if ai == i and aj == bj and bk == k:
                            t[idx] = 1.0
                        else:
                            t[idx] = 0.0
                        idx += 1
                
                try:
                    w, res, rk, sv = np.linalg.lstsq(A, t, rcond=None)
                    W[g, :] = w
                except:
                    solved = False
                    break
            
            if not solved:
                continue
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _als_optimize(n, m, p, initial_rank, max_time=50):
    """Alternating Least Squares for tensor decomposition."""
    start = time.time()
    deadline = start + max_time
    
    for target_rank in range(initial_rank - 1, max(20, initial_rank - 30), -1):
        if time.time() > deadline:
            break
        
        for seed in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update U
                GV = V.T @ V
                GW = W.T @ W
                A_mat = GV * GW + 1e-8 * np.eye(target_rank)
                
                for a in range(n * m):
                    bv = np.zeros(target_rank)
                    i = a // m; j = a % m
                    for k in range(p):
                        b = j * p + k
                        g = k * n + i
                        bv += V[b, :] * W[g, :]
                    try:
                        U[a, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                # Update V
                GU = U.T @ U
                GW = W.T @ W
                A_mat = GU * GW + 1e-8 * np.eye(target_rank)
                
                for b in range(m * p):
                    bv = np.zeros(target_rank)
                    j = b // p; k = b % p
                    for i in range(n):
                        a = i * m + j
                        g = k * n + i
                        bv += U[a, :] * W[g, :]
                    try:
                        V[b, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                # Update W
                GU = U.T @ U
                GV = V.T @ V
                A_mat = GU * GV + 1e-8 * np.eye(target_rank)
                
                for g in range(p * n):
                    bv = np.zeros(target_rank)
                    k = g // n; i = g % n
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        bv += U[a, :] * V[b, :]
                    try:
                        W[g, :] = np.linalg.solve(A_mat, bv)
                    except:
                        pass
                
                if it % 5 == 4:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a,:] * V[b,:] * W[g,:])
                                err += (val - 1.0)**2
                    np.random.seed(42)
                    for _ in range(100):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a//m; j = a%m
                        jp = b//p; k = b%p
                        kn = g//n; ip = g%n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a,:] * V[b,:] * W[g,:])
                            err += val**2
                    
                    if err < 1e-4:
                        Ur = _round_to_half_integer(U)
                        Vr = _round_to_half_integer(V)
                        Wr = _round_to_half_integer(W)
                        if _verify_decomposition(Ur, Vr, Wr, n, m, p):
                            return Ur, Vr, Wr
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try laser method first with various target ranks
    # Theoretical lower bound for 4x5x6 is around 4*5*6/2 = 60 for some methods
    # Try from lower to higher
    for target_rank in range(50, best_rank, 3):
        if time.time() > deadline - 60:
            break
        
        for seed in range(5):
            if time.time() > deadline - 55:
                break
            
            result = _try_laser_decomposition(n, m, p, target_rank, seed=seed)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                    break
        
        if time.time() > deadline - 60:
            break
    
    # Try structured decompositions if time remains
    if time.time() < deadline - 30:
        for target_rank in range(max(40, best_rank - 20), best_rank, 5):
            if time.time() > deadline - 25:
                break
            for seed in [42, 123, 456]:
                if time.time() > deadline - 20:
                    break
                result = _try_structured_decomposition(n, m, p, target_rank, seed=seed)
                if result is not None:
                    U, V, W = result
                    if U.shape[1] < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = U.shape[1]
                        break
    
    # Try ALS if time remains
    if time.time() < deadline - 15:
        remaining = int(deadline - time.time() - 5)
        if remaining > 5:
            result = _als_optimize(n, m, p, initial_rank=best_rank, max_time=remaining)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
