"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from functools import lru_cache

_DEADLINE = 140.0  # 10s safety margin before 150s kill
_start_time = None
_cache = {}


def _exhaustive_maxcut(n, edges):
    """Compute exact MaxCut value via exhaustive search."""
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    best = 0
    for assignment in product([0, 1], repeat=n):
        cut = 0
        for u, v in edge_set:
            if assignment[u] != assignment[v]:
                cut += 1
        best = max(best, cut)
    return best


def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut for given QAOA angles.
    
    For p-layer QAOA with angles (gamma_1, ..., gamma_p, beta_1, ..., beta_p),
    the expected cut can be computed analytically.
    """
    p = len(gammas)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # For 2-layer QAOA, we can compute the expected cut analytically
    # The probability that edge (u,v) is cut depends on the angles
    # For unweighted MaxCut with C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    
    # Using the analytical formula for QAOA expected cut
    # For p=2: E[cut] = sum_{(u,v) in E} P[cut(u,v)]
    
    # The probability that nodes u and v end up in different partitions
    # after applying the QAOA circuit can be computed from the angles
    
    # For a single edge, the cut probability depends on:
    # - The initial state |+>^n
    # - The cost unitary exp(-i gamma * C)
    # - The mixer unitary exp(-i beta * sum X)
    
    # For small n, we can compute this exactly using the state vector
    # But for efficiency, let's use the analytical approximation
    
    # Actually, for n <= 10, we can compute the exact expected value
    # by tracking the state through the circuit
    
    # Initialize state: |+>^n = (1/sqrt(2^n)) * sum_{x} |x>
    # This is a uniform superposition
    
    # Apply exp(-i gamma_1 * C) where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # Then exp(-i beta_1 * sum X)
    # Then exp(-i gamma_2 * C)  
    # Then exp(-i beta_2 * sum X)
    
    # For n=10, the state vector has 2^10 = 1024 components, which is manageable
    
    dim = 2 ** n
    # Initial state: uniform superposition
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Build the cost Hamiltonian matrix C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # For each edge, (1 - Z_u Z_v)/2 = (1/2)(I - Z_u Z_v)
    # Z_u Z_v is diagonal with entries +1 if x_u == x_v, -1 if x_u != x_v
    
    # We'll apply the unitaries sequentially
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma * C)
        # C is diagonal in the computational basis
        # C|x> = c(x) |x> where c(x) = number of edges cut by assignment x
        c_vals = np.zeros(dim)
        for x in range(dim):
            bits = [(x >> i) & 1 for i in range(n)]
            cut_count = 0
            for u, v in edge_set:
                if bits[u] != bits[v]:
                    cut_count += 1
            c_vals[x] = cut_count
        
        # exp(-i gamma * C) is diagonal with entries exp(-i gamma * c(x))
        phase_cost = np.exp(-1j * gamma * c_vals)
        state = phase_cost * state
        
        # Apply exp(-i beta * sum X)
        # sum X = sum_i X_i
        # X flips each bit, so sum X |x> = sum_i |x with bit i flipped>
        # exp(-i beta * sum X) = product... no, it's not a product
        # We need to apply the full matrix
        
        # For efficiency, use the fact that exp(-i beta * sum_i X_i) can be applied
        # by rotating in the X basis
        # Actually, let's just build the matrix for n <= 10
        # dim = 1024 for n=10, matrix is 1024x1024 = ~1M entries, OK
        
        # Build exp(-i beta * sum X)
        # sum X = sum_i X_i
        # This matrix has entries: <x| (sum_i X_i) |y> = sum_i <x|X_i|y>
        # X_i flips bit i, so <x|X_i|y> = 1 if y = x with bit i flipped, else 0
        # So (sum_i X_i)|y> = sum_i |y with bit i flipped>
        
        # For the exponential, we can use matrix exponential or diagonalize
        # Since sum X has eigenvalues that are integers from -n to n in steps of 2
        # Actually, let's just compute it directly for small n
        
        # Alternative: use the fact that we can apply the mixer by
        # transforming to X basis, applying phase, transforming back
        # But simpler: just build the matrix
        
        if n <= 10:
            # Build sum X matrix
            sumX = np.zeros((dim, dim), dtype=np.complex128)
            for y in range(dim):
                for i in range(n):
                    x = y ^ (1 << i)  # flip bit i
                    sumX[x, y] += 1.0
            
            # Compute exp(-i beta * sumX) using scipy or numpy
            from scipy.linalg import expm
            mixer = expm(-1j * beta * sumX)
            state = mixer @ state
        else:
            # For larger n, use approximation
            pass
    
    # Compute expected cut
    # E[cut] = sum_{(u,v) in E} P[cut(u,v)]
    # P[cut(u,v)] = sum_x |<x|state>|^2 * [x_u != x_v]
    
    probs = np.abs(state) ** 2
    expected = 0.0
    for u, v in edge_set:
        for x in range(dim):
            bits_u = (x >> u) & 1
            bits_v = (x >> v) & 1
            if bits_u != bits_v:
                expected += probs[x]
    
    return expected


def _optimize_angles_for_graph(n, edges, p, max_iter=500, timeout=10.0):
    """Optimize QAOA angles for a single graph."""
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    exact_opt = _exhaustive_maxcut(n, edges)
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    # Initial guess
    best_angles = np.array([[0.35, 0.25]] * p)
    best_ratio = _expected_cut(n, edges, best_angles[:, 0], best_angles[:, 1]) / exact_opt
    
    # Use Nelder-Mead or similar
    from scipy.optimize import minimize
    
    def objective(angles_flat):
        gammas = angles_flat[:p]
        betas = angles_flat[p:]
        exp_cut = _expected_cut(n, edges, gammas, betas)
        return -exp_cut / exact_opt  # minimize negative ratio
    
    x0 = np.concatenate([best_angles[:, 0], best_angles[:, 1]])
    
    # Multiple restarts
    for restart in range(5):
        if time.time() - _start_time > timeout:
            break
        
        if restart == 0:
            x0_trial = x0.copy()
        else:
            x0_trial = np.random.uniform(-np.pi, np.pi, size=2*p)
        
        try:
            result = minimize(objective, x0_trial, method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if result.fun < -best_ratio:
                best_ratio = -result.fun
                best_angles = np.column_stack([result.x[:p], result.x[p:]])
        except Exception:
            pass
    
    return best_angles


def angles(n, edges, p):
    global _start_time, _cache
    
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we have cached result for this graph
    cache_key = (n, tuple(sorted(edges)), p)
    if cache_key in _cache:
        return _cache[cache_key]
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Not enough time, return default
        result = np.tile([0.35, 0.25], (p, 1))
        _cache[cache_key] = result
        return result
    
    # Optimize angles for this graph
    try:
        result = _optimize_angles_for_graph(n, edges, p, timeout=_DEADLINE - elapsed - 2)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    _cache[cache_key] = result
    return result
# EVOLVE-BLOCK-END

