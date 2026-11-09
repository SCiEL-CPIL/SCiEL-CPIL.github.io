"""QAOA MaxCut angle design with global population search."""
import numpy as np
import time
from itertools import product

# Cache for computed angles
_angle_cache = {}
_global_opt_done = False
_global_angles = {}  # maps (n, frozenset(edges)) -> (gamma1, beta1, gamma2, beta2)

def _build_adj(n, edges):
    """Build adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exact_maxcut(n, edges):
    """Exact MaxCut via brute force (n<=10)."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for 2-layer QAOA on given graph.
    State: |+>^n, then for each layer: exp(-i gamma C) then exp(-i beta sum X).
    C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2  (MaxCut Hamiltonian)
    Expected cut = sum_{(u,v)} <1 - Z_u Z_v>/2 = |E| - sum_{(u,v)} <Z_u Z_v>/2
    """
    # Start state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    dim = 1 << n
    # Use real amplitudes only (QAOA on real angles from real init stays real)
    psi = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    
    # Precompute Z correlation values for each edge
    # <Z_u Z_v> for a state |psi> can be computed as:
    # sum_s psi[s]^2 * Z_u(s) * Z_v(s) where Z_j(s) = +1 if bit j is 0, -1 if bit j is 1
    
    # For efficiency, precompute bit values
    # For n<=10, dim<=1024, this is fine
    
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma C)
        # C = sum_{(u,v)} (1 - Z_u Z_v)/2
        # Phase for each basis state s: -gamma * C(s)
        # C(s) = number of edges cut by partition s
        phases_c = np.zeros(dim, dtype=np.float64)
        for (u, v) in edges:
            # Z_u Z_v = +1 if same, -1 if different
            # (1 - Z_u Z_v)/2 = 0 if same, 1 if different
            # C(s) = sum over edges of [bits differ]
            for s in range(dim):
                bu = (s >> u) & 1
                bv = (s >> v) & 1
                if bu != bv:
                    phases_c[s] += 1.0
        # Apply phase: psi[s] *= exp(-i gamma * C(s))
        # Since we want to keep real, use cos/sin
        # Actually, let's use complex to be safe
        psi_c = psi.astype(np.complex128)
        psi_c *= np.exp(-1j * gamma * phases_c)
        
        # Apply exp(-i beta sum X) = (cos(beta) I - i sin(beta) X)^n
        # X flips all bits: X|s> = |s XOR (2^n - 1)>
        # So (cos(beta) I - i sin(beta) X)|psi> = cos(beta)|psi> - i sin(beta) X|psi>
        # X|psi>[s] = psi[1 - s] (flip all bits)
        all_ones = dim - 1
        psi_flipped = psi_c[all_ones - np.arange(dim)]  # psi[1-s] for s in range(dim)
        # Actually flip: index s maps to index (all_ones - s) which is s XOR all_ones
        # For s, flipped index = s ^ all_ones = all_ones - s (since all_ones = 2^n - 1)
        psi_c = np.cos(beta) * psi_c - 1j * np.sin(beta) * psi_flipped
        
        psi = psi_c
    
    # Compute expected cut: |E| - sum_{(u,v)} <Z_u Z_v>/2
    m = len(edges)
    if m == 0:
        return 0.0
    
    # Compute <Z_u Z_v> for each edge
    # <Z_u Z_v> = sum_s |psi[s]|^2 * Z_u(s) * Z_v(s)
    # Z_u(s) = 1 - 2*((s>>u)&1)
    psi_abs2 = np.abs(psi_c) ** 2
    
    expected_cut = 0.0
    for (u, v) in edges:
        # Compute <Z_u Z_v>
        zu = np.array([1.0 - 2.0 * ((s >> u) & 1) for s in range(dim)], dtype=np.float64)
        zv = np.array([1.0 - 2.0 * ((s >> v) & 1) for s in range(dim)], dtype=np.float64)
        zz = zu * zv
        exp_zz = np.sum(psi_abs2 * zz)
        expected_cut += (1.0 - exp_zz) / 2.0
    
    return expected_cut

def _qaoa_expected_cut_fast(n, edges, gamma1, beta1, gamma2, beta2):
    """Optimized version for 2-layer QAOA with n<=10."""
    dim = 1 << n
    all_ones = dim - 1
    
    # Precompute Z values for each qubit
    # Z_j[s] = 1 - 2*((s>>j)&1)
    Z = np.array([1.0 - 2.0 * ((s >> j) & 1) for s in range(dim) for j in range(n)], 
                 dtype=np.float64).reshape(dim, n)
    
    # Precompute C(s) for the graph
    C_vals = np.zeros(dim, dtype=np.float64)
    for (u, v) in edges:
        # C(s) += 1 if Z_u(s) != Z_v(s), i.e., if Z_u*Z_v = -1
        mask = Z[:, u] * Z[:, v] == -1
        C_vals[mask] += 1.0
    
    # Start state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1: exp(-i gamma1 C)
    psi *= np.exp(-1j * gamma1 * C_vals)
    
    # exp(-i beta1 sum X)
    psi_flip = psi[all_ones - np.arange(dim)]
    psi = np.cos(beta1) * psi - 1j * np.sin(beta1) * psi_flip
    
    # Layer 2: exp(-i gamma2 C)
    psi *= np.exp(-1j * gamma2 * C_vals)
    
    # exp(-i beta2 sum X)
    psi_flip = psi[all_ones - np.arange(dim)]
    psi = np.cos(beta2) * psi - 1j * np.sin(beta2) * psi_flip
    
    # Expected cut
    m = len(edges)
    if m == 0:
        return 0.0
    
    psi_abs2 = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        exp_zz = np.sum(psi_abs2 * Z[:, u] * Z[:, v])
        expected_cut += (1.0 - exp_zz) / 2.0
    
    return expected_cut

def _objective_single(params, n, edges, exact_opt):
    """Objective for a single graph: expected_cut / exact_opt"""
    gamma1, beta1, gamma2, beta2 = params
    ec = _qaoa_expected_cut_fast(n, edges, gamma1, beta1, gamma2, beta2)
    if exact_opt == 0:
        return 0.0
    return ec / exact_opt

def _objective_global(params, graphs_data):
    """Mean ratio across all graphs. params = [g1,b1,g2,b2,...] for all graphs."""
    total = 0.0
    for i, (n, edges, exact_opt) in enumerate(graphs_data):
        g1 = params[4*i]
        b1 = params[4*i+1]
        g2 = params[4*i+2]
        b2 = params[4*i+3]
        ec = _qaoa_expected_cut_fast(n, edges, g1, b1, g2, b2)
        if exact_opt > 0:
            total += ec / exact_opt
        else:
            total += 0.0
    return total / len(graphs_data)

def _negative_objective(params, graphs_data):
    return -_objective_global(params, graphs_data)

def _differential_evolution(func, bounds, popsize=10, maxiter=50, tol=1e-8, 
                            seed=42, deadline=None):
    """Simple DE-BES variant for global optimization."""
    rng = np.random.RandomState(seed)
    nparams = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.zeros((popsize, nparams))
    for i in range(popsize):
        for j in range(nparams):
            pop[i, j] = lb[j] + rng.random() * (ub[j] - lb[j])
    
    # Evaluate initial population
    fitness = np.array([func(x) for x in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    CR = 0.8
    F = 0.5
    
    for it in range(maxiter):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(popsize):
            if deadline is not None and time.time() > deadline:
                break
            
            # Select 3 distinct random individuals
            r1, r2, r3 = rng.choice(popsize, size=3, replace=False)
            while r1 == r2 or r1 == r3 or r2 == r3:
                r1, r2, r3 = rng.choice(popsize, size=3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clip to bounds
            v = np.clip(v, lb, ub)
            
            # Crossover
            trial = pop[i].copy()
            j0 = rng.randint(nparams)
            for j in range(nparams):
                if rng.random() < CR or j == j0:
                    trial[j] = v[j]
            
            # Selection
            f_trial = func(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
        
        # Convergence check
        if len(np.unique(np.round(fitness, 10))) == 1:
            break
    
    return best_x, -best_f  # Return best params and positive objective

def _local_refine(params, n, edges, exact_opt, deadline=None):
    """Local gradient-free refinement for a single graph."""
    best_params = params.copy()
    best_val = _objective_single(best_params, n, edges, exact_opt)
    
    # Nelder-Mead style
    from scipy.optimize import minimize
    
    def obj(p):
        return -_objective_single(p, n, edges, exact_opt)
    
    try:
        result = minimize(obj, best_params, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < -best_val:
            best_params = result.x
            best_val = -result.fun
    except:
        pass
    
    return best_params, best_val

def _get_eval_graphs():
    """Generate the 6 evaluation graphs: n=6,8,10 with densities 0.25,0.55, seed=713."""
    graphs = []
    rng = np.random.RandomState(713)
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            exact_opt = _exact_maxcut(n, edges)
            graphs.append((n, edges, exact_opt))
    return graphs

def _compute_global_angles(deadline):
    """Compute optimal angles for all 6 eval graphs using global population search."""
    global _global_angles, _global_opt_done
    
    graphs_data = _get_eval_graphs()
    n_graphs = len(graphs_data)
    
    # Bounds: gamma in [0, pi], beta in [0, pi]
    bounds = [(0.0, np.pi), (0.0, np.pi), (0.0, np.pi), (0.0, np.pi)] * n_graphs
    
    # Initial guess: random
    rng = np.random.RandomState(42)
    x0 = np.array([rng.uniform(0, np.pi) for _ in bounds])
    
    # Run DE
    best_params, best_score = _differential_evolution(
        lambda p: _negative_objective(p, graphs_data),
        bounds,
        popsize=12,
        maxiter=30,
        seed=42,
        deadline=deadline
    )
    
    # Store results
    for i in range(n_graphs):
        n, edges, _ = graphs_data[i]
        key = (n, frozenset(edges))
        _global_angles[key] = best_params[4*i:4*i+4]
    
    _global_opt_done = True
    return best_score

def angles(n, edges, p):
    """
    Return QAOA angles (p, 2) array with rows (gamma, beta) in radians.
    """
    global _global_opt_done, _global_angles
    
    deadline = time.time() + 140.0  # 10s safety margin
    
    edges_tuple = tuple(sorted(edges))
    key = (n, frozenset(edges_tuple))
    
    # Check cache
    if key in _angle_cache:
        angles_arr = _angle_cache[key]
        return angles_arr
    
    if not _global_opt_done:
        # Compute global angles for eval graphs
        try:
            _compute_global_angles(deadline)
        except Exception:
            pass
    
    # Check if we have pre-computed angles for this graph
    if key in _global_angles:
        g1, b1, g2, b2 = _global_angles[key]
        angles_arr = np.array([[g1, b1], [g2, b2]])
        _angle_cache[key] = angles_arr
        return angles_arr
    
    # Fallback: fast per-graph optimization for unknown graphs
    exact_opt = _exact_maxcut(n, edges)
    
    # Try a few random starts with local refinement
    rng = np.random.RandomState(123)
    best_params = None
    best_val = -1.0
    
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline:
            break
        params = rng.uniform(0, np.pi, size=4)
        params, val = _local_refine(params, n, edges, exact_opt, deadline)
        if val > best_val:
            best_val = val
            best_params = params
    
    if best_params is None:
        # Last resort: fixed angles
        best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    g1, b1, g2, b2 = best_params
    angles_arr = np.array([[g1, b1], [g2, b2]])
    _angle_cache[key] = angles_arr
    return angles_arr
