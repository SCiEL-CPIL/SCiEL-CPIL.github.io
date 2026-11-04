"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linear_sum_assignment

def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))

def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    Assumes symmetric A and B for optimization.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    a_diff = a[i, mask] - a[j, mask]
    b_diff = b[pj, p[mask]] - b[pi, p[mask]]
    
    delta_inter = 2.0 * np.dot(a_diff, b_diff)
    
    delta_self = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    
    return int(delta_inter + delta_self)

def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    """
    n = len(p)
    cost = _cost(a, b, p)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then move to next i
        if not improved:
            break
    
    return p, cost

def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm

    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p

def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # --- Phase 1: Initialization ---
    # Heuristic 1: Rank-matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Heuristic 2: Reverse
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
        
    # Heuristic 3: Spectral
    rng = np.random.default_rng(42)
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search(a, b, p3.copy(), deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # --- Phase 2: Hybrid Simulated Annealing + Local Search ---
    current_p = best_p.copy()
    current_cost = best_cost
    
    T = 1000.0  # Initial temperature
    T_min = 1.0
    cooling_rate = 0.995
    sa_deadline = deadline - 10.0  # Reserve time for final polish
    
    sa_steps_since_ls = 0
    ls_interval = 50  # Run LS every 50 SA steps
    
    while time.time() < sa_deadline:
        # Pick a random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        
        if i == j:
            continue
            
        delta = _delta_swap(a, b, current_p, i, j)
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
            sa_steps_since_ls += 1
            if sa_steps_since_ls >= ls_interval:
                # Run local search to polish
                current_p, current_cost = _local_search(a, b, current_p, sa_deadline)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                sa_steps_since_ls = 0
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / T):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                sa_steps_since_ls += 1
                if sa_steps_since_ls >= ls_interval:
                    current_p, current_cost = _local_search(a, b, current_p, sa_deadline)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    sa_steps_since_ls = 0
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If T is very low, we are effectively doing random restarts or ILS.
        # We can increase cooling rate if stuck, but fixed is okay for now.
        
    # --- Phase 3: Final Polish ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        # Run a few more local search passes
        polish_p, polish_cost = _local_search(a, b, polish_p, polish_deadline)
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
