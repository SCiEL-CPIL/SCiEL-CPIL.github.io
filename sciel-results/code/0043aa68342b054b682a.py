"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute the cost change when swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    # For each position k, the contribution changes because:
    # - a[k,i] * b[p[k], p[i]] becomes a[k,i] * b[p[k], p[j]]
    # - a[i,k] * b[p[i], p[k]] becomes a[i,k] * b[p[j], p[k]]
    # Similarly for j
    
    # Vectorized computation of delta
    # d[k] = a[k,i]*(b[p[k],pj] - b[p[k],pi]) + a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    #        + a[k,j]*(b[p[k],pi] - b[p[k],pj]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    
    bpj_col = b[:, pj][p]  # b[p[k], pj] for all k
    bpi_col = b[:, pi][p]  # b[p[k], pi] for all k
    bpj_row = b[pj, :][p]  # b[pj, p[k]] for all k
    bpi_row = b[pi, :][p]  # b[pi, p[k]] for all k
    
    diff_col = bpj_col - bpi_col  # b[p[k],pj] - b[p[k],pi]
    diff_row = bpj_row - bpi_row  # b[pj,p[k]] - b[pi,p[k]]
    
    delta = (a[:, i] * diff_col + 
             a[i, :] * diff_row + 
             a[:, j] * (-diff_col) + 
             a[j, :] * (-diff_row))
    
    # Remove self-contributions (i and j don't interact with themselves in the swap)
    delta[i] = 0
    delta[j] = 0
    
    # Add the direct interaction between i and j
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    
    return int(np.sum(delta) + direct)


def _local_search(a, b, p, deadline, best_cost, best_p):
    """First-improvement local search using swap moves."""
    n = len(p)
    cur_cost = _qap_cost(a, b, p)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur_cost += d
                    improved = True
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_p = p.copy()
                    break
    return best_cost, best_p


def _tabu_search(a, b, p, deadline, best_cost, best_p, rng):
    """Tabu search with adaptive tabu tenure."""
    n = len(p)
    cur_cost = _qap_cost(a, b, p)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p.copy()
    
    # Tabu memory: (i, j) -> iteration when it becomes non-tabu
    tabu_until = {}
    tabu_tenure_base = 7
    tabu_tenure_var = 3
    it = 0
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        it += 1
        no_improve += 1
        
        # Adaptive tabu tenure
        tenure = tabu_tenure_base + rng.integers(0, tabu_tenure_var + 1)
        
        # Find best non-tabu move (or best tabu move if no non-tabu improvement)
        best_d = 0
        best_i, best_j = -1, -1
        best_tabu_d = 0
        best_tabu_i, best_tabu_j = -1, -1
        
        # Sample random pairs to keep it fast, or do full scan
        # For n=100, full scan is 4950 pairs, which is feasible
        for i in range(n):
            if time.time() >= deadline:
                return best_cost, best_p
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                
                # Check if tabu
                key = (i, j)
                is_tabu = key in tabu_until and it < tabu_until[key]
                
                if not is_tabu:
                    if d < best_d:
                        best_d = d
                        best_i, best_j = i, j
                else:
                    if d < best_tabu_d:
                        best_tabu_d = d
                        best_tabu_i, best_tabu_j = i, j
        
        # Decide which move to take
        if best_d < 0:
            # Take best improving non-tabu move
            i, j = best_i, best_j
        elif best_tabu_d < 0 and no_improve >= max_no_improve:
            # Aspiration: take best tabu move if we've stagnated
            i, j = best_tabu_i, best_tabu_j
            no_improve = 0
        else:
            # No improving move found; perturb randomly
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            d = _swap_delta(a, b, p, i, j)
        
        if i >= 0 and i < n and j >= 0 and j < n and i != j:
            # Execute the move
            p[i], p[j] = p[j], p[i]
            cur_cost += d
            
            # Update tabu memory
            key = (i, j)
            if i > j:
                key = (j, i)
            tabu_until[key] = it + tenure
            # Clean up old tabu entries
            for k in list(tabu_until.keys()):
                if it >= tabu_until[k]:
                    del tabu_until[k]
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()
                no_improve = 0
    
    return best_cost, best_p


def _constructive_start(a, b, rng):
    """Generate a good initial permutation using rank matching."""
    n = len(a)
    
    # Rank facilities by total flow
    flow_sums = a.sum(axis=1)
    # Rank locations by total distance
    dist_sums = b.sum(axis=1)
    
    # Sort facilities in descending order of flow
    facility_order = np.argsort(-flow_sums, kind='stable')
    # Sort locations in ascending order of distance (central first)
    location_order = np.argsort(dist_sums, kind='stable')
    
    # Assign
    p = np.empty(n, dtype=np.int64)
    for idx, fac in enumerate(facility_order):
        p[fac] = location_order[idx]
    
    # Add some randomization
    num_swaps = rng.integers(5, 20)
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p[i], p[j] = p[j], p[i]
    
    return p


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    best_cost = float('inf')
    best_p = None
    
    # Phase 1: Constructive start + local search
    p = _constructive_start(a, b, rng)
    best_cost, best_p = _local_search(a, b, p, deadline - 80.0, best_cost, best_p)
    
    # Phase 2: Tabu search to escape local optima
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        tabu_deadline = deadline - 15.0
        best_cost, best_p = _tabu_search(a, b, best_p.copy(), tabu_deadline, 
                                          best_cost, best_p, rng)
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2.0:
        best_cost, best_p = _local_search(a, b, best_p, deadline, best_cost, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
