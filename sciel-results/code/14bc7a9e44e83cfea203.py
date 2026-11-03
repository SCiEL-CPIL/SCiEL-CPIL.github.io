"""
QAP Solver: Tabu Search with Hungarian Algorithm Perturbation (ILS-Hungarian).

Combines the robust Tabu Search structure of the current program with the
global reassignment capability of the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
as a perturbation mechanism. This allows the search to escape local optima
by occasionally performing a globally optimal reassignment based on marginal
contributions, rather than just random swaps.

Key features:
1. Fast O(n) swap delta for the main Tabu Search loop.
2. Periodic Hungarian Algorithm perturbation to jump to a new basin of attraction.
3. Adaptive tabu tenure and frequency-based diversification.
4. Multi-start with diverse initializations.
5. Strict time management with safety margins.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute cost change when swapping positions i and j.
    a_sym = a + a.T
    """
    x = p[i]
    y = p[j]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    val = np.dot(diff_a, diff_b[p])
    return val - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]

def _greedy_init(a, b, seed=42):
    """Greedy rank-matching initialization."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.copy()

def _random_init(n, seed=42):
    """Random permutation."""
    rng = random.Random(seed)
    p = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        p[i], p[j] = p[j], p[i]
    return p

def _hungarian_perturb(a, b, p, rng, num_perturb=5):
    """
    Perform a perturbation using the Hungarian algorithm.
    1. Select a subset of facilities to reassign.
    2. Compute the marginal cost for each facility-location pair.
    3. Solve the assignment problem for the subset.
    4. Update the permutation.
    
    This is a global move that can jump far in the search space.
    """
    n = len(p)
    # Select a random subset of facilities to reassign
    # Using a subset makes it faster than a full n x n Hungarian solve
    # But for n=256, full solve is ~O(n^3) = 16M ops, which is fast enough in scipy
    # However, to be safe and allow more iterations, let's try full solve first.
    # If it's too slow, we can reduce to subsets.
    
    # Compute the cost matrix for the Hungarian algorithm.
    # The cost of assigning facility i to location l is the marginal contribution.
    # C[i, l] = sum_j a[i, j] * b[l, p[j]] + sum_j a[j, i] * b[p[j], l]
    # Note: This ignores the interaction between the reassigned facilities themselves,
    # which is a common approximation in ILS-Hungarian methods.
    
    # Vectorized computation of marginal costs
    # b_p = b[p, :] is the current distance matrix indexed by facilities
    # We want C[i, l] = sum_j a[i, j] b[l, p[j]] + a[j, i] b[p[j], l]
    
    # Let's compute this efficiently.
    # term1[i, l] = sum_j a[i, j] * b[l, p[j]]
    # term2[i, l] = sum_j a[j, i] * b[p[j], l]
    
    # b_l_p = b[:, p] -> shape (n, n), where b_l_p[l, j] = b[l, p[j]]
    b_l_p = b[:, p] # (n, n)
    
    # term1 = a @ b_l_p
    # a is (n, n), b_l_p is (n, n)
    # term1[i, l] = sum_j a[i, j] * b_l_p[j, l]? No.
    # b_l_p[j, l] = b[j, p[l]]? No, b[:, p] means column j of result is b[:, p[j]].
    # So b_l_p[l, j] = b[l, p[j]].
    # We want sum_j a[i, j] * b[l, p[j]] = sum_j a[i, j] * b_l_p[l, j].
    # This is (a @ b_l_p.T)[i, l]?
    # (a @ b_l_p.T)[i, l] = sum_j a[i, j] * b_l_p.T[j, l] = sum_j a[i, j] * b_l_p[l, j]. Yes.
    
    term1 = a @ b_l_p.T # (n, n)
    
    # term2[i, l] = sum_j a[j, i] * b[p[j], l]
    # b_p_l = b[p, :] -> shape (n, n), where b_p_l[j, l] = b[p[j], l]
    b_p_l = b[p, :] # (n, n)
    
    # term2[i, l] = sum_j a[j, i] * b_p_l[j, l]
    # This is (a.T @ b_p_l)[i, l]?
    # (a.T @ b_p_l)[i, l] = sum_j a.T[i, j] * b_p_l[j, l] = sum_j a[j, i] * b_p_l[j, l]. Yes.
    
    term2 = a.T @ b_p_l # (n, n)
    
    C = term1 + term2
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    # row_ind are facilities, col_ind are locations.
    # We need to update p such that p[row_ind[k]] = col_ind[k].
    # But p is a permutation of facilities to locations.
    # If we reassign all facilities, we just set p = col_ind?
    # Wait, p[i] is the location of facility i.
    # linear_sum_assignment returns pairs (facility, location) that minimize sum C[i, l].
    # So yes, the new permutation is simply p_new[i] = col_ind[row_ind.index(i)]?
    # Since row_ind is a permutation of 0..n-1, we can just do:
    # p_new = np.empty(n)
    # p_new[row_ind] = col_ind
    
    # However, we might want to only reassign a subset to preserve some structure.
    # For now, let's do full reassignment as it's a strong perturbation.
    # To avoid getting stuck in the same local optimum, we add some noise to C.
    noise = rng.random_sample((n, n)) * 1000 # Small noise to break ties
    C_noisy = C + noise
    
    row_ind, col_ind = linear_sum_assignment(C_noisy)
    
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    return p_new

def _tabu_search(a, b, p, time_limit, start_time, seed=42):
    """
    Tabu Search with Hungarian Perturbation.
    """
    n = len(a)
    rng = random.Random(seed)
    a_sym = a + a.T  # Precompute symmetric flow matrix
    
    # Tabu parameters
    tabu_tenure = 20
    tabu_list = np.zeros((n, n), dtype=bool)
    move_history = []
    
    # Long-term memory
    freq = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        freq[i, p[i]] += 1
        
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    no_improve = 0
    
    freq_weight = 0.01
    freq_threshold = 100
    
    iteration = 0
    perturb_interval = 500 # Perturb every 500 iterations
    
    while time.time() - start_time < time_limit - 5:
        # Un-tabu old moves
        while move_history and move_history[0][0] + tabu_tenure <= iteration:
            _, ti, tj = move_history.pop(0)
            tabu_list[ti, tj] = False
            tabu_list[tj, ti] = False
        
        # Search for best move
        num_candidates = 50
        best_delta = np.inf
        best_move = None
        best_is_tabu = False
        best_aspiration = False
        
        for _ in range(num_candidates):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            
            delta = _swap_delta(a_sym, b, p, i, j)
            
            if iteration > freq_threshold:
                freq_penalty = freq_weight * (freq[i, p[i]] + freq[j, p[j]])
                adjusted_delta = delta + freq_penalty
            else:
                adjusted_delta = delta
            
            is_tabu = tabu_list[i, j]
            aspiration = is_tabu and (current_cost + delta < best_cost)
            
            if not is_tabu and delta < best_delta:
                best_delta = adjusted_delta
                best_move = (i, j)
                best_is_tabu = False
                best_aspiration = False
            elif is_tabu and aspiration and delta < best_delta:
                best_delta = adjusted_delta
                best_move = (i, j)
                best_is_tabu = True
                best_aspiration = True
        
        if best_move is None:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                i = (i + 1) % n
            best_move = (i, j)
            best_is_tabu = False
            best_aspiration = False
            best_delta = _swap_delta(a_sym, b, p, i, j)
        
        i, j = best_move
        x = p[i]
        y = p[j]
        
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        freq[i, y] += 1
        freq[j, x] += 1
        freq[i, x] -= 1
        freq[j, y] -= 1
        
        if not best_is_tabu or best_aspiration:
            tabu_list[i, j] = True
            tabu_list[j, i] = True
            move_history.append((iteration, i, j))
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        iteration += 1
        
        # Hungarian Perturbation
        if iteration % perturb_interval == 0:
            # Only perturb if we are stuck or randomly
            if no_improve > 100 or rng.random() < 0.3:
                p_new = _hungarian_perturb(a, b, p, rng)
                cost_new = _qap_cost(a, b, p_new)
                
                # Accept if better, or with some probability (simulated annealing style)
                # Since we don't have temperature, just accept if better or if very stuck
                if cost_new < best_cost or no_improve > 500:
                    p = p_new
                    current_cost = cost_new
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    no_improve = 0
                    # Reset tabu and freq
                    tabu_list.fill(False)
                    move_history.clear()
                    freq.fill(0)
                    for i in range(n):
                        freq[i, p[i]] += 1
                        
        if time.time() - start_time > time_limit - 8:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start
    num_restarts = 2
    
    for r in range(num_restarts):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 30:
            break
            
        if r == 0:
            p_init = _greedy_init(a, b, seed=42 + r)
        else:
            p_init = _random_init(n, seed=123 + r)
            
        ts_time = min(remaining_time - 10, time_limit / num_restarts)
        p_best, cost_best = _tabu_search(a, b, p_init.copy(), ts_time, start_time, seed=1000 + r)
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
            
    if best_p is None:
        best_p = _greedy_init(a, b, seed=0)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
