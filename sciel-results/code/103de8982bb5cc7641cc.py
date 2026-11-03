"""Simulated Annealing approach with fast swap evaluation and diversification."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _init_sa(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute symmetric flow matrix for faster delta calculation
    # a_sym[i, k] = a[i, k] + a[k, i]
    a_sym = a + a.T
    
    # Initial solution: Greedy rank matching (same as before, good starting point)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # SA Parameters
    # Tail256c is a hard instance. We need a long run.
    # Start with a high temperature to allow exploration, cool down slowly.
    # Estimate initial temperature based on typical swap deltas.
    # Let's sample a few random swaps to estimate delta magnitude.
    sample_deltas = []
    for _ in range(100):
        i, j = random.sample(range(n), 2)
        x, y = p[i], p[j]
        diff_a = a_sym[i, :] - a_sym[j, :]
        diff_b = b[y, :] - b[x, :]
        temp_vec = diff_a * diff_b[p]
        delta = np.sum(temp_vec) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
        sample_deltas.append(abs(delta))
    
    avg_delta = np.mean(sample_deltas) if sample_deltas else 1000.0
    if avg_delta < 1e-6:
        avg_delta = 1000.0
        
    T_start = 10.0 * avg_delta
    T_end = 1e-6 * avg_delta
    # Number of iterations to fit in time
    # We estimate time per iteration.
    # One iteration involves:
    # 1. Pick 2 random indices
    # 2. Compute delta (O(n) vectorized)
    # 3. Accept/Reject
    # 4. Update permutation if accepted
    # 5. Check time
    
    # Let's run for a fixed number of iterations or until time limit.
    # We will use a while loop with time checks.
    
    iterations = 0
    # Cooling schedule: Geometric cooling is standard.
    # T_k = T_start * (T_end / T_start)^(k / K)
    # We don't know K in advance, so we can use a linear or geometric step based on time.
    # Better: Use a geometric cooling factor per iteration.
    # If we expect ~100k iterations in 115s, that's ~1.15us per iter? No, numpy overhead is higher.
    # Let's assume ~10k-50k iterations are feasible.
    # Let's target ~50,000 iterations.
    target_iters = 50000
    alpha = (T_end / T_start) ** (1.0 / target_iters)
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        # Pick random pair
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Compute delta
        # delta = sum_{k != i,j} (a_sym[i,k] - a_sym[j,k]) * (b[y,p[k]] - b[x,p[k]])
        # Using precomputed a_sym
        diff_a = a_sym[i, :] - a_sym[j, :]
        diff_b = b[y, :] - b[x, :]
        
        # Vectorized dot product
        # temp = diff_a * diff_b[p]
        # This creates a temp array of size n.
        # To be faster, we can use np.dot(diff_a, diff_b[p])
        # But we need to subtract the i and j terms.
        
        # Let's optimize the delta calculation.
        # diff_a and diff_b are vectors of size n.
        # We need sum_{k} diff_a[k] * diff_b[p[k]] - diff_a[i]*diff_b[x] - diff_a[j]*diff_b[y]
        
        # np.dot is fast.
        # Let's try to avoid creating the intermediate array diff_b[p] if possible?
        # No, fancy indexing is necessary.
        
        # Alternative: Precompute b rows? No, p changes.
        
        # Let's just do the dot product.
        # To save time, we can check if delta is likely to be large? No.
        
        # One optimization: If we store b as a matrix, diff_b[p] is a gather operation.
        
        val = np.dot(diff_a, diff_b[p])
        delta = val - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            accept = True
        else:
            # Deterioration
            # Prob = exp(-delta / T)
            # If T is very small, exp(-large) is 0.
            # Optimization: if delta > 0 and T < delta / 50, reject immediately.
            if T < 1e-10:
                accept = False
            else:
                # random() is fast
                if random.random() < np.exp(-delta / T):
                    accept = True
                else:
                    accept = False
        
        if accept:
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= alpha
        iterations += 1
        
        # Safety check every 1000 iterations
        if iterations % 1000 == 0:
             if time.time() - start_time > time_limit:
                 break

    return best_p.tolist()

def solve(a, b):
    return _init_sa(a, b)
# EVOLVE-BLOCK-END
