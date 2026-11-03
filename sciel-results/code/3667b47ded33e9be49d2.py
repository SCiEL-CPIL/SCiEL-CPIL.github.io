"""
Strategy: Parallel Island Model with Block Rotation and Error Repair.
Replaces sequential Hungarian+2opt with concurrent islands using k-opt moves.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

# Constants
N = 256
TIME_LIMIT = 110.0  # Safety margin from 120s
NUM_ISLANDS = 4
BLOCK_SIZES = [2, 3, 4]  # k-opt block sizes
MIGRATION_INTERVAL = 5.0  # seconds

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym=None):
    """Compute delta for swapping positions i and j."""
    if a_sym is None:
        a_sym = a + a.T
    x = p[i]
    y = p[j]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    return delta

def _block_rotation_delta(a, b, p, start, length, a_sym=None):
    """
    Compute delta for rotating a block of length 'length' starting at 'start' by 1 position.
    p[start:start+length] -> p[start+1:start+length] + p[start]
    """
    if a_sym is None:
        a_sym = a + a.T
        
    # Indices involved
    idxs = list(range(start, start + length))
    # The element at 'start' moves to 'start+length' (conceptually, but it's a rotation)
    # Actually, rotation right by 1: 
    # old: [x0, x1, ..., x_{L-1}]
    # new: [x_{L-1}, x0, ..., x_{L-2}]
    
    # Let's just compute the cost difference by summing affected terms.
    # Affected facilities: idxs
    # Affected locations: p[idxs]
    
    old_cost_part = 0.0
    new_cost_part = 0.0
    
    # This is O(L^2) per check, which is small for L<=4.
    # But we need to check interaction with ALL other facilities.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Terms involving any i in idxs or j in idxs change.
    
    # Efficient way:
    # Delta = New - Old
    # Old terms:
    # 1. Internal to block: sum_{i,j in idxs} a[i,j] b[p[i], p[j]]
    # 2. External: sum_{k not in idxs, i in idxs} a[i,k] b[p[i], p[k]] + sum_{k not in idxs, j in idxs} a[k,j] b[p[k], p[j]]
    
    # Let's compute the contribution of the block to the total cost, and its interaction with the rest.
    
    # To speed up, we can precompute row sums of a and b? 
    # For now, simple calculation for small L.
    
    # Create a mask for indices in the block
    mask = np.zeros(N, dtype=bool)
    mask[idxs] = True
    
    # Old cost involving block:
    # C_old = sum_{i in idxs, j in idxs} a[i,j] b[p[i], p[j]] 
    #       + sum_{i in idxs, j not in idxs} a[i,j] b[p[i], p[j]]
    #       + sum_{i not in idxs, j in idxs} a[i,j] b[p[i], p[j]]
    
    # New permutation for the block:
    p_new_block = np.roll(p[idxs], 1)
    
    # New cost involving block:
    # C_new = sum_{i in idxs, j in idxs} a[i,j] b[p_new_block[i-start], p_new_block[j-start]]
    #       + sum_{i in idxs, j not in idxs} a[i,j] b[p_new_block[i-start], p[j]]
    #       + sum_{i not in idxs, j in idxs} a[i,j] b[p[i], p_new_block[j-start]]
    
    # We can compute these using vectorized ops.
    # Let I = idxs, J = complement
    I = np.array(idxs)
    J = np.setdiff1d(np.arange(N), I)
    
    # Old
    # Internal
    old_int = np.sum(a[np.ix_(I, I)] * b[np.ix_(p[I], p[I])])
    # I to J
    old_ij = np.sum(a[np.ix_(I, J)] * b[np.ix_(p[I], p[J])])
    # J to I
    old_ji = np.sum(a[np.ix_(J, I)] * b[np.ix_(p[J], p[I])])
    
    # New
    p_new_I = p_new_block
    # Internal
    new_int = np.sum(a[np.ix_(I, I)] * b[np.ix_(p_new_I, p_new_I)])
    # I to J
    new_ij = np.sum(a[np.ix_(I, J)] * b[np.ix_(p_new_I, p[J])])
    # J to I
    new_ji = np.sum(a[np.ix_(J, I)] * b[np.ix_(p[J], p_new_I)])
    
    return (new_int + new_ij + new_ji) - (old_int + old_ij + old_ji)

def _error_repair(a, b, p, a_sym, time_limit, max_steps=100):
    """
    Targeted repair: Find the facility with highest marginal cost and try to fix it.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Compute marginal cost for each facility i:
    # m[i] = sum_j a[i,j] b[p[i], p[j]] + sum_j a[j,i] b[p[j], p[i]]
    # This is the contribution of facility i to the total cost (double counted in total, but okay for ranking).
    
    # Vectorized marginal cost calculation
    # C_i = sum_j a[i,j] b[p[i], p[j]] + sum_j a[j,i] b[p[j], p[i]]
    # Note: b is symmetric, so b[p[i], p[j]] = b[p[j], p[i]]
    # C_i = sum_j (a[i,j] + a[j,i]) * b[p[i], p[j]]
    
    a_sym = a + a.T
    # B_p = b[np.ix_(p, p)]
    # marginals = a_sym @ B_p -> row sums?
    # marginals[i] = sum_j a_sym[i,j] * b[p[i], p[j]]
    
    B_p = b[np.ix_(p, p)]
    marginals = a_sym @ B_p
    marginals = marginals.sum(axis=1)
    
    # Find the facility with highest marginal cost
    worst_fac = np.argmax(marginals)
    
    # Try to find a better location for worst_fac
    current_loc = p[worst_fac]
    
    # Calculate cost if we move worst_fac to every other location
    # Delta for moving i from loc_x to loc_y:
    # Remove i from x, add i to y.
    # This is effectively a swap with a "dummy" or just a reassignment.
    # Since it's a permutation, we must swap with some other facility.
    # Strategy: Swap worst_fac with the facility at the location that minimizes the new marginal cost?
    
    # Simpler: Try swapping worst_fac with every other facility j.
    # Pick the j that gives the best delta.
    best_delta = 0
    best_j = -1
    
    for j in range(n):
        if j == worst_fac:
            continue
        delta = _swap_delta(a, b, best_p, worst_fac, j, a_sym)
        if delta < best_delta:
            best_delta = delta
            best_j = j
            
    if best_j != -1:
        best_p[worst_fac], best_p[best_j] = best_p[best_j], best_p[worst_fac]
        best_cost += best_delta
        
    return best_p, best_cost

def _island_worker(a, b, seed, time_limit):
    """
    Worker for one island.
    """
    n = len(a)
    np.random.seed(seed)
    random.seed(seed)
    
    a_sym = a + a.T
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Initial solution: Rank Matching
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Perturb initial solution
    for _ in range(20):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # 1. Block Rotation Search
        improved = False
        for _ in range(50):  # Limit iterations per phase
            if time.time() - start_time > time_limit:
                break
            
            # Randomly choose a block size and start position
            k = random.choice(BLOCK_SIZES)
            if k > n:
                continue
            start = random.randint(0, n - k)
            
            delta = _block_rotation_delta(a, b, best_p, start, k, a_sym)
            
            if delta < 0:
                # Perform rotation
                block = best_p[start:start+k]
                best_p[start:start+k] = np.roll(block, 1)
                best_cost += delta
                improved = True
        
        # 2. Error Repair (Structural change)
        if time.time() - start_time < time_limit:
            repaired_p, repaired_cost = _error_repair(a, b, best_p, a_sym, time_limit, max_steps=10)
            if repaired_cost < best_cost:
                best_p = repaired_p
                best_cost = repaired_cost
        
        # 3. Occasional Random Restart / Perturbation
        if time.time() - start_time < time_limit and random.random() < 0.1:
            # Perturb current best
            p_pert = best_p.copy()
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            # Quick local search on perturbed
            # Just a few swaps
            for _ in range(10):
                i, j = random.sample(range(n), 2)
                delta = _swap_delta(a, b, p_pert, i, j, a_sym)
                if delta < 0:
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                    # Update cost roughly or recompute? 
                    # Recomputing is O(n^2), which is 65k ops. Fast enough for 10 steps.
            
            p_pert_cost = _qap_cost(a, b, p_pert)
            if p_pert_cost < best_cost:
                best_p = p_pert
                best_cost = p_pert_cost
                
    return best_p, best_cost

def _solve_tai256c(a, b, time_limit=TIME_LIMIT):
    n = len(a)
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallelism
    # Note: Passing large arrays to processes might have overhead. 
    # For 256x256 int64, that's ~512KB per matrix. Acceptable.
    
    # If we are in a restricted environment where multiprocessing is slow to start,
    # we might need to fall back to threads (which won't help with GIL for numpy, 
    # but numpy releases GIL for large ops). 
    # However, the worker logic is mostly Python loops for block rotation deltas, 
    # so threads won't parallelize the CPU work. 
    # Let's try ProcessPool. If it fails or is too slow to start, we run sequentially.
    
    # To be safe against environment restrictions, we will run the islands sequentially 
    # if multiprocessing is not available or if we detect high overhead, 
    # but the prompt implies standard library is available. 
    # Let's stick to a robust implementation: 
    # If we use ProcessPool, we must ensure the function is picklable. 
    # Since we are inside a single file, global functions are fine.
    
    # However, 150s wall clock. Process startup can take 1-2s. 
    # With 4 islands, that's okay.
    
    # Alternative: Use Threads if we optimize the inner loop to use Numpy heavily.
    # My _block_rotation_delta uses numpy ops, so it releases GIL.
    # Let's use ThreadPoolExecutor to avoid pickle overhead and potential issues with 
    # multiprocessing in some evaluation sandboxes.
    # Since Numpy releases GIL for matrix operations, threads will actually parallelize 
    # the heavy lifting in _block_rotation_delta and _qap_cost.
    
    import threading
    
    results = [None] * NUM_ISLANDS
    errors = [None] * NUM_ISLANDS
    
    def run_island(idx, seed):
        try:
            results[idx] = _island_worker(a, b, seed, time_limit)
        except Exception as e:
            errors[idx] = e
            
    threads = []
    for i in range(NUM_ISLANDS):
        seed = 42 + i * 100
        t = threading.Thread(target=run_island, args=(i, seed))
        threads.append(t)
        t.start()
        
    # Wait for all threads to finish or timeout
    # We set a time limit inside the worker, so they should finish around time_limit.
    # We need to join them.
    deadline = start_time + time_limit + 5 # Small buffer
    for t in threads:
        remaining = deadline - time.time()
        if remaining > 0:
            t.join(timeout=remaining)
            
    # Collect results
    best_p = None
    best_cost = np.inf
    
    for i in range(NUM_ISLANDS):
        if results[i] is not None:
            p, cost = results[i]
            if cost < best_cost:
                best_cost = cost
                best_p = p
                
    if best_p is None:
        # Fallback: Random
        best_p = np.arange(n)
        best_cost = _qap_cost(a, b, best_p)
        
    # Final Polish: Use the best solution found and do a quick 2-opt sweep
    # This ensures we are at a local optimum for single swaps.
    a_sym = a + a.T
    improved = True
    while improved and time.time() - start_time < time_limit - 2.0:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if time.time() - start_time > time_limit - 2.0:
                    break
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
                    
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
