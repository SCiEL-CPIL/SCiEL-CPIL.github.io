"""
Strategy: Simulated Annealing with Vectorized Swap Deltas and Multi-Start.

This approach fundamentally changes the search paradigm from deterministic local refinement (2-opt)
to stochastic global search (Simulated Annealing).

Key Structural Differences:
1. **Search Method**: Instead of strictly improving 2-opt, we use Simulated Annealing.
   - Accepts worsening moves with probability exp(-delta/T).
   - Uses a cooling schedule (Geometric or Fast) to balance exploration and exploitation.
   - This allows escaping the local optima that trap the current 2-opt based program.

2. **Initialization**: 
   - Replaces heavy PCA/Spectral initialization with a fast "Greedy + Random" multi-start.
   - We generate a few diverse starting points (random, sorted-by-row-sum, random-perturbed)
     and let the SA process find the best basin. This is cheaper to compute and more robust
     to different instance structures than relying on a single spectral embedding.

3. **Speed Optimization**:
   - The swap delta calculation is heavily vectorized using NumPy.
   - Pre-computed row/col sums of B are used to update the cost incrementally in O(n) 
     instead of O(n^2) for full cost recalculation, and O(1) for the delta logic itself 
     if we maintain the necessary aggregates. 
     *Correction*: For QAP, the delta for swapping positions i and j involves interactions 
     with all other k. We compute this in O(n) vectorized operations per swap.
   - Time slicing: The SA loop checks the deadline every 1000 steps to ensure we return 
     the best solution found before the 120s limit.

4. **Diversification**:
   - If the SA stagnates (no improvement for a long time), we perform a "kick" 
     (random perturbation) and restart the cooling schedule from a higher temperature.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _swap_delta_vectorized(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    
    Cost = sum_{u,v} A[u,v] B[p[u], p[v]]
    
    Let u=i, v=j.
    Old terms involving i or j:
    1. A[i,i]B[p[i],p[i]] + A[j,j]B[p[j],p[j]]
    2. A[i,j]B[p[i],p[j]] + A[j,i]B[p[j],p[i]]
    3. sum_{k!=i,j} (A[i,k]B[p[i],p[k]] + A[k,i]B[p[k],p[i]] + A[j,k]B[p[j],p[k]] + A[k,j]B[p[k],p[j]])
    
    New terms (swap p[i] and p[j], let u'=p[j], v'=p[i]):
    1. A[i,i]B[v',v'] + A[j,j]B[u',u']
    2. A[i,j]B[v',u'] + A[j,i]B[u',v']
    3. sum_{k!=i,j} (A[i,k]B[v',p[k]] + A[k,i]B[p[k],v'] + A[j,k]B[u',p[k]] + A[k,j]B[p[k],u'])
    
    Delta = New - Old
    """
    n = len(p)
    u = p[i]
    v = p[j]
    
    # Terms 1 & 2 (Diagonal and Cross)
    old_diag = a[i, i] * b[u, u] + a[j, j] * b[v, v]
    new_diag = a[i, i] * b[v, v] + a[j, j] * b[u, u]
    
    old_cross = a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_cross = a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    # Terms 3 (Interactions with k != i, j)
    # We need sum_{k!=i,j} (A[i,k]B[u,p[k]] + A[k,i]B[p[k],u] + A[j,k]B[v,p[k]] + A[k,j]B[p[k],v])
    # vs
    # sum_{k!=i,j} (A[i,k]B[v,p[k]] + A[k,i]B[p[k],v] + A[j,k]B[u,p[k]] + A[k,j]B[p[k],u])
    
    # Vectorized calculation
    # Create a mask for k != i, j
    # To avoid creating a boolean mask and indexing (which creates a copy), 
    # we can compute the sum over ALL k and subtract the i and j terms.
    # However, A[i,i] and A[j,j] terms are already handled in diag.
    # The cross terms A[i,j] and A[j,i] are handled in cross.
    # So the "k" sum should strictly exclude i and j.
    
    # Let's use the subtraction method for speed, as n=256 is small enough that 
    # full vector operations are fast, but masking is also fine.
    # Actually, simple indexing with boolean mask is O(n) and very fast in NumPy.
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    pk = p[mask]
    ai_k = a[i, mask]
    aj_k = a[j, mask]
    ki_a = a[mask, i]
    kj_a = a[mask, j]
    
    # b[u, pk], b[pk, u], b[v, pk], b[pk, v]
    b_u_pk = b[u, pk]
    b_pk_u = b[pk, u]
    b_v_pk = b[v, pk]
    b_pk_v = b[pk, v]
    
    old_inter = np.sum(ai_k * b_u_pk + ki_a * b_pk_u + aj_k * b_v_pk + kj_a * b_pk_v)
    new_inter = np.sum(ai_k * b_v_pk + ki_a * b_pk_v + aj_k * b_u_pk + kj_a * b_pk_u)
    
    delta = (new_diag + new_cross + new_inter) - (old_diag + old_cross + old_inter)
    return int(delta)

def _simulated_annealing(a, b, p_init, time_limit, start_time, rng):
    """
    Perform Simulated Annealing on the QAP.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Initial temperature
    # Estimate typical delta magnitude. 
    # For random swaps, delta is often in the range of thousands.
    # Let's set T0 high enough to accept most moves initially.
    T0 = 10000.0 
    T_end = 1.0
    # Cooling factor
    alpha = 0.9995
    
    T = T0
    
    # Track stagnation
    last_improve_time = time.time()
    stagnation_threshold = 5.0 # seconds
    
    while time.time() - start_time < time_limit:
        # Pick random i, j
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        delta = _swap_delta_vectorized(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                last_improve_time = time.time()
        else:
            # Worsening, accept with probability exp(-delta/T)
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
        
        # Cooling
        T *= alpha
        
        # Restart if stagnated
        if time.time() - last_improve_time > stagnation_threshold:
            # Perturb current solution
            for _ in range(10):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            # Reset temperature
            T = T0 * 0.5
            last_improve_time = time.time()
            
    return best_p, best_cost

def _greedy_init(a, b, rng):
    """
    Fast greedy initialization.
    1. Sort facilities by row sum (descending).
    2. Sort locations by row sum (descending).
    3. Assign in order.
    4. Apply a few random swaps to diversify.
    """
    n = len(a)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort indices
    fac_order = np.argsort(-a_sums, kind='stable')
    loc_order = np.argsort(-b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    # Facility at index fac_order[k] goes to location loc_order[k]
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
        
    # Randomize slightly to avoid deterministic bias
    for _ in range(20):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            p[i], p[j] = p[j], p[i]
            
    return p

def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: 120s limit. 
    # We need to return well before 150s (evaluator kill) and ideally before 120s (task limit).
    # Let's aim for 115s max.
    total_time_budget = 115.0
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = float('inf')
    
    # Phase 1: Multi-Start Initialization
    # We will run SA on the best of a few initial candidates.
    # Instead of running SA on all, we pick the best initial cost and run SA on that,
    # or run SA in parallel slices? 
    # Given the 120s limit, running one long SA with restarts is safer than many short ones.
    # However, a good start helps.
    
    # Generate 3 diverse starts
    starts = []
    # 1. Greedy
    p_greedy = _greedy_init(a, b, rng)
    starts.append(p_greedy)
    
    # 2. Random
    p_rand = np.random.permutation(n)
    starts.append(p_rand)
    
    # 3. Anti-Greedy (sort one asc, one desc)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    fac_order = np.argsort(a_sums, kind='stable') # Asc
    loc_order = np.argsort(-b_sums, kind='stable') # Desc
    p_anti = np.empty(n, dtype=np.int64)
    for k in range(n):
        p_anti[fac_order[k]] = loc_order[k]
    starts.append(p_anti)
    
    # Evaluate starts
    costs = [_qap_cost(a, b, p) for p in starts]
    best_start_idx = np.argmin(costs)
    best_p = starts[best_start_idx].copy()
    best_cost = costs[best_start_idx]
    
    # Phase 2: Simulated Annealing
    # We allocate the remaining time to SA.
    elapsed = time.time() - start_time
    remaining_time = total_time_budget - elapsed
    
    if remaining_time > 10:
        # Run SA
        p_sa, cost_sa = _simulated_annealing(a, b, best_p, remaining_time - 2, start_time, rng)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
            
    # Phase 3: Final Polish with 2-Opt (Deterministic)
    # SA might leave us in a slightly non-optimal local minimum.
    # A quick 2-opt polish can squeeze out the last few percent.
    elapsed = time.time() - start_time
    remaining_time = total_time_budget - elapsed
    
    if remaining_time > 5:
        # Simple 2-opt local search
        p_polish = best_p.copy()
        current_cost = best_cost
        improved = True
        polish_limit = remaining_time - 2
        
        while improved and time.time() - start_time < polish_limit:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() - start_time > polish_limit:
                        break
                    delta = _swap_delta_vectorized(a, b, p_polish, i, j)
                    if delta < 0:
                        p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p_polish.copy()
                        improved = True
                if time.time() - start_time > polish_limit:
                    break
                    
    return best_p.tolist()
# EVOLVE-BLOCK-END
