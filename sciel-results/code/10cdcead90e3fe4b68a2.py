"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        """Compute exact QAP cost."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        delta = 0
        
        # Diagonal terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with other facilities k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return delta

    def tabu_search(p_init, max_iter=5000, tabu_tenure=10, sample_size=200, seed=42):
        """
        Perform Tabu Search with randomized neighborhood sampling.
        """
        rng = np.random.RandomState(seed)
        p = p_init.copy()
        curr_cost = compute_cost(p)
        best_p = p.copy()
        best_cost = curr_cost
        
        # Tabu list: store (i, j, expiry_iteration)
        tabu_list = {}
        # We use a dictionary or list. For speed, a list of tuples (i, j, expiry) 
        # and filtering is O(len(list)). Since tenure is small, list is fine.
        # Or better, a 2D array of expiry times? N=150, 150x150 = 22500 entries. 
        # Let's use a 2D array of ints, 0 means not tabu.
        tabu_expiry = np.zeros((n, n), dtype=int)
        
        no_improve_count = 0
        stagnation_limit = 500
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Sample random pairs
            # To ensure we cover the neighborhood reasonably well, we sample sample_size pairs
            # We avoid sampling (i, i)
            indices = rng.choice(n * (n - 1), sample_size, replace=False)
            # Map index to (i, j)
            # i = idx // (n-1), j = idx % (n-1)
            # if j >= i, j += 1
            # This mapping is tricky. Easier:
            i_indices, j_indices = rng.choice(n, size=(2, sample_size), replace=False)
            # Ensure i != j
            # rng.choice with replace=False on 2D array doesn't guarantee i!=j across rows if not careful?
            # Actually, rng.choice(n, size=(2, sample_size), replace=False) picks 2*sample_size unique numbers?
            # No, it picks from the population n, with replace=False, so all 2*sample_size are unique.
            # So i_indices[k] != j_indices[k] is not guaranteed? 
            # Wait, if I pick 400 unique numbers from 150, that's impossible if sample_size=200.
            # n=150. 2*200 = 400 > 150. So replace=False will fail or return fewer?
            # It will raise an error if sample size > population size with replace=False.
            # So we must use replace=True and then filter, or use a different sampling strategy.
            
            # Better sampling:
            # Generate sample_size unique pairs (i, j) with i != j.
            # We can generate i and j independently with replacement=True, then filter i!=j.
            # Or use a while loop to generate valid pairs.
            
            valid_pairs = []
            while len(valid_pairs) < sample_size:
                i_val, j_val = rng.choice(n, 2, replace=False)
                valid_pairs.append((i_val, j_val))
            
            best_delta = np.inf
            best_i, best_j = -1, -1
            best_is_tabu = False
            best_tabu_delta = np.inf
            best_tabu_i, best_tabu_j = -1, -1
            
            for i_val, j_val in valid_pairs:
                delta = compute_swap_delta(i_val, j_val, p)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i_val, j_val
                    best_is_tabu = (tabu_expiry[i_val, j_val] > it) or (tabu_expiry[j_val, i_val] > it)
                
                # Check for best tabu move (for aspiration)
                if (tabu_expiry[i_val, j_val] > it) or (tabu_expiry[j_val, i_val] > it):
                    if delta < best_tabu_delta:
                        best_tabu_delta = delta
                        best_tabu_i, best_tabu_j = i_val, j_val
            
            # Aspiration criterion: if the best tabu move improves the global best, accept it.
            if best_is_tabu:
                if best_delta < best_cost:
                    # Accept tabu move
                    i_val, j_val = best_i, best_j
                    delta = best_delta
                else:
                    # Pick best non-tabu move
                    # We need to find the best non-tabu move from the samples.
                    # Our logic above only tracked the absolute best.
                    # Let's re-evaluate or track best non-tabu separately.
                    # For simplicity in this loop, if best is tabu and not aspiration, 
                    # we might have to pick the next best. 
                    # To avoid complex tracking, we can just skip the move if it's tabu and not aspiration?
                    # No, that wastes the iteration.
                    # Let's modify the loop to track best_non_tabu as well.
                    
                    pass # Logic needs refinement. Let's restart the selection logic inside the loop properly.
            
            # --- Refined Selection Logic ---
            best_non_tabu_delta = np.inf
            best_non_tabu_i, best_non_tabu_j = -1, -1
            best_tabu_asp_delta = np.inf
            best_tabu_asp_i, best_tabu_asp_j = -1, -1
            
            for i_val, j_val in valid_pairs:
                delta = compute_swap_delta(i_val, j_val, p)
                is_tabu = (tabu_expiry[i_val, j_val] > it) or (tabu_expiry[j_val, i_val] > it)
                
                if is_tabu:
                    if delta < best_tabu_asp_delta:
                        best_tabu_asp_delta = delta
                        best_tabu_asp_i, best_tabu_asp_j = i_val, j_val
                else:
                    if delta < best_non_tabu_delta:
                        best_non_tabu_delta = delta
                        best_non_tabu_i, best_non_tabu_j = i_val, j_val
            
            # Decide which move to take
            if best_non_tabu_delta < best_cost and best_non_tabu_delta < best_tabu_asp_delta:
                # Take best non-tabu if it's good, or if it's better than tabu aspiration
                # Actually, standard TS: take best non-tabu. If best non-tabu is worse than best tabu (which is aspiration), take tabu.
                if best_tabu_asp_delta < best_cost and best_tabu_asp_delta < best_non_tabu_delta:
                    i_val, j_val = best_tabu_asp_i, best_tabu_asp_j
                    delta = best_tabu_asp_delta
                else:
                    i_val, j_val = best_non_tabu_i, best_non_tabu_j
                    delta = best_non_tabu_delta
            else:
                if best_tabu_asp_delta < best_cost:
                    i_val, j_val = best_tabu_asp_i, best_tabu_asp_j
                    delta = best_tabu_asp_delta
                else:
                    i_val, j_val = best_non_tabu_i, best_non_tabu_j
                    delta = best_non_tabu_delta
            
            if i_val == -1:
                break
            
            # Apply move
            p[i_val], p[j_val] = p[j_val], p[i_val]
            curr_cost += delta
            
            # Update Tabu List
            tabu_expiry[i_val, j_val] = it + tabu_tenure
            tabu_expiry[j_val, i_val] = it + tabu_tenure
            
            # Update Best
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Diversification / Reheating
            if no_improve_count > stagnation_limit:
                # Perturb current solution: do 5 random swaps
                for _ in range(5):
                    i_val, j_val = rng.choice(n, 2, replace=False)
                    p[i_val], p[j_val] = p[j_val], p[i_val]
                curr_cost = compute_cost(p)
                no_improve_count = 0
                # Reset tabu list? Or keep it? Keep it to avoid immediate backtracking.
        
        return best_p, best_cost

    # 1. Initializations
    best_p = None
    best_cost = np.inf
    
    # Initialization 1: Hungarian on a @ b.T
    C1 = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C1)
    p1 = np.empty(n, dtype=np.int64)
    p1[row_ind] = col_ind
    cost1 = compute_cost(p1)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Initialization 2: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central
    cost2 = compute_cost(p2)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()
    
    # Initialization 3: Random
    rng_init = np.random.RandomState(123)
    p3 = rng_init.permutation(n)
    cost3 = compute_cost(p3)
    if cost3 < best_cost:
        best_cost = cost3
        best_p = p3.copy()

    # 2. Tabu Search from best initialization
    # Run multiple times with different seeds to diversify
    num_ts_runs = 3
    for run in range(num_ts_runs):
        if time.time() > deadline - 10.0:
            break
        seed = 42 + run
        p_ts, cost_ts = tabu_search(best_p, max_iter=2000, tabu_tenure=10, sample_size=150, seed=seed)
        if cost_ts < best_cost:
            best_cost = cost_ts
            best_p = p_ts.copy()
        
        # Also try from a perturbed version of best_p
        if time.time() > deadline - 10.0:
            break
        rng_pert = np.random.RandomState(1000 + run)
        p_pert = best_p.copy()
        for _ in range(10):
            i_val, j_val = rng_pert.choice(n, 2, replace=False)
            p_pert[i_val], p_pert[j_val] = p_pert[j_val], p_pert[i_val]
        p_ts2, cost_ts2 = tabu_search(p_pert, max_iter=1000, tabu_tenure=10, sample_size=150, seed=seed+100)
        if cost_ts2 < best_cost:
            best_cost = cost_ts2
            best_p = p_ts2.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
