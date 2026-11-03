"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    """
    GRASP + Vectorized 2-swap local search for QAP.
    Combines greedy randomized construction with fast delta computation.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def compute_cost(p):
        """Compute the QAP cost for a given permutation p."""
        return np.sum(a * b[np.ix_(p, p)])

    def compute_swap_delta(p, i, j):
        """
        Compute the change in cost if we swap facilities i and j in permutation p.
        That is, if we swap p[i] and p[j].
        
        Let p' be p with p[i] and p[j] swapped.
        delta = cost(p') - cost(p)
        
        cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
        
        Only terms involving index i or j change.
        """
        li = p[i]
        lj = p[j]
        
        # Self terms
        delta = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms (i,j) and (j,i)
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with k != i, j
        # We need sums over all k except i and j.
        
        # Vectorized computation of the four sum terms:
        # S1 = sum_{k!=i,j} a[k,i] * (b[p[k],lj] - b[p[k],li])
        # S2 = sum_{k!=i,j} a[i,k] * (b[lj,p[k]] - b[li,p[k]])
        # S3 = sum_{k!=i,j} a[k,j] * (b[p[k],li] - b[p[k],lj])
        # S4 = sum_{k!=i,j} a[j,k] * (b[li,p[k]] - b[lj,p[k]])
        
        # Compute full sums first:
        S1_full = np.dot(a[:, i], b[p, lj]) - np.dot(a[:, i], b[p, li])
        S2_full = np.dot(a[i, :], b[lj, p]) - np.dot(a[i, :], b[li, p])
        S3_full = np.dot(a[:, j], b[p, li]) - np.dot(a[:, j], b[p, lj])
        S4_full = np.dot(a[j, :], b[li, p]) - np.dot(a[j, :], b[lj, p])
        
        # Subtract the k=i and k=j contributions
        S1_sub = a[i, i] * (b[li, lj] - b[li, li]) + a[j, i] * (b[lj, li] - b[lj, lj])
        S2_sub = a[i, i] * (b[lj, li] - b[li, li]) + a[i, j] * (b[lj, lj] - b[li, lj])
        S3_sub = a[i, j] * (b[li, li] - b[li, lj]) + a[j, j] * (b[lj, li] - b[lj, lj])
        S4_sub = a[j, i] * (b[li, li] - b[lj, li]) + a[j, j] * (b[li, lj] - b[lj, lj])
        
        delta += (S1_full - S1_sub) + (S2_full - S2_sub) + (S3_full - S3_sub) + (S4_full - S4_sub)
        
        return delta

    def local_search(p, current_cost, deadline):
        """First-improvement 2-swap local search."""
        p = p.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = compute_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, current_cost

    def grasp_construct(alpha=0.3):
        """
        Greedy Randomized construction.
        Assign facilities one by one to locations.
        """
        unassigned_facilities = list(range(n))
        unassigned_locations = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        placed = []
        
        # Precompute row sums of a for potential use
        a_row_sums = a.sum(axis=1)
        
        while unassigned_facilities:
            if time.time() >= deadline:
                # Fill remaining randomly
                remaining_locs = unassigned_locations[:]
                random.shuffle(remaining_locs)
                for idx, fac in enumerate(unassigned_facilities):
                    p[fac] = remaining_locs[idx]
                break
                
            # Choose next facility to place
            # Weight by "heaviness" (sum of flows)
            weights = np.array([a_row_sums[fac] for fac in unassigned_facilities])
            weights = weights - weights.min() + 1.0  # Ensure positive
            weights = weights ** (1.0 / alpha) if alpha > 0 else weights
            probs = weights / weights.sum()
            
            idx = np.random.choice(len(unassigned_facilities), p=probs)
            fac = unassigned_facilities.pop(idx)
            
            # Compute cost of placing fac at each unassigned location
            costs = np.zeros(len(unassigned_locations))
            for l_idx, loc in enumerate(unassigned_locations):
                # Immediate cost: a[fac, fac]*b[loc, loc] + sum_{j in placed} (a[fac,j]*b[loc, p[j]] + a[j,fac]*b[p[j], loc])
                cost = a[fac, fac] * b[loc, loc]
                for j in placed:
                    cost += a[fac, j] * b[loc, p[j]] + a[j, fac] * b[p[j], loc]
                costs[l_idx] = cost
            
            # Select location from candidates
            # Sort by cost
            sorted_indices = np.argsort(costs)
            sorted_costs = costs[sorted_indices]
            
            # Select from top candidates with some randomness
            # Use a threshold based on alpha
            min_cost = sorted_costs[0]
            max_cost = sorted_costs[min(len(sorted_costs) - 1, int(alpha * len(sorted_costs)))]
            
            # Filter candidates within [min_cost, max_cost]
            candidate_mask = sorted_costs <= max_cost
            candidate_indices = sorted_indices[candidate_mask]
            
            # Choose randomly from candidates
            chosen_idx = np.random.choice(candidate_indices)
            chosen_loc = unassigned_locations[chosen_idx]
            
            p[fac] = chosen_loc
            placed.append(fac)
            unassigned_locations.pop(chosen_idx)
        
        return p

    def rank_matching_init():
        """Rank-matching initialization."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # Multiple restarts
    best_p = None
    best_cost = np.inf
    
    num_restarts = 8
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Determine time allocation for local search
        ls_time = min(time_per_restart * 0.7, remaining - 2.0)
        if ls_time < 0.5:
            break
        
        t_limit = time.time() + ls_time
        
        # Choose initialization method
        if restart == 0:
            p = rank_matching_init()
        elif restart == 1:
            p = np.random.permutation(n).astype(np.int64)
        elif restart == 2:
            # Inverse rank matching
            heavy = np.argsort(a.sum(axis=1), kind="stable")
            central = np.argsort(-b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif restart < 5:
            # GRASP with varying alpha
            alpha = 0.1 + 0.1 * (restart - 3)
            p = grasp_construct(alpha=alpha)
        else:
            # Random with perturbation
            p = np.random.permutation(n).astype(np.int64)
            for _ in range(20):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        
        # Run local search
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, t_limit)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Use remaining time for additional local search on best solution
        if time.time() < deadline - 1.0:
            p, current_cost = local_search(best_p.copy(), best_cost, deadline - 0.5)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()

    if best_p is None:
        # Fallback
        best_p = np.random.permutation(n).astype(np.int64)

    return best_p.tolist()
# EVOLVE-BLOCK-END
