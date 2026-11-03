"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    """
    GRASP for QAP: Greedy Randomized construction + vectorized 2-swap local search.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute symmetric versions for cost computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For symmetric a and b (QAPLIB instances are typically symmetric), 
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Note: a and b may not be symmetric in general, but QAPLIB tho150 is symmetric.
    # We'll handle the general case.

    def compute_cost(p):
        """Compute the QAP cost for a given permutation p."""
        # p[i] = location assigned to facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as: for each i, sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] @ b[:, p[i]])[p[i]] ... no
        # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Let's compute this efficiently.
        # Create matrix B_p where B_p[i,j] = b[p[i], p[j]]
        # Then cost = sum_{i,j} a[i,j] * B_p[i,j] = np.sum(a * B_p)
        # B_p = b[np.ix_(p, p)]
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    def compute_swap_delta(p, i, j):
        """
        Compute the change in cost if we swap facilities i and j in permutation p.
        That is, if we swap p[i] and p[j].
        
        Let p' be p with p[i] and p[j] swapped.
        delta = cost(p') - cost(p)
        
        cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
        
        Only terms involving index i or j change.
        Terms that change:
        - (i, i): a[i,i] * b[p[i], p[i]] -> a[i,i] * b[p[j], p[j]]
        - (j, j): a[j,j] * b[p[j], p[j]] -> a[j,j] * b[p[i], p[i]]
        - (i, j): a[i,j] * b[p[i], p[j]] -> a[i,j] * b[p[j], p[i]]
        - (j, i): a[j,i] * b[p[j], p[i]] -> a[j,i] * b[p[i], p[j]]
        - (k, i) for k != i,j: a[k,i] * b[p[k], p[i]] -> a[k,i] * b[p[k], p[j]]
        - (i, k) for k != i,j: a[i,k] * b[p[i], p[k]] -> a[i,k] * b[p[j], p[k]]
        - (k, j) for k != i,j: a[k,j] * b[p[k], p[j]] -> a[k,j] * b[p[k], p[i]]
        - (j, k) for k != i,j: a[j,k] * b[p[j], p[k]] -> a[j,k] * b[p[i], p[k]]
        
        Let's define:
        li = p[i], lj = p[j]
        
        delta = a[i,i]*(b[lj,lj] - b[li,li]) 
              + a[j,j]*(b[li,li] - b[lj,lj])
              + a[i,j]*(b[lj,li] - b[li,lj])
              + a[j,i]*(b[li,lj] - b[lj,li])
              + sum_{k != i,j} a[k,i]*(b[p[k],lj] - b[p[k],li])
              + sum_{k != i,j} a[i,k]*(b[lj,p[k]] - b[li,p[k]])
              + sum_{k != i,j} a[k,j]*(b[p[k],li] - b[p[k],lj])
              + sum_{k != i,j} a[j,k]*(b[li,p[k]] - b[lj,p[k]])
        
        We can vectorize the sums over k.
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
        # Let's create a mask or just compute over all k and subtract i and j contributions.
        
        # sum_{k != i,j} a[k,i] * (b[p[k], lj] - b[p[k], li])
        # = sum_{k} a[k,i] * (b[p[k], lj] - b[p[k], li]) - a[i,i]*(b[li,lj]-b[li,li]) - a[j,i]*(b[lj,li]-b[lj,lj])
        
        # Let's compute the full sum and subtract.
        # For the term: sum_k a[k,i] * b[p[k], lj]
        # This is (a[:,i] * b[p, lj]).sum() where p is the permutation
        # Actually: sum_k a[k,i] * b[p[k], lj] = np.dot(a[:, i], b[p, lj])
        # Similarly for others.
        
        # Let's precompute b_p = b[p, :] for efficiency? No, p changes.
        # Let's just compute directly.
        
        # Vectorized computation of the four sum terms:
        # S1 = sum_{k!=i,j} a[k,i] * (b[p[k],lj] - b[p[k],li])
        # S2 = sum_{k!=i,j} a[i,k] * (b[lj,p[k]] - b[li,p[k]])
        # S3 = sum_{k!=i,j} a[k,j] * (b[p[k],li] - b[p[k],lj])
        # S4 = sum_{k!=i,j} a[j,k] * (b[li,p[k]] - b[lj,p[k]])
        
        # Compute full sums first:
        # S1_full = np.dot(a[:, i], b[p, lj]) - np.dot(a[:, i], b[p, li])
        # S2_full = np.dot(a[i, :], b[lj, p]) - np.dot(a[i, :], b[li, p])  # Note: b[lj, p] means b[lj, p[k]]
        # S3_full = np.dot(a[:, j], b[p, li]) - np.dot(a[:, j], b[p, lj])
        # S4_full = np.dot(a[j, :], b[li, p]) - np.dot(a[j, :], b[lj, p])
        
        # Then subtract the k=i and k=j contributions.
        
        # For S1: k=i gives a[i,i]*(b[li,lj]-b[li,li]), k=j gives a[j,i]*(b[lj,li]-b[lj,lj])
        # For S2: k=i gives a[i,i]*(b[lj,li]-b[li,li]), k=j gives a[i,j]*(b[lj,lj]-b[li,lj])
        # For S3: k=i gives a[i,j]*(b[li,li]-b[li,lj]), k=j gives a[j,j]*(b[lj,li]-b[lj,lj])
        # For S4: k=i gives a[j,i]*(b[li,li]-b[lj,li]), k=j gives a[j,j]*(b[li,lj]-b[lj,lj])
        
        S1_full = np.dot(a[:, i], b[p, lj]) - np.dot(a[:, i], b[p, li])
        S2_full = np.dot(a[i, :], b[lj, p]) - np.dot(a[i, :], b[li, p])
        S3_full = np.dot(a[:, j], b[p, li]) - np.dot(a[:, j], b[p, lj])
        S4_full = np.dot(a[j, :], b[li, p]) - np.dot(a[j, :], b[lj, p])
        
        S1_sub = a[i, i] * (b[li, lj] - b[li, li]) + a[j, i] * (b[lj, li] - b[lj, lj])
        S2_sub = a[i, i] * (b[lj, li] - b[li, li]) + a[i, j] * (b[lj, lj] - b[li, lj])
        S3_sub = a[i, j] * (b[li, li] - b[li, lj]) + a[j, j] * (b[lj, li] - b[lj, lj])
        S4_sub = a[j, i] * (b[li, li] - b[lj, li]) + a[j, j] * (b[li, lj] - b[lj, lj])
        
        delta += (S1_full - S1_sub) + (S2_full - S2_sub) + (S3_full - S3_sub) + (S4_full - S4_sub)
        
        return delta

    def local_search(p, current_cost, deadline):
        """Vectorized 2-swap local search."""
        p = p.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all pairs (i, j) with i < j
            # For efficiency, we can sample or do full passes.
            # For n=150, there are ~11175 pairs. Let's do full passes.
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    delta = compute_swap_delta(p, i, j)
                    if delta < 0:
                        # Accept the swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    def grasp_construct(alpha=0.3):
        """
        Greedy Randomized construction.
        Assign facilities one by one to locations.
        At each step, compute the immediate cost contribution of assigning facility i to location l.
        The immediate cost when placing facility i at location l, given already placed facilities,
        is: sum_{j placed} (a[i,j] + a[j,i]) * b[l, p[j]] + a[i,i] * b[l,l]
        (assuming we consider the contribution to the total cost from terms involving i)
        
        Actually, the total cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
        When we place facility i at location l, the new terms added are:
        - (i, i): a[i,i] * b[l,l]
        - (i, j) for each already placed j: a[i,j] * b[l, p[j]]
        - (j, i) for each already placed j: a[j,i] * b[p[j], l]
        
        So immediate cost = a[i,i]*b[l,l] + sum_{j in placed} (a[i,j]*b[l,p[j]] + a[j,i]*b[p[j],l])
        
        We maintain a list of unassigned facilities and unassigned locations.
        """
        n_facilities = list(range(n))
        n_locations = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        placed = []
        
        # Precompute for speed
        random.shuffle(n_facilities)
        
        for step in range(n):
            if time.time() >= deadline:
                break
            # Choose next facility to assign
            # We can choose randomly from remaining, or use a greedy criterion.
            # For GRASP, let's pick the facility with the highest "urgency" or just random.
            # Let's pick randomly for diversity.
            idx = random.randint(0, len(n_facilities) - 1)
            i = n_facilities.pop(idx)
            
            # Compute cost of assigning i to each remaining location
            costs = np.zeros(len(n_locations), dtype=np.float64)
            for loc_idx, l in enumerate(n_locations):
                cost = a[i, i] * b[l, l]
                if placed:
                    # Vectorized sum over placed facilities
                    # placed is a list of facility indices
                    # p[placed] gives their locations
                    # We need sum_{j in placed} (a[i,j]*b[l,p[j]] + a[j,i]*b[p[j],l])
                    # This is: a[i, placed] dot b[l, p[placed]] + a[placed, i] dot b[p[placed], l]
                    placed_arr = np.array(placed)
                    p_placed = p[placed_arr]
                    cost += np.dot(a[i, placed_arr], b[l, p_placed])
                    cost += np.dot(a[placed_arr, i], b[p_placed, l])
                costs[loc_idx] = cost
            
            # Restricted candidate list: select locations with cost in [min, min + alpha*(max-min)]
            min_cost = np.min(costs)
            max_cost = np.max(costs)
            threshold = min_cost + alpha * (max_cost - min_cost)
            candidates = [loc_idx for loc_idx in range(len(n_locations)) if costs[loc_idx] <= threshold]
            
            # Randomly select from candidates
            chosen_loc_idx = random.choice(candidates)
            l = n_locations.pop(chosen_loc_idx)
            p[i] = l
            placed.append(i)
        
        return p

    # Main GRASP loop
    best_p = None
    best_cost = np.inf
    
    # First, try the rank-matching heuristic as a starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    cost_init = compute_cost(p_init)
    
    if cost_init < best_cost:
        best_cost = cost_init
        best_p = p_init.copy()
    
    # Now do local search on the initial solution
    p_ls, cost_ls = local_search(p_init, cost_init, deadline)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # GRASP iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Construct a new solution using GRASP
        p_grasp = grasp_construct(alpha=0.2)
        
        # Compute cost
        cost_grasp = compute_cost(p_grasp)
        
        # Local search
        p_ls, cost_ls = local_search(p_grasp, cost_grasp, deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # If we're running out of time, break
        if time.time() >= deadline - 5:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
