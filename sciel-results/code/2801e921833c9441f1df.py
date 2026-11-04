"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        total = 0
        for i in range(n):
            total += np.dot(a[i, :], b[p[i], :])
        return int(total)

    def greedy_construction(a, b, rng):
        """
        Linearized Greedy Construction.
        Builds permutation by assigning facilities one by one.
        """
        n = len(a)
        # Precompute row sums of A to prioritize high-flow facilities
        a_sums = np.sum(a, axis=1)
        
        p = np.full(n, -1, dtype=np.int64)
        assigned_facilities = np.zeros(n, dtype=bool)
        assigned_locations = np.zeros(n, dtype=bool)
        
        # C[i, j] is the marginal cost of assigning facility i to location j
        # given the current partial assignment.
        C = np.zeros((n, n), dtype=np.float64)
        
        # Order of facilities to assign (heuristic: high flow first)
        # We can also randomize this order slightly for diversity
        order = np.argsort(-a_sums, kind='stable')
        
        for step in range(n):
            if time.time() > deadline - 5:
                break
                
            # Select next facility
            # Pick the first unassigned facility in the order
            i = -1
            for idx in order:
                if not assigned_facilities[idx]:
                    i = idx
                    break
            
            if i == -1:
                break
                
            # Find best location for facility i
            # C[i, :] contains the marginal costs for all locations
            # Only consider unassigned locations
            costs = C[i, :].copy()
            costs[assigned_locations] = np.inf
            
            # Add small random noise to break ties and allow exploration
            # Noise scale can be adjusted
            noise_scale = 1e-3 * np.max(np.abs(C[i, :])) if np.max(np.abs(C[i, :])) > 0 else 1e-3
            costs[~assigned_locations] += rng.uniform(0, noise_scale, size=np.sum(~assigned_locations))
            
            j = np.argmin(costs)
            
            # Assign facility i to location j
            p[i] = j
            assigned_facilities[i] = True
            assigned_locations[j] = True
            
            # Update C for all unassigned facilities k and locations l
            # C[k, l] += a[k, i] * b[l, j] + a[i, k] * b[j, l]
            # Vectorized update
            # a[:, i] is the column of A for facility i
            # b[:, j] is the column of B for location j
            # b[j, :] is the row of B for location j
            # a[i, :] is the row of A for facility i
            
            # We need to update C[k, l] for k not assigned, l not assigned
            # C += outer(a[:, i], b[j, :]) + outer(a[i, :], b[:, j])
            # But we only care about unassigned k and l.
            # Doing the full outer product is O(n^2), which is fine.
            
            # Note: a and b are int64, but C is float64 to avoid overflow during accumulation?
            # Max value: sum of 150 terms. Max a,b ~ 1000? 150 * 1000 * 1000 = 1.5e8. Fits in int32.
            # But let's use float64 for safety and ease of noise addition.
            
            update1 = np.outer(a[:, i], b[j, :])
            update2 = np.outer(a[i, :], b[:, j])
            
            C += update1 + update2
            
        # If we didn't assign all (due to time), fill remaining randomly
        unassigned_facs = np.where(~assigned_facilities)[0]
        unassigned_locs = np.where(~assigned_locations)[0]
        if len(unassigned_facs) > 0:
            # Randomly assign remaining
            rng.shuffle(unassigned_locs)
            for idx, fac in enumerate(unassigned_facs):
                if idx < len(unassigned_locs):
                    p[fac] = unassigned_locs[idx]
                    assigned_locations[unassigned_locs[idx]] = True
                else:
                    # Should not happen if counts match
                    p[fac] = 0 

        return p

    def compute_swap_delta(i, j, p, a, b):
        """
        Compute the change in cost when swapping locations of facilities i and j.
        """
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0

        delta = 0.0

        # Diagonal terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])

        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])

        # Terms with other facilities k (vectorized)
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

        return int(delta)

    def local_search(p, max_time):
        """
        Steepest descent local search with 2-opt swaps.
        """
        cost = compute_cost(p)
        improved = True
        
        while improved and time.time() < max_time:
            improved = False
            best_delta = 0
            best_pair = None

            # Find the best improving swap
            for i in range(n):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() >= max_time:
                        break
                    delta = compute_swap_delta(i, j, p, a, b)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)

            if best_pair is not None:
                i, j = best_pair
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                improved = True
            else:
                break

        return p, cost

    def perturb(p, strength, rng):
        """
        Perturb the permutation by performing 'strength' random swaps.
        """
        p = p.copy()
        for _ in range(strength):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    # Main Logic
    rng = np.random.RandomState(42)
    
    best_p = None
    best_cost = float('inf')
    
    # Phase 1: Generate a few good initial solutions using LGC
    num_initials = 5
    for k in range(num_initials):
        if time.time() > deadline - 10:
            break
        
        # Slightly vary the LGC by changing the facility order or noise
        # For simplicity, we just use the standard LGC but with different RNG seeds for noise
        rng_k = np.random.RandomState(12345 + k)
        p_init = greedy_construction(a, b, rng_k)
        
        # Refine with local search
        p_ls, cost_ls = local_search(p_init, deadline - 5)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # If no solution was found (should not happen), fallback to random
    if best_p is None:
        best_p = rng.permutation(n).astype(np.int64)
        best_cost = compute_cost(best_p)

    # Phase 2: Iterated Local Search (ILS)
    current_p = best_p.copy()
    current_cost = best_cost
    
    while time.time() < deadline - 5:
        # Perturb
        strength = 5 + int(5 * (1 - time.time() / deadline))
        strength = max(3, min(15, strength))
        perturbed_p = perturb(current_p, strength, rng)
        
        # Local search from perturbed solution
        p_ls, cost_ls = local_search(perturbed_p, deadline - 5)
        
        # Accept if better
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
            current_p = p_ls.copy()
            current_cost = cost_ls
        else:
            # Sometimes accept worse solutions to escape local optima (Simulated Annealing style)
            # Simple probability based on improvement gap
            if rng.random() < 0.05:
                current_p = p_ls.copy()
                current_cost = cost_ls

    return best_p.tolist()

# EVOLVE-BLOCK-END
