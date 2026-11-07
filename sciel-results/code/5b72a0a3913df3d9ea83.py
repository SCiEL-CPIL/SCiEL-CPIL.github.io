"""Capacitated facility location solver: cluster-based construction + vectorized local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)

    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    return result


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment. Returns inf if infeasible."""
    n_customers = len(demands)
    n_facilities = len(capacities)

    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]

    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')

    open_cost_total = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_cost_total += open_cost[f]

    assign_cost_total = 0.0
    for i in range(n_customers):
        assign_cost_total += assign_cost[i, result[i]]

    return open_cost_total + assign_cost_total


def _cluster_based_init(demands, capacities, open_cost, assign_cost, rng):
    """
    Cluster-based construction:
    1. Randomly select a subset of facilities to open.
    2. For each open facility, greedily assign customers in order of distance.
    3. Use vectorized operations for speed.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = -np.ones(n_customers, dtype=int)
    load = np.zeros(n_facilities)

    # Shuffle facilities to randomize which ones we try
    fac_order = rng.permutation(n_facilities)

    # For each facility, in random order, try to fill it with cheapest customers
    for f in fac_order:
        if load[f] >= capacities[f]:
            continue
        # Get unassigned customers
        unassigned = np.where(result < 0)[0]
        if len(unassigned) == 0:
            break

        # Sort unassigned customers by cost to facility f
        costs_f = assign_cost[unassigned, f]
        order = np.argsort(costs_f)
        sorted_unassigned = unassigned[order]

        # Assign customers to f as long as capacity allows
        for i in sorted_unassigned:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
            else:
                # Can't fit this one, but maybe smaller ones later? 
                # For simplicity, skip (greedy by cost, not by demand)
                # This might leave some unassigned
                pass

    # Handle any unassigned customers: assign to facility with most remaining capacity
    unassigned = np.where(result < 0)[0]
    for i in unassigned:
        remaining = capacities - load
        f = np.argmax(remaining)
        load[f] += demands[i]
        result[i] = f

    return result


def _local_search_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=50):
    """
    Vectorized local search: try moving each customer to the best feasible facility.
    Uses batch evaluation to reduce Python loop overhead.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)

    best_result = result.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]

    # Precompute: for each customer, the current cost and current facility
    current_facs = best_result.copy()
    current_costs = assign_cost[np.arange(n_customers), current_facs]

    for _pass in range(max_passes):
        any_improved = False

        # For each customer, find the best alternative facility
        # We'll process customers in random order to avoid systematic bias
        for i in np.random.permutation(n_customers):
            cf = current_facs[i]
            di = demands[i]
            cur_cost_i = assign_cost[i, cf]

            # Current facility load
            cf_load = load[cf]

            # Find best alternative
            # For each facility f, the delta is:
            # -cur_cost_i + assign_cost[i,f] + open_cost changes
            # Open cost changes:
            #   - if cf becomes empty (cf_load == di), save open_cost[cf]
            #   - if f was empty (load[f] == 0), pay open_cost[f]
            #
            # Feasibility: load[f] + di <= capacities[f]

            # Vectorized: compute delta for all facilities
            # Start with assignment cost delta
            delta = assign_cost[i, :] - cur_cost_i

            # Feasibility mask
            feasible = (load + di <= capacities)
            feasible[cf] = False  # Don't move to same facility

            if not np.any(feasible):
                continue

            # Open cost adjustments
            # If cf becomes empty
            if cf_load == di:
                # For all f: subtract open_cost[cf]
                delta[feasible] -= open_cost[cf]

            # For facilities that are currently empty: add open_cost[f]
            empty_mask = (load == 0) & feasible
            delta[empty_mask] += open_cost[empty_mask]

            # Find best
            best_idx = np.argmin(np.where(feasible, delta, np.inf))
            best_delta = delta[best_idx]

            if best_delta < -1e-9 and feasible[best_idx]:
                # Perform the move
                bf = best_idx
                load[cf] -= di
                load[bf] += di
                current_facs[i] = bf
                current_costs[i] = assign_cost[i, bf]
                any_improved = True

        if not any_improved:
            break

    best_result = current_facs
    return best_result


def _pair_swap_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=30):
    """
    Pair swap: try swapping customers between different facilities.
    Process in batches for speed.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)

    best_result = result.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]

    for _pass in range(max_passes):
        any_improved = False

        # Group customers by facility
        fac_to_custs = {f: [] for f in range(n_facilities)}
        for i in range(n_customers):
            f = best_result[i]
            fac_to_custs[f].append(i)

        # Try swaps between facilities that have customers
        non_empty_facs = [f for f in range(n_facilities) if len(fac_to_custs[f]) > 0]

        for idx_a, fa in enumerate(non_empty_facs):
            for fb in non_empty_facs[idx_a + 1:]:
                if fa == fb:
                    continue

                custs_a = fac_to_custs[fa]
                custs_b = fac_to_custs[fb]

                # Try all pairs (i in A, j in B)
                # This is O(|A|*|B|) which can be large, but with 100 customers
                # and 20 facilities, average ~5 per facility, so ~25 pairs per pair of facs
                for i in custs_a:
                    di = demands[i]
                    cost_ia = assign_cost[i, fa]
                    cost_ib = assign_cost[i, fb]

                    for j in custs_b:
                        dj = demands[j]
                        cost_ja = assign_cost[j, fa]
                        cost_jb = assign_cost[j, fb]

                        # Feasibility check
                        new_load_fa = load[fa] - di + dj
                        new_load_fb = load[fb] - dj + di

                        if new_load_fa > capacities[fa] + 1e-9:
                            continue
                        if new_load_fb > capacities[fb] + 1e-9:
                            continue

                        # Delta cost
                        delta = -cost_ia - cost_jb + cost_ib + cost_ja

                        # Open cost changes
                        # fa: was load[fa], now new_load_fa
                        # fb: was load[fb], now new_load_fb
                        # Current open: open_cost[fa] if load[fa]>0 else 0, same for fb
                        # Both are non-empty since they have customers
                        # After swap, check if either becomes empty
                        old_open = open_cost[fa] + open_cost[fb]
                        new_open = 0.0
                        if new_load_fa > 0:
                            new_open += open_cost[fa]
                        if new_load_fb > 0:
                            new_open += open_cost[fb]

                        delta += new_open - old_open

                        if delta < -1e-9:
                            # Perform swap
                            load[fa] -= di
                            load[fb] -= dj
                            load[fa] += dj
                            load[fb] += di
                            best_result[i] = fb
                            best_result[j] = fa

                            # Update customer lists
                            fac_to_custs[fa].remove(i)
                            fac_to_custs[fa].append(j)
                            fac_to_custs[fb].remove(j)
                            fac_to_custs[fb].append(i)

                            any_improved = True

        if not any_improved:
            break

    return best_result


def _lncs(result, demands, capacities, open_cost, assign_cost, num_restarts=8):
    """Large Neighborhood Search with randomized restarts."""
    n_customers = len(demands)
    n_facilities = len(capacities)

    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)

    if not np.isfinite(best_cost):
        return best_result

    rng = np.random.RandomState(42)

    for restart in range(num_restarts):
        # Perturb: randomly reassign a subset
        perturbed = best_result.copy()
        subset_size = max(5, n_customers // 8)
        indices = rng.choice(n_customers, size=subset_size, replace=False)

        # Remove these customers from load
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            if i not in indices:
                load[perturbed[i]] += demands[i]

        # Reassign the subset using greedy (cheapest feasible)
        for i in indices:
            order = np.argsort(assign_cost[i])
            assigned = False
            for f in order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    perturbed[i] = f
                    assigned = True
                    break
            if not assigned:
                remaining = capacities - load
                f = np.argmax(remaining)
                load[f] += demands[i]
                perturbed[i] = f

        # Local search on perturbed solution
        improved = _local_search_vectorized(perturbed, demands, capacities, open_cost, assign_cost, max_passes=30)
        cost = _compute_cost(improved, demands, capacities, open_cost, assign_cost)

        if np.isfinite(cost) and cost < best_cost - 1e-9:
            best_result = improved
            best_cost = cost

    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)

    n_customers = len(demands)
    n_facilities = len(capacities)

    rng = np.random.RandomState(12345)

    # Phase 1: Multiple diverse initial solutions, pick best
    candidates = []
    candidates.append(_greedy_assign(demands, capacities, open_cost, assign_cost))
    candidates.append(_cluster_based_init(demands, capacities, open_cost, assign_cost, rng))

    # Try a few more cluster-based inits with different seeds
    for seed in [7, 42, 99]:
        rng2 = np.random.RandomState(seed)
        candidates.append(_cluster_based_init(demands, capacities, open_cost, assign_cost, rng2))

    # Pick best initial candidate
    best_init = None
    best_init_cost = float('inf')
    for cand in candidates:
        c = _compute_cost(cand, demands, capacities, open_cost, assign_cost)
        if np.isfinite(c) and c < best_init_cost:
            best_init_cost = c
            best_init = cand

    result = best_init

    # Phase 2: Local search
    if time.time() < deadline - 30:
        result = _local_search_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=50)

    # Phase 3: Pair swap search
    if time.time() < deadline - 20:
        result = _pair_swap_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=30)

    # Phase 4: More local search
    if time.time() < deadline - 10:
        result = _local_search_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=50)

    # Phase 5: LNS if time remains
    if time.time() < deadline - 5:
        result = _lncs(result, demands, capacities, open_cost, assign_cost, num_restarts=8)

    # Final local search pass
    if time.time() < deadline - 1:
        result = _local_search_vectorized(result, demands, capacities, open_cost, assign_cost, max_passes=20)

    # Ensure validity
    load = np.zeros(n_facilities)
    valid = True
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            valid = False
            break
        load[f] += demands[i]

    if valid:
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                valid = False
                break

    if not valid:
        result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    return result
# EVOLVE-BLOCK-END
