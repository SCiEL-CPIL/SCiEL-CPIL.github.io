"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_init(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    order = np.argsort(-demands)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                c = assign_cost[i, f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        if best_f >= 0:
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            best_f = np.argmax(rem)
            load[best_f] += demands[i]
            result[i] = best_f
    return result

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        load[f] += demands[i]
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    total += np.sum(assign_cost[np.arange(len(result)), result])
    return total

def _is_valid(result, demands, capacities):
    n_fac = len(capacities)
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return False
        load[f] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f] + 1e-9:
            return False
    return True

def _pass_moves(result, load, demands_f, capacities, open_cost, assign_cost, n_cust, n_fac):
    """One pass of best-improvement single-customer moves. Returns (improved, cost_delta)."""
    total_delta = 0.0
    improved = False
    # Sort customers by current assignment cost descending to prioritize high-cost relocations
    order = np.argsort(-assign_cost[np.arange(n_cust), result])
    for i in order:
        cur_f = result[i]
        best_new_f = -1
        best_delta = 0.0
        for f in range(n_fac):
            if f == cur_f:
                continue
            if load[f] + demands_f[i] > capacities[f] + 1e-9:
                continue
            delta = assign_cost[i, f] - assign_cost[i, cur_f]
            open_delta = 0.0
            if load[f] <= 1e-9:
                open_delta += open_cost[f]
            if load[cur_f] - demands_f[i] <= 1e-9:
                open_delta -= open_cost[cur_f]
            total_delta_f = delta + open_delta
            if total_delta_f < best_delta - 1e-12:
                best_delta = total_delta_f
                best_new_f = f
        if best_new_f >= 0 and best_delta < -1e-9:
            load[cur_f] -= demands_f[i]
            load[best_new_f] += demands_f[i]
            result[i] = best_new_f
            total_delta += best_delta
            improved = True
    return improved, total_delta

def _pass_swaps(result, load, demands_f, capacities, open_cost, assign_cost, n_cust, n_fac):
    """One pass of best-improvement two-customer swaps. Returns (improved, cost_delta)."""
    total_delta = 0.0
    improved = False
    # For efficiency, only consider swaps where both customers change facilities
    # and the swap reduces cost. We iterate over pairs but limit work.
    # Group customers by facility for quick access
    fac_customers = {}
    for i in range(n_cust):
        f = result[i]
        if f not in fac_customers:
            fac_customers[f] = []
        fac_customers[f].append(i)
    
    # Try swaps between customers at different facilities
    facs = list(fac_customers.keys())
    for fa_idx in range(len(facs)):
        fa = facs[fa_idx]
        for fb_idx in range(fa_idx + 1, len(facs)):
            fb = facs[fb_idx]
            for i in fac_customers[fa]:
                for j in fac_customers[fb]:
                    # Swap i and j: i goes to fb, j goes to fa
                    di = demands_f[i]
                    dj = demands_f[j]
                    # Check capacity after swap
                    # fa loses di, gains dj; fb loses dj, gains di
                    if load[fa] - di + dj > capacities[fa] + 1e-9:
                        continue
                    if load[fb] - dj + di > capacities[fb] + 1e-9:
                        continue
                    # Compute delta
                    delta_assign = (assign_cost[i, fb] + assign_cost[j, fa]) - (assign_cost[i, fa] + assign_cost[j, fb])
                    # Open cost changes: fa and fb are both non-empty before (they have at least i and j)
                    # After swap, fa has dj (if dj>0 it's still non-empty), fb has di (still non-empty)
                    # So no open cost change unless one facility becomes empty (impossible since each had at least one other customer potentially)
                    # Actually fa had at least i, after removing i and adding j, if j was the only other... no, fa still has j now
                    # fb had at least j, after removing j and adding i, fb still has i now
                    # So open cost is unchanged
                    open_delta = 0.0
                    total_delta_f = delta_assign + open_delta
                    if total_delta_f < -1e-9:
                        # Apply swap
                        load[fa] -= di
                        load[fb] -= dj
                        load[fa] += dj
                        load[fb] += di
                        result[i] = fb
                        result[j] = fa
                        total_delta += total_delta_f
                        improved = True
                        # Update groups
                        fac_customers[fa].remove(i)
                        fac_customers[fb].remove(j)
                        fac_customers[fb].append(i)
                        fac_customers[fa].append(j)
                        break
                else:
                    continue
                break
    return improved, total_delta

def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    DEADLINE = 143.0  # safety margin before 150s
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Greedy initialization
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    # Precompute
    demands_f = demands.astype(float)
    capacities_f = capacities.astype(float)
    open_cost_f = np.asarray(open_cost, dtype=float)
    
    # Build load array
    load = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load[result[i]] += demands_f[i]
    
    current_cost = best_cost
    
    # Multi-pass local search: alternate move and swap passes until no improvement or time runs out
    max_passes = 500
    for p in range(max_passes):
        if time.time() - t0 > DEADLINE:
            break
        # Move pass
        improved, delta = _pass_moves(result, load, demands_f, capacities_f, open_cost_f, assign_cost, n_cust, n_fac)
        if improved:
            current_cost += delta
            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_result = result.copy()
        # Swap pass
        improved_s, delta_s = _pass_swaps(result, load, demands_f, capacities_f, open_cost_f, assign_cost, n_cust, n_fac)
        if improved_s:
            current_cost += delta_s
            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_result = result.copy()
        if not improved and not improved_s:
            break
    
    # Final single-pass move improvement on best_result
    load2 = np.zeros(n_fac, dtype=float)
    for i in range(n_cust):
        load2[best_result[i]] += demands_f[i]
    improved2 = True
    while improved2 and time.time() - t0 < DEADLINE:
        improved2, _ = _pass_moves(best_result, load2, demands_f, capacities_f, open_cost_f, assign_cost, n_cust, n_fac)
    
    # Ensure validity
    if not _is_valid(best_result, demands, capacities):
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return best_result
# EVOLVE-BLOCK-END
