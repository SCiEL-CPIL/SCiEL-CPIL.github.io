# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin
    
    # Precompute sorted facility indices by assignment cost for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
    def _compute_obj(result):
        """Compute objective for a valid assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    def _greedy(order):
        """Greedy assign in given order, cheapest feasible facility first."""
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                return None
        return result
    
    # Try multiple orderings
    best_result = None
    best_cost = float('inf')
    
    # Ordering 1: largest demand first
    order1 = np.argsort(-demands)
    r = _greedy(order1)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Ordering 2: smallest demand first
    order2 = np.argsort(demands)
    r = _greedy(order2)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Ordering 3: random
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline:
            break
        order = rng.permutation(n_cust)
        r = _greedy(order)
        if r is not None:
            c = _compute_obj(r)
            if c < best_cost:
                best_cost = c
                best_result = r.copy()
    
    if best_result is None:
        # Fallback: round-robin
        result = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        valid = True
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities):
            # Just return something; will be invalid but we need to return something
            return result
        return result
    
    # Local search: swap customers between facilities
    def _local_search(result, max_iters=500):
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        # Track which facilities are open
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        current_obj = _compute_obj(result)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                cur_f = result[i]
                # Try moving to another facility
                for f in fac_order[i]:
                    if f == cur_f:
                        continue
                    # Check capacity
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    # Compute change in objective
                    # Remove i from cur_f
                    new_load_cur = load[cur_f] - demands[i]
                    # Add i to f
                    new_load_f = load[f] + demands[i]
                    
                    # Change in assignment cost
                    delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                    
                    # Change in open cost
                    delta_open = 0.0
                    if new_load_cur == 0 and load[cur_f] > 0:
                        # cur_f becomes closed
                        delta_open -= open_cost[cur_f]
                    if new_load_f == 0 and load[f] == 0:
                        # f becomes open
                        delta_open += open_cost[f]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Apply the move
                        load[cur_f] = new_load_cur
                        load[f] = new_load_f
                        result[i] = f
                        if new_load_cur == 0:
                            open_mask[cur_f] = False
                        if new_load_f > 0:
                            open_mask[f] = True
                        current_obj += total_delta
                        improved = True
                        break
        
        return result
    
    best_result = _local_search(best_result, max_iters=800)
    best_cost = _compute_obj(best_result)
    
    # Try swapping pairs of customers
    def _pair_swap_search(result, max_rounds=100):
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        current_obj = _compute_obj(result)
        rng_swap = np.random.RandomState(99)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            # Pick two random customers
            i, j = rng_swap.choice(n_cust, size=2, replace=False)
            fi, fj = result[i], result[j]
            if fi == fj:
                continue
            # Check if swap is feasible
            # After swap: i->fj, j->fi
            # Load change at fi: -demands[i] + demands[j]
            # Load change at fj: -demands[j] + demands[i]
            new_load_fi = load[fi] - demands[i] + demands[j]
            new_load_fj = load[fj] - demands[j] + demands[i]
            if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                continue
            
            # Change in assignment cost
            delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
            
            # Change in open cost (only if a facility goes from 0 to >0 or vice versa)
            delta_open = 0.0
            if new_load_fi == 0 and load[fi] > 0:
                delta_open -= open_cost[fi]
            if new_load_fi > 0 and load[fi] == 0:
                delta_open += open_cost[fi]
            if new_load_fj == 0 and load[fj] > 0:
                delta_open -= open_cost[fj]
            if new_load_fj > 0 and load[fj] == 0:
                delta_open += open_cost[fj]
            
            total_delta = delta_assign + delta_open
            if total_delta < -1e-9:
                load[fi] = new_load_fi
                load[fj] = new_load_fj
                result[i] = fj
                result[j] = fi
                current_obj += total_delta
        
        return result
    
    best_result = _pair_swap_search(best_result, max_rounds=200)
    best_cost = _compute_obj(best_result)
    
    # Another round of local search
    if time.time() < deadline - 5:
        best_result = _local_search(best_result, max_iters=500)
        best_cost = _compute_obj(best_result)
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
