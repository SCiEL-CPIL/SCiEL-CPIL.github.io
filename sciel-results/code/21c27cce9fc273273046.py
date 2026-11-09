"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from random import Random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if S is sum-free."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def greedy_init(elements, order_seed=0):
        """Greedy initialization using conflict-degree heuristic."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Precompute conflict degree: for each element, how many other elements
        # would it conflict with (i.e., pairs (a,b) where a+b=elements[i] or
        # elements[i]+a is in elements or 2*elements[i] is in elements)
        # Simpler: count how many j satisfy elements[i]+elements[j] in elements
        # or elements[j]-elements[i] in elements (meaning elements[i]+something=elements[j])
        
        conflict_degree = [0] * n
        for i in range(n):
            vi = elements[i]
            for j in range(n):
                if i == j:
                    continue
                vj = elements[j]
                # Check if vi + vj is in elements
                if (vi + vj) in val_to_idx:
                    conflict_degree[i] += 1
                # Check if vj - vi is in elements (means vi + (vj-vi) = vj)
                if (vj - vi) in val_to_idx:
                    conflict_degree[i] += 1
            # Check 2*vi
            if (2 * vi) in val_to_idx:
                conflict_degree[i] += 1
        
        # Sort by conflict degree (ascending - least conflicting first)
        order = sorted(range(n), key=lambda i: conflict_degree[i])
        
        included = set()
        included_vals = set()
        for idx in order:
            c_val = elements[idx]
            ok = True
            # Check 1: is there a, b in included with a+b = c_val?
            for a_val in included_vals:
                b_val = c_val - a_val
                if b_val in included_vals:
                    ok = False
                    break
            if not ok:
                continue
            # Check 2: is there a in included with a + c_val in included?
            for a_val in included_vals:
                if (a_val + c_val) in included_vals:
                    ok = False
                    break
            if not ok:
                continue
            # Check 3: is 2*c_val in included?
            if (2 * c_val) in included_vals:
                ok = False
                continue
            if ok:
                included.add(idx)
                included_vals.add(c_val)
        
        return [elements[i] for i in included]

    def local_search(elements, initial_subset, time_limit=30.0):
        """Local search with simulated annealing to escape local optima."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Convert initial_subset to set of indices
        included = set()
        for v in initial_subset:
            if v in val_to_idx:
                included.add(val_to_idx[v])
        
        included_vals = set(elements[i] for i in included)
        
        best_size = len(included)
        best_included = set(included)
        best_included_vals = set(included_vals)
        
        start_time = time.time()
        rng = random.Random(42)
        
        # Precompute for each idx:
        # plus[idx] = set of j such that elements[idx] + elements[j] is in elements
        # This helps with quick checks
        plus = [set() for _ in range(n)]
        for i in range(n):
            vi = elements[i]
            for j in range(n):
                s = vi + elements[j]
                if s in val_to_idx:
                    plus[i].add(j)
        
        def can_add(idx, cur_included_vals):
            """Check if idx can be added to set with values cur_included_vals. O(|S|) but with set lookups."""
            c_val = elements[idx]
            # Check 1: a + b = c_val for a, b in cur_included_vals
            for a_val in cur_included_vals:
                b_val = c_val - a_val
                if b_val in cur_included_vals:
                    return False
            # Check 2: a + c_val in cur_included_vals for a in cur_included_vals
            for a_val in cur_included_vals:
                if (a_val + c_val) in cur_included_vals:
                    return False
            # Check 3: 2*c_val in cur_included_vals
            if (2 * c_val) in cur_included_vals:
                return False
            return True
        
        def try_additions(cur_included, cur_vals):
            """Try to add any element that fits. Returns (new_included, new_vals, added_any)."""
            added_any = False
            for idx in range(n):
                if idx not in cur_included:
                    if can_add(idx, cur_vals):
                        cur_included.add(idx)
                        cur_vals.add(elements[idx])
                        added_any = True
            return cur_included, cur_vals, added_any
        
        def try_swaps(cur_included, cur_vals):
            """Try single swaps: remove one, add another. Returns (new_included, new_vals, swapped)."""
            for out_idx in list(cur_included):
                temp_included = cur_included - {out_idx}
                temp_vals = cur_vals - {elements[out_idx]}
                for in_idx in range(n):
                    if in_idx not in temp_included:
                        if can_add(in_idx, temp_vals):
                            temp_included.add(in_idx)
                            temp_vals.add(elements[in_idx])
                            return temp_included, temp_vals, True
            return cur_included, cur_vals, False
        
        # Simulated annealing parameters
        initial_temp = 1.0
        cooling_rate = 0.995
        min_temp = 0.01
        temp = initial_temp
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - start_time < time_limit:
            # Phase 1: Greedy additions
            included, included_vals, added_any = try_additions(included, included_vals)
            if added_any:
                no_improve_count = 0
                if len(included) > best_size:
                    best_size = len(included)
                    best_included = set(included)
                    best_included_vals = set(included_vals)
                continue
            
            # Phase 2: Try swaps
            new_included, new_vals, swapped = try_swaps(included, included_vals)
            if swapped:
                included, included_vals = new_included, new_vals
                no_improve_count = 0
                if len(included) > best_size:
                    best_size = len(included)
                    best_included = set(included)
                    best_included_vals = set(included_vals)
                continue
            
            # Phase 3: Simulated annealing perturbation
            # Remove 1-3 random elements, then try to fill back
            num_remove = rng.randint(1, 3)
            to_remove = rng.sample(list(included), min(num_remove, len(included)))
            for idx in to_remove:
                included.discard(idx)
                included_vals.discard(elements[idx])
            
            # Try to add back as many as possible
            included, included_vals, _ = try_additions(included, included_vals)
            
            # Try swaps to improve
            new_included, new_vals, swapped = try_swaps(included, included_vals)
            if swapped:
                included, included_vals = new_included, new_vals
            
            # Accept or reject based on simulated annealing
            if len(included) >= best_size:
                no_improve_count = 0
                if len(included) > best_size:
                    best_size = len(included)
                    best_included = set(included)
                    best_included_vals = set(included_vals)
            else:
                # Accept with probability exp(-delta/temp)
                delta = best_size - len(included)
                if temp > min_temp:
                    accept_prob = math.exp(-delta / temp)
                    if rng.random() < accept_prob:
                        # Accept the worse solution
                        pass
                    else:
                        # Revert to best
                        included = set(best_included)
                        included_vals = set(best_included_vals)
                else:
                    # Revert to best
                    included = set(best_included)
                    included_vals = set(best_included_vals)
                
                no_improve_count += 1
            
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
                # Restart with higher temperature if stuck
                if no_improve_count > max_no_improve:
                    temp = initial_temp
                    no_improve_count = 0
                    # Reset to best
                    included = set(best_included)
                    included_vals = set(best_included_vals)
        
        return [elements[i] for i in best_included]

    def solve_with_ilp(elements):
        """Solve using ILP if available and fast enough."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            n = len(elements)
            if n == 0:
                return []
            
            val_to_idx = {v: i for i, v in enumerate(elements)}
            
            # Collect constraints
            constraint_rows = []
            for i in range(n):
                for j in range(i, n):
                    s = elements[i] + elements[j]
                    if s in val_to_idx:
                        k = val_to_idx[s]
                        constraint_rows.append((i, j, k))
            
            M = len(constraint_rows)
            
            if M == 0:
                return elements[:]
            
            # For large n, ILP might be slow. Use it only for small n.
            if n > 100:
                return None  # Skip ILP for large sets
            
            A = np.zeros((M, n), dtype=np.float64)
            for idx, (i, j, k) in enumerate(constraint_rows):
                A[idx, i] += 1.0
                A[idx, j] += 1.0
                A[idx, k] += 1.0
            
            c = -np.ones(n)
            bounds = Bounds(lb=0, ub=1)
            cons = LinearConstraint(A, lb=0, ub=2)
            integrality = np.ones(n)
            
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.success:
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                return None
                
        except Exception:
            return None

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try ILP first for small sets
        ilp_result = solve_with_ilp(elems)
        if ilp_result is not None:
            results.append(ilp_result)
            continue
        
        # Greedy initialization
        initial = greedy_init(elems)
        
        # Local search with time budget
        remaining_time = deadline - time.time()
        time_per_instance = max(5.0, remaining_time / 3.0)
        subset = local_search(elems, initial, time_limit=time_per_instance)
        
        results.append(subset)
        
        # Check if we're running out of time
        if time.time() > deadline - 10:
            # Return what we have
            break
    
    # Ensure we have 3 results
    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elems = instance(size, seed)
        # Fallback: odd numbers
        results.append([x for x in elems if x % 2])
    
    return results
# EVOLVE-BLOCK-END
