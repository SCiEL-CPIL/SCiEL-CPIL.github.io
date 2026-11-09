import time
import random
import math
import numpy as np
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin

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

    def solve_ilp(elements):
        """Solve the sum-free subset problem using MILP with sparse constraints."""
        n = len(elements)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Collect constraint data: for each unique (i, j, k) multiset where v_i + v_j = v_k
        # Store as list of (row_indices, row_values)
        seen = set()
        constraint_data = []
        
        for i in range(n):
            vi = elements[i]
            for j in range(i, n):
                vj = elements[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    
                    # Build counts
                    counts = {}
                    counts[i] = counts.get(i, 0) + 1
                    counts[j] = counts.get(j, 0) + 1
                    counts[k] = counts.get(k, 0) + 1
                    
                    key = tuple(sorted((idx, cnt) for idx, cnt in counts.items()))
                    
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    row_indices = list(counts.keys())
                    row_values = [counts[idx] for idx in row_indices]
                    constraint_data.append((row_indices, row_values))
        
        if not constraint_data:
            return elements[:]
        
        # Build sparse matrix
        m = len(constraint_data)
        row_ptr = [0]
        col_idx = []
        data = []
        
        for row_indices, row_values in constraint_data:
            col_idx.extend(row_indices)
            data.extend(row_values)
            row_ptr.append(row_ptr[-1] + len(row_indices))
        
        A_sparse = csr_matrix((data, col_idx, row_ptr), shape=(m, n))
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n)
        
        # Constraints: A @ x <= 2
        cons = LinearConstraint(A_sparse, lb=-np.inf, ub=2)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all integer
        integrality = np.ones(n)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            if res.success:
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                return None
        except Exception:
            return None

    def local_search_polish(elements, initial_solution):
        """Improve solution by trying to add elements and removing blockers."""
        if not initial_solution:
            return initial_solution
        
        solution = set(initial_solution)
        elements_set = set(elements)
        
        def can_add(v, current_set):
            """Check if v can be added to current_set while maintaining sum-free property."""
            for a in current_set:
                if (v - a) in current_set or (v + a) in current_set:
                    return False
                if 2 * v in current_set:
                    return False
            return True
        
        # Try adding elements that aren't in the solution
        added = True
        while added and time.time() < deadline - 10:
            added = False
            for v in elements:
                if v not in solution and can_add(v, solution):
                    solution.add(v)
                    added = True
        
        # Try remove-and-retry: for each element in solution, try removing it
        # and see if we can add more elements
        for v in list(solution):
            if time.time() > deadline - 10:
                break
            trial_solution = solution - {v}
            # Try to add elements that were blocked by v
            new_size = len(trial_solution)
            for w in elements:
                if w not in trial_solution and can_add(w, trial_solution):
                    trial_solution.add(w)
            if len(trial_solution) > len(solution):
                solution = trial_solution
        
        return list(solution)

    def heuristic_fallback(elements):
        """Simple greedy fallback if ILP fails."""
        included_vals = set()
        for v in sorted(elements, key=abs, reverse=True):
            ok = True
            for a in included_vals:
                if (v - a) in included_vals or (v + a) in included_vals:
                    ok = False
                    break
                if 2 * v in included_vals:
                    ok = False
                    break
            if ok:
                included_vals.add(v)
        return list(included_vals)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try ILP first
        ilp_result = solve_ilp(elems)
        
        if ilp_result is not None:
            # Verify and polish
            if is_sum_free(ilp_result):
                polished = local_search_polish(elems, ilp_result)
                if is_sum_free(polished):
                    results.append(polished)
                else:
                    results.append(ilp_result)
            else:
                results.append(heuristic_fallback(elems))
        else:
            results.append(heuristic_fallback(elems))
            
        # Check time
        if time.time() > deadline - 5:
             break

    # Ensure we have 3 results
    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elems = instance(size, seed)
        results.append(heuristic_fallback(elems))
        
    return results
