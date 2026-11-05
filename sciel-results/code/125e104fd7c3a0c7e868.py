"""Exact ILP solver for large sum-free subsets using scipy.optimize.milp."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_instance_ilp(elements, time_budget):
        """
        Formulate the sum-free subset problem as an ILP and solve it.
        Variables: x_i in {0, 1} for each element i.
        Objective: Maximize sum(x_i).
        Constraints: For all distinct a, b, c in elements such that a + b = c,
                     x_a + x_b + x_c <= 2.
                     Note: The condition "no a+b=c with a,b,c in S" means that
                     if a+b=c, we cannot have x_a=1 AND x_b=1 AND x_c=1.
                     So x_a + x_b + x_c <= 2 is the correct linear constraint.
                     We must consider all ordered pairs (a,b) from the set of elements
                     such that a+b is also in the set of elements.
        """
        n = len(elements)
        elem_set = set(elements)
        
        # Precompute all constraints
        # For each pair (a, b) in elements, if a+b is in elements, add constraint.
        # We need to be careful about duplicates and ordering.
        # The constraint is symmetric in a and b, but the set of triples is what matters.
        # Let's collect all unique triples (a, b, c) with a+b=c.
        # To avoid O(n^3) or too many duplicate constraints, we can just iterate a, b.
        
        # Optimization: 
        # Instead of building a dense matrix, we can build a sparse matrix if scipy supports it.
        # scipy.optimize.milp accepts sparse matrices.
        
        # Let's collect constraints in a list of (indices, coeffs)
        # coeffs are always 1, 1, 1 for x_a, x_b, x_c.
        # We need to map element values to indices.
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # List of constraints: each is a list of (idx, coeff)
        # We will build a sparse matrix later.
        # For now, let's just count and collect.
        
        # Actually, building the sparse matrix directly is better.
        # Rows: constraints. Cols: variables (n).
        
        # We need to identify all triples (a, b, c) in elements such that a+b=c.
        # a and b can be the same.
        # c must be in elements.
        
        # To avoid adding redundant constraints (e.g., a+b=c and b+a=c), 
        # we can enforce a <= b if we consider unordered pairs? 
        # No, the constraint x_a + x_b + x_c <= 2 is the same regardless of order.
        # So we can just iterate all a, b in elements, check if a+b in elements, 
        # and add the constraint. Duplicates are okay for the solver, but 
        # we can deduplicate by sorting (a, b, c) or just (min(a,b), max(a,b), c).
        
        # Let's use a set of tuples to store unique constraints.
        # A constraint is defined by the set of indices involved.
        # Since the coefficients are all 1, the constraint is symmetric.
        # So (idx_a, idx_b, idx_c) is the same as (idx_b, idx_a, idx_c).
        # We can store sorted tuple of indices.
        
        constraint_indices = set()
        
        # Iterate through all pairs
        # n=240 -> 240*240 = 57600 pairs. This is fast.
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                c = a + b
                if c in val_to_idx:
                    k = val_to_idx[c]
                    # Store sorted indices to deduplicate
                    # Note: a, b, c are distinct values? 
                    # If a=1, b=1, c=2. Indices might be different if 1 appears once? 
                    # The problem states "distinct nonzero integers". So values are unique.
                    # Thus indices are unique.
                    # However, a and b could be the same value? 
                    # "a=b allowed" in the sum-free definition. 
                    # But the set contains distinct integers. So a and b are the same value only if i=j.
                    # So indices are (i, j, k).
                    # If i == j, it's (i, i, k).
                    # If i != j, it's (i, j, k).
                    # The constraint is x_i + x_j + x_k <= 2.
                    # This is the same as x_j + x_i + x_k <= 2.
                    # So we can sort the indices to deduplicate.
                    # But wait, if i=j, sorting gives (i, i, k).
                    # If i!=j, sorting gives (min(i,j), max(i,j), k).
                    # This correctly deduplicates (i,j,k) and (j,i,k).
                    
                    # However, is it possible that the same triple of indices is generated 
                    # by different (a,b) pairs? 
                    # a+b=c. If elements are distinct, the representation is unique for values.
                    # So (val_a, val_b, val_c) is unique.
                    # Thus (idx_a, idx_b, idx_c) is unique for ordered pairs.
                    # Sorting handles the symmetry of a and b.
                    
                    sorted_idx = tuple(sorted((i, j, k)))
                    constraint_indices.add(sorted_idx)
        
        # If there are no constraints, the answer is all elements.
        if not constraint_indices:
            return elements[:]
        
        # Build sparse matrix
        # Number of constraints m = len(constraint_indices)
        # Number of variables n
        m = len(constraint_indices)
        
        # For scipy sparse matrix, we need row, col, data arrays.
        # Each constraint has 3 non-zero entries (coeff 1).
        # So total non-zeros is 3*m.
        
        rows = []
        cols = []
        data = []
        
        for idx, (i, j, k) in enumerate(constraint_indices):
            rows.extend([idx, idx, idx])
            cols.extend([i, j, k])
            data.extend([1, 1, 1])
            
        # Create sparse matrix
        # A_ub x <= b_ub
        # Here A_ub is the constraint matrix, b_ub is the vector of 2s.
        # We use LinearConstraint(A, lb, ub).
        # We want A x <= 2. So lb = -inf, ub = 2.
        
        # scipy.sparse.csr_matrix
        import scipy.sparse as sp
        A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
        
        # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
        c = -np.ones(n)
        
        # Constraints: A x <= 2
        # LinearConstraint(A, -np.inf, 2)
        # But milp expects bounds on the linear form? 
        # No, LinearConstraint(A, lb, ub) means lb <= A x <= ub.
        # So lb = -np.inf * m, ub = 2 * m.
        
        # Note: milp minimizes c.T x.
        
        constraints = LinearConstraint(A, -np.inf, 2)
        
        # Bounds on x: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x_i must be integer (0 or 1)
        integrality = np.ones(n)
        
        # Solve
        # Set time limit? 
        # scipy.optimize.milp does not have a direct time limit parameter in the function call 
        # in older versions, but HiGHS solver (default) might respect some options.
        # Actually, `options` parameter can be passed.
        # Let's check if we can pass a time limit.
        # According to docs, `options` is a dict passed to the solver.
        # For HiGHS, 'time_limit' might be an option? 
        # Let's assume we can't strictly enforce it inside milp easily without checking docs.
        # However, we can run it in a thread? No, GIL.
        # We can just call it and hope it's fast. For n=240, m~few thousand, it should be very fast.
        # If it's too slow, we might timeout the whole program.
        # To be safe, we can try to solve with a small time limit if supported.
        # Let's try passing 'time_limit' in options. If it fails, we fallback.
        
        try:
            # options for HiGHS
            # https://github.com/ERGO-Code/HiGHS
            # Common options: 'time_limit' (seconds)
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': int(time_budget)}
            )
        except Exception as e:
            # If options not supported or error, try without options
            try:
                res = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds
                )
            except Exception as e2:
                # Fallback to heuristic if ILP fails completely
                return heuristic_fallback(elements)
        
        if res.status == 0:  # Optimal
            x = res.x
            # x might be float, round to 0/1
            x = np.round(x).astype(int)
            return [elements[i] for i in range(n) if x[i] == 1]
        elif res.status == 1:  # Feasible (not necessarily optimal, e.g. due to time limit)
            x = res.x
            x = np.round(x).astype(int)
            # Ensure validity? The solver should return a feasible integer solution if integrality is set.
            # But sometimes it returns fractional if not converged? 
            # With integrality=1, it should be integer.
            # Let's verify and repair if needed.
            subset = set(elements[i] for i in range(n) if x[i] == 1)
            # Quick check and repair
            # If invalid, remove violating elements greedily
            valid_subset = set()
            for v in elements:
                if v not in valid_subset:
                    # Check if adding v to valid_subset creates violation
                    # v = a+b? a+v? v+a?
                    can_add = True
                    for a in valid_subset:
                        if (v - a) in valid_subset or (v + a) in valid_subset or (2*v) in valid_subset:
                            can_add = False
                            break
                    if can_add:
                        valid_subset.add(v)
            return list(valid_subset)
        else:
            # Infeasible or other error. Fallback.
            return heuristic_fallback(elements)

    def heuristic_fallback(elements):
        # Simple greedy: sort by absolute value? Or random?
        # Let's use a random greedy with repair.
        rng = random.Random(42)
        subset = set()
        shuffled = elements[:]
        rng.shuffle(shuffled)
        for v in shuffled:
            can_add = True
            for a in subset:
                if (v - a) in subset or (v + a) in subset or (2*v) in subset:
                    can_add = False
                    break
            if can_add:
                subset.add(v)
        return list(subset)

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    
    # Allocate time budget
    # ILP is usually very fast for these sizes.
    # Let's give it a generous budget, but check time before each.
    
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback
            results.append(heuristic_fallback(elems))
            continue
            
        # Time budget for this instance
        # If we have 145s total, and 3 instances, we can give ~40s each.
        # But ILP should be much faster.
        time_budget = 40
        if i == 2:
            time_budget = 60 # Last one gets more if needed, but unlikely
        
        # Ensure we don't exceed deadline
        remaining = DEADLINE - time.time()
        if time_budget > remaining - 1:
            time_budget = max(1, remaining - 1)
            
        result = solve_instance_ilp(elems, time_budget)
        results.append(result)
        
        # Check time
        if time.time() - start_time > DEADLINE:
             # Fill remaining with fallbacks
             while len(results) < 3:
                 results.append(heuristic_fallback(instances[len(results)]))
             break

    return results
# EVOLVE-BLOCK-END
