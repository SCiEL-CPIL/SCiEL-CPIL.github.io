"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_constraints(elements):
        """Build all constraint triples (i,j,k) where elements[i]+elements[j]=elements[k]."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        constraint_set = set()
        
        # We need to find all (i, j, k) such that elements[i] + elements[j] == elements[k]
        # and i <= j to avoid duplicate symmetric pairs for the sum, but the constraint
        # x_i + x_j + x_k <= 2 applies to the set of indices.
        # Note: The problem states a+b=c with a,b,c in the set. 
        # If i=j, it's 2*a=c. If i!=j, it's a+b=c.
        
        for i in range(n):
            for j in range(i, n):
                c_val = elements[i] + elements[j]
                if c_val in val_to_idx:
                    k = val_to_idx[c_val]
                    # We store the set of indices involved in the violation.
                    # The constraint is that not all three can be 1.
                    # So x_i + x_j + x_k <= 2.
                    # We add the tuple of indices to the set to deduplicate
                    # if the same triple is found via different i,j? 
                    # No, i,j determines the sum. k is determined.
                    # However, if i,j,k are distinct, the set {i,j,k} is unique.
                    # If i=j, the set is {i,k}. Constraint x_i + x_k <= 1? 
                    # Wait. If a=a, then 2a=c. The condition is no a+b=c.
                    # So if 2a=c, we cannot have a and c in the set.
                    # This means x_a + x_c <= 1.
                    # In the general form x_i + x_j + x_k <= 2:
                    # If i=j, we have x_i + x_i + x_k <= 2 => 2x_i + x_k <= 2.
                    # Since x_i is binary, if x_i=1, 2+x_k <= 2 => x_k=0. Correct.
                    # If x_i=0, x_k can be 0 or 1. Correct.
                    # So the form x_i + x_j + x_k <= 2 works for all cases.
                    
                    # We need to represent this in the sparse matrix.
                    # The row will have 1s at indices i, j, k.
                    # If i==j, we have 2 at index i and 1 at index k?
                    # No, the variable is x_i. The term is x_i (for a) + x_i (for b) + x_k (for c).
                    # So coefficient for x_i is 2, for x_k is 1.
                    # Let's just store the indices and multiplicities.
                    
                    # For simplicity in the sparse matrix builder, we can just add rows.
                    # If i == j:
                    #   coeffs: {i: 2, k: 1}
                    # Else:
                    #   coeffs: {i: 1, j: 1, k: 1}
                    
                    constraint_set.add((i, j, k))
        
        return list(constraint_set)

    def solve_ilp(elements, time_budget):
        """Solve the sum-free subset problem exactly using ILP."""
        n = len(elements)
        if n == 0:
            return []
        
        constraints = build_constraints(elements)
        
        if not constraints:
            return elements[:]
        
        m = len(constraints)
        
        # Build sparse constraint matrix
        rows = []
        cols = []
        data = []
        
        for idx, (i, j, k) in enumerate(constraints):
            if i == j:
                # 2*x_i + 1*x_k <= 2
                rows.extend([idx, idx])
                cols.extend([i, k])
                data.extend([2, 1])
            else:
                # 1*x_i + 1*x_j + 1*x_k <= 2
                rows.extend([idx, idx, idx])
                cols.extend([i, j, k])
                data.extend([1, 1, 1])
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
        
        # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
        c = -np.ones(n)
        
        # Constraints: A x <= 2
        constraints_obj = LinearConstraint(A, -np.inf, 2)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x_i must be 0 or 1
        integrality = np.ones(n)
        
        # Solve with time limit
        try:
            result = milp(
                c=c,
                constraints=constraints_obj,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_budget}
            )
            
            if result.status == 0 or result.status == 2:  # Optimal or feasible
                x = result.x
                # Extract selected elements
                selected = [elements[i] for i in range(n) if x[i] > 0.5]
                return selected
            else:
                # If no solution found, return empty
                return []
        except Exception:
            return []

    def greedy_max_abs(S):
        """Fallback greedy heuristic: pick largest absolute values first."""
        S_sorted = sorted(S, key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            if x not in C_set:
                valid = True
                if 2 * x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                        if x - c in C_set:
                            valid = False
                            break
                if valid:
                    C.append(x)
                    C_set.add(x)
        return C

    def is_sum_free(subset):
        """Verify sum-free property."""
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                c = a + subset[j]
                if c in s_set:
                    return False
        return True

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time budget: 40s per instance, leaving 20s margin
    # Total 120s for solving, 25s margin
    time_per_instance = 40.0
    
    for size, seed in instances:
        vals = instance(size, seed)
        
        # Check remaining time
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        if remaining < 10:
            # Fallback to greedy if running out of time
            res = greedy_max_abs(vals)
            results.append(res)
            continue
        
        # Allocate budget for this instance
        budget = min(time_per_instance, remaining - 5.0)
        
        sol = solve_ilp(vals, budget)
        
        if not sol or not is_sum_free(sol):
            # If ILP failed or returned invalid, use greedy
            sol = greedy_max_abs(vals)
            
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
