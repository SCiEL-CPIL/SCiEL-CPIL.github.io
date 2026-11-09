"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_milp(S):
        """
        Solve the maximum sum-free subset problem using MILP.
        Variables x_i in {0,1} indicate if S[i] is in the subset.
        Constraints: For all a,b,c in S with a+b=c, x_a + x_b + x_c <= 2.
        Objective: Maximize sum(x_i).
        """
        n = len(S)
        if n == 0:
            return []
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # Collect constraints
        # We need to handle the case where a+b=c.
        # We iterate over all pairs (i, j) with i <= j to cover all ordered pairs (a,b)
        # because a+b=c is symmetric in a,b.
        # The constraint is x_i + x_j + x_k <= 2.
        
        A_rows = []
        b_vals = []
        
        # Optimization: Use a set to avoid duplicate constraints if the same triple is found multiple times
        # Although (i,j) is unique for i<=j, the triple (i,j,k) is determined.
        # However, if S has duplicates (it doesn't, they are distinct), or if different pairs sum to same c.
        # E.g., 1+4=5 and 2+3=5. Constraints: x_1+x_4+x_5<=2 and x_2+x_3+x_5<=2. These are distinct.
        # So we just generate all pairs.
        
        for i in range(n):
            for j in range(i, n):
                s_val = S[i] + S[j]
                if s_val in val_to_idx:
                    k = val_to_idx[s_val]
                    # Constraint: x_i + x_j + x_k <= 2
                    # Create a sparse row
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    row[k] = 1
                    A_rows.append(row)
                    b_vals.append(2)
        
        if not A_rows:
            # No constraints, take all
            return S[:]
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        # Objective: Maximize sum(x) -> Minimize -sum(x)
        c = -np.ones(n)
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(n)
        ub = np.ones(n)
        
        # Integrality: x_i must be integer
        integrality = np.ones(n)
        
        constraints = [
            LinearConstraint(A, lb_ub=[0, 2]), # Wait, LinearConstraint takes (A, lb, ub) or (A, lb_ub)
            # Actually, scipy.optimize.milp takes a list of LinearConstraint objects or an array.
            # Let's use the standard form: A @ x >= lb, A @ x <= ub
            # Our constraints are A @ x <= 2.
            # So lb = -inf, ub = 2.
            LinearConstraint(A, -np.inf, 2),
        ]
        
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0: # Optimal
                x = res.x
                # x is float, should be 0 or 1
                selected_indices = np.where(x > 0.5)[0]
                return [S[i] for i in selected_indices]
            else:
                # Fallback if MILP fails
                return greedy_fallback(S)
        except Exception:
            return greedy_fallback(S)

    def greedy_fallback(S):
        """Fast greedy heuristic if MILP fails or times out."""
        S_set = set(S)
        # Try to maximize size.
        # Heuristic: Sort by absolute value descending? Or just greedy addition.
        # Let's try a few orderings and pick the best.
        best_subset = []
        
        def can_add(x, current_set):
            if 2 * x in current_set:
                return False
            for a in current_set:
                if a + x in current_set:
                    return False
                if x - a in current_set:
                    return False
            return True
        
        orderings = [
            lambda x: -abs(x),
            lambda x: abs(x),
            lambda x: x,
            lambda x: -x,
        ]
        
        for key in orderings:
            S_sorted = sorted(S, key=key)
            current = []
            current_set = set()
            for x in S_sorted:
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
            if len(current) > len(best_subset):
                best_subset = current
        
        # Local search improvement on the best greedy result
        improved = True
        current_set = set(best_subset)
        while improved:
            improved = False
            # Try adding
            for x in S:
                if x not in current_set:
                    if can_add(x, current_set):
                        current_set.add(x)
                        improved = True
            # Try swapping: remove one, add one
            # This is expensive, so we'll just stick to additions for fallback
            # A simple removal-re-addition loop might help escape local optima
            for a in list(current_set):
                temp_set = set(current_set)
                temp_set.discard(a)
                add_count = 0
                for x in S:
                    if x not in temp_set and can_add(x, temp_set):
                        add_count += 1
                if add_count > 1:
                    current_set = temp_set
                    improved = True
        
        return list(current_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE - 10:
            # Not enough time, return empty or best effort
            results.append([])
            continue
            
        S = instance(size, seed)
        
        # Solve using MILP
        # MILP is very fast for n=240.
        subset = solve_milp(S)
        
        # Verify
        if not is_sum_free(subset):
            subset = greedy_fallback(S)
            
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
