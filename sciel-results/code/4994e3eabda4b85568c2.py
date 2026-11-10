import time
import numpy as np
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512] with maximum cardinality.
    Uses Mixed Integer Linear Programming (MILP) via SciPy to find the optimal solution.
    Falls back to a greedy + local search approach if MILP fails or times out.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    n = 512

    # Helper function for greedy construction (fallback)
    def greedy_construction(n):
        selected = set()
        for x in range(1, n + 1):
            # Check if adding x creates a 4-AP where x is the largest term
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a >= 1 and (a in selected) and (a + d in selected) and (a + 2 * d in selected):
                    valid = False
                    break
            if valid:
                selected.add(x)
        return sorted(selected)

    # Helper function for local search improvement
    def local_search_improve(S, n, time_limit=10):
        """
        Improve set S using local search.
        Tries to swap elements or add elements by removing conflicts.
        """
        S_set = set(S)
        start = time.time()
        
        def is_4ap_free_add(x, S_set):
            """Check if x can be added to S_set without creating a 4-AP."""
            for d in range(1, (n - x) // 3 + 1):
                # x is smallest: x, x+d, x+2d, x+3d
                if (x+d in S_set) and (x+2*d in S_set) and (x+3*d in S_set):
                    return False
            for d in range(1, (x - 1) // 3 + 1):
                # x is largest: x-3d, x-2d, x-d, x
                a = x - 3 * d
                if a >= 1 and (a in S_set) and (a + d in S_set) and (a + 2 * d in S_set):
                    return False
                # x is 2nd: x-d, x, x+d, x+2d
                if x - d >= 1 and (x - d in S_set) and (x + d in S_set) and (x + 2 * d in S_set):
                    return False
                # x is 3rd: x-2d, x-d, x, x+d
                if x - 2 * d >= 1 and (x - 2 * d in S_set) and (x - d in S_set) and (x + d in S_set):
                    return False
            return True

        improved = False
        while time.time() - start < time_limit:
            # Try to add a random element not in S
            candidates = [x for x in range(1, n + 1) if x not in S_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            added = False
            for x in candidates:
                if is_4ap_free_add(x, S_set):
                    S_set.add(x)
                    added = True
                    improved = True
                    break
            
            if not added:
                # Try a swap: remove u, add v
                # Pick a random u from S
                u_list = list(S_set)
                if not u_list:
                    break
                u = random.choice(u_list)
                # Try to find a v that can be added if u is removed
                # This is expensive to check directly. 
                # Alternative: Just try to add random v, if conflict, 
                # try to remove one element from the conflict.
                
                # Simpler heuristic: Random restart or perturbation
                # Let's try to find a v that conflicts with u, and see if removing u allows v
                # This is complex. Let's stick to simple add-if-possible and maybe random removals to unlock adds.
                
                # Try removing a random element to see if it allows adding multiple others?
                # For now, if no simple add, try a few random swaps
                for _ in range(10):
                    if time.time() - start > time_limit:
                        break
                    u = random.choice(list(S_set))
                    v_candidates = [x for x in range(1, n + 1) if x not in S_set]
                    if not v_candidates:
                        break
                    v = random.choice(v_candidates)
                    
                    S_test = S_set - {u}
                    if is_4ap_free_add(v, S_test):
                        S_set = S_test | {v}
                        improved = True
                        break
        
        return sorted(S_set)

    # Attempt MILP Solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Generate all 4-APs
        aps = []
        for d in range(1, (n - 3) // 3 + 1):
            max_a = n - 3 * d
            for a in range(1, max_a + 1):
                # a, a+d, a+2d, a+3d
                # 0-indexed
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

        num_aps = len(aps)
        num_vars = n

        # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
        c = -np.ones(num_vars)

        # Constraint Matrix A (sparse)
        rows = []
        cols = []
        data = []
        for i, (a, b, c_idx, d_idx) in enumerate(aps):
            rows.extend([i, i, i, i])
            cols.extend([a, b, c_idx, d_idx])
            data.extend([1, 1, 1, 1])

        A = csr_matrix((data, (rows, cols)), shape=(num_aps, num_vars), dtype=np.int8)

        # Constraints: A @ x <= 3
        constraints = [LinearConstraint(A, lb=-np.inf, ub=3)]

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)

        # Integrality: All variables are integers
        integrality = np.ones(num_vars)

        # Solve
        res = milp(
            c=c,
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
            options={"time_limit": 60, "mip_rel_gap": 0.0}
        )

        if res.status == 0:  # Optimal solution found
            x = res.x
            selected = [i + 1 for i in range(n) if x[i] > 0.5]
            # Verify
            S_set = set(selected)
            # Quick check
            is_valid = True
            for a in S_set:
                for d in range(1, (n - a) // 3 + 1):
                    if (a + d in S_set) and (a + 2 * d in S_set) and (a + 3 * d in S_set):
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                return sorted(selected)
            else:
                # If invalid (should not happen), fall back
                pass
        elif res.status == 1:  # Iteration limit reached, might have a feasible solution
            if res.x is not None:
                x = res.x
                selected = [i + 1 for i in range(n) if x[i] > 0.5]
                S_set = set(selected)
                # Verify
                is_valid = True
                for a in S_set:
                    for d in range(1, (n - a) // 3 + 1):
                        if (a + d in S_set) and (a + 2 * d in S_set) and (a + 3 * d in S_set):
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    # This is a feasible solution, might not be optimal but better than greedy?
                    # We can try to improve it with local search
                    improved = local_search_improve(selected, n, time_limit=10)
                    return improved
                else:
                    pass

    except Exception as e:
        # Fallback to Greedy + Local Search
        pass

    # Fallback: Greedy + Local Search
    greedy_set = greedy_construction(n)
    improved_set = local_search_improve(greedy_set, n, time_limit=20)
    
    return improved_set
