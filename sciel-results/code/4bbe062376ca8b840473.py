def construct():
    import time
    from math import gcd
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = 85.0  # safety margin

    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)

    # Build conflict matrix
    # conflict[i, j] = 1 if lcm(nums[i], nums[j]) <= 256
    # We want to select a subset S such that for all i, j in S, i != j, lcm(nums[i], nums[j]) > 256.
    # This is equivalent to finding a Maximum Weight Independent Set in the graph G where
    # an edge exists between i and j if they CONFLICT (i.e., lcm <= 256).
    # Variables x_i in {0, 1}.
    # Maximize sum( (1/nums[i]) * x_i )
    # Subject to: x_i + x_j <= 1 for all (i, j) where conflict exists.
    #             x_i >= 0

    # Precompute conflicts
    # To save memory and time, we can list the constraints directly.
    # n = 255. Pairs = ~32k. This is small enough for MIP.
    
    conflict_pairs = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            if a * b // gcd(a, b) <= 256:
                conflict_pairs.append((i, j))

    # Objective coefficients
    c = np.array([1.0 / x for x in nums])

    # Constraints: x_i + x_j <= 1
    # A_ub * x <= b_ub
    if conflict_pairs:
        A_ub = np.zeros((len(conflict_pairs), n))
        for k, (i, j) in enumerate(conflict_pairs):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
        b_ub = np.ones(len(conflict_pairs))
    else:
        A_ub = np.zeros((0, n))
        b_ub = np.zeros(0)

    constraints = [
        LinearConstraint(A_ub, -np.inf, b_ub),
        LinearConstraint(np.eye(n), 0, 1)  # 0 <= x_i <= 1
    ]

    integrality = np.ones(n)  # All variables are integers (binary)
    bounds = Bounds(0, 1)

    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # res.x contains the optimal solution
            # Extract indices where x_i is close to 1
            selected_indices = [i for i in range(n) if res.x[i] > 0.5]
            if selected_indices:
                return sorted(nums[i] for i in selected_indices)
            else:
                # Fallback if MIP returns empty (should not happen as we can always pick one)
                return [2]
        else:
            # Fallback to greedy if MIP fails
            pass
    except Exception:
        pass

    # Fallback: Greedy heuristic
    # Sort by 1/x descending (x ascending)
    # Add if no conflict with currently selected
    conflict_set = [set() for _ in range(n)]
    for i, j in conflict_pairs:
        conflict_set[i].add(j)
        conflict_set[j].add(i)
        
    selected = set()
    # Sort indices by value ascending (so 1/x is descending)
    order = sorted(range(n), key=lambda i: nums[i])
    for i in order:
        if time.time() > deadline:
            break
        if all(j not in selected for j in conflict_set[i]):
            selected.add(i)
            
    if selected:
        return sorted(nums[i] for i in selected)
    else:
        return [2]
