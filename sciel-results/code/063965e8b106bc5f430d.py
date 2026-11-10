"""Branch-and-bound maximum weight independent set on 3-AP conflict graph."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256
    weights = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Build conflict graph: edge between u, v if they form a 3-AP with some w in [1, n]
    # A 3-AP is (a, b, c) with a + c = 2*b.
    # Two numbers u, v conflict if there exists w such that {u, v, w} is a 3-AP.
    # Cases:
    #   u, v are endpoints: w = (u+v)/2, need (u+v) even and 1 <= w <= n, w != u, w != v
    #   u, v are endpoint and middle: if u < v, then w = 2*v - u (v is middle, u is left endpoint), need 1 <= w <= n
    #   or w = 2*u - v (u is middle, v is left endpoint), need 1 <= w <= n
    # Actually, for any pair (u, v) with u != v, they conflict if:
    #   (u + v) % 2 == 0 and 1 <= (u+v)//2 <= n and (u+v)//2 != u and (u+v)//2 != v
    #   OR 1 <= 2*v - u <= n and 2*v - u != u and 2*v - u != v  (v is middle, u is left)
    #   OR 1 <= 2*u - v <= n and 2*u - v != u and 2*u - v != v  (u is middle, v is left)

    # Use bitmasks for efficiency
    # conflicts[u] is a bitmask of all v that conflict with u
    conflicts = [0] * (n + 1)
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            conflict = False
            # Case 1: u and v are endpoints, middle is (u+v)/2
            if (u + v) % 2 == 0:
                mid = (u + v) // 2
                if 1 <= mid <= n and mid != u and mid != v:
                    conflict = True
            # Case 2: u is left endpoint, v is middle, right endpoint is 2*v - u
            if not conflict:
                w = 2 * v - u
                if 1 <= w <= n and w != u and w != v:
                    conflict = True
            # Case 3: v is left endpoint, u is middle, right endpoint is 2*u - v
            if not conflict:
                w = 2 * u - v
                if 1 <= w <= n and w != u and w != v:
                    conflict = True
            if conflict:
                conflicts[u] |= (1 << v)
                conflicts[v] |= (1 << u)

    # Branch and bound for maximum weight independent set
    # We process vertices in order of decreasing weight (smallest index first = highest weight)
    # Order: 1, 2, 3, ..., n (since weight = 1/i is decreasing)

    best_weight = [0.0]
    best_set_mask = [0]

    # Suffix sum of weights for upper bound
    suffix_w = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_w[i] = suffix_w[i + 1] + weights[i]

    # For fast checking, precompute for each vertex the list of conflicting vertices
    conflict_list = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        mask = conflicts[u]
        for v in range(1, n + 1):
            if mask & (1 << v):
                conflict_list[u].append(v)

    # Greedy initial solution
    def greedy_solution():
        S = set()
        for x in range(1, n + 1):
            can_add = True
            for y in S:
                if conflicts[x] & (1 << y):
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S

    initial_S = greedy_solution()
    initial_weight = sum(weights[x] for x in initial_S)
    best_weight[0] = initial_weight
    best_set_mask[0] = 0
    for x in initial_S:
        best_set_mask[0] |= (1 << x)

    # Branch and bound
    # State: current index i, current mask of selected vertices, current weight
    # We decide for vertex i whether to include or exclude it
    # Upper bound: current_weight + sum of weights of all remaining vertices that don't conflict with selected

    def branch_and_bound(i, selected_mask, current_weight):
        if i > n:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set_mask[0] = selected_mask
            return

        if time.time() - start_time > DEADLINE:
            return

        # Upper bound: current_weight + sum of weights of vertices from i to n
        # that are not in conflict with selected_mask
        ub = current_weight
        for j in range(i, n + 1):
            if selected_mask & (1 << j):
                continue
            # Check if j conflicts with any selected vertex
            if conflicts[j] & selected_mask:
                continue
            ub += weights[j]
            if ub <= best_weight[0]:
                break

        if ub <= best_weight[0]:
            return

        # Try including vertex i
        if not (selected_mask & (1 << i)):
            # Check if i conflicts with any selected vertex
            if not (conflicts[i] & selected_mask):
                branch_and_bound(i + 1, selected_mask | (1 << i), current_weight + weights[i])

        # Try excluding vertex i
        branch_and_bound(i + 1, selected_mask, current_weight)

    # Start branch and bound
    branch_and_bound(1, 0, 0.0)

    # Extract solution from best_set_mask
    result = []
    for x in range(1, n + 1):
        if best_set_mask[0] & (1 << x):
            result.append(x)

    # Verify no 3-AP
    def has_3ap(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b:
                    continue
                c = 2 * b - a
                if c in S_set and c != a and c != b:
                    return True
        return False

    if not result:
        # Fallback to greedy if B&B found nothing
        result = list(initial_S)

    assert not has_3ap(result), "Solution contains a 3-AP!"
    assert len(result) > 0, "Solution is empty!"

    return result
# EVOLVE-BLOCK-END
