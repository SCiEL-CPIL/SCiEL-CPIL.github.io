import math
import random
import time

def construct():
    deadline = time.time() + 145.0
    N = 511
    C = 512

    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Precompute conflict masks as bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Precompute coprime masks (can_add check)
    coprime_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) == 1:
                mask |= (1 << b)
        coprime_mask[a] = mask

    # Greedy initial solution
    S_mask = 0
    for a in range(N, 0, -1):
        if (S_mask & conflict_mask[a]) == 0:
            S_mask |= (1 << a)

    def mask_to_list(mask):
        result = []
        for a in range(1, N + 1):
            if mask & (1 << a):
                result.append(a)
        return result

    best_S_mask = S_mask
    best_score = sum(w[a] for a in range(1, N + 1) if best_S_mask & (1 << a))

    # Branch and bound for Maximum Weight Independent Set
    # Use recursive branching with upper bound pruning

    def get_available_vertices(mask):
        """Get list of vertices not in mask, sorted by weight descending"""
        available = []
        for a in range(N, 0, -1):
            if not (mask & (1 << a)):
                available.append(a)
        return available

    def upper_bound(available_mask, current_score):
        """Upper bound: current score + sum of weights of all available vertices
        that could potentially be added (greedy relaxation)"""
        # Simple bound: current score + sum of all remaining weights
        # This is loose but fast
        total = current_score
        for a in range(N, 0, -1):
            if available_mask & (1 << a):
                total += w[a]
        return total

    def better_upper_bound(available_mask, current_score):
        """Better upper bound using greedy coloring heuristic on the subgraph"""
        # For each available vertex, check if it conflicts with any other available vertex
        # Upper bound = current_score + sum of max(weights of independent sets in connected components)
        # Simplified: use a greedy independent set on available vertices as a lower bound
        # and add that to current_score
        greedy_score = 0.0
        temp_mask = available_mask
        while temp_mask:
            # Find highest weight vertex
            best_a = 0
            best_w = -1.0
            for a in range(N, 0, -1):
                if temp_mask & (1 << a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
            if best_a == 0:
                break
            greedy_score += best_w
            # Remove best_a and all its conflicts
            temp_mask &= ~(1 << best_a)
            temp_mask &= ~conflict_mask[best_a]
        return current_score + greedy_score

    # Branch and bound
    # State: (current_mask, available_mask, current_score, depth)
    # We branch on the highest-weight available vertex: include it or exclude it

    nodes_visited = 0
    max_nodes = 5000000  # Limit to avoid timeout

    def branch_and_bound(current_mask, available_mask, current_score, depth):
        nonlocal best_S_mask, best_score, nodes_visited, deadline

        if time.time() > deadline:
            return

        nodes_visited += 1
        if nodes_visited > max_nodes:
            return

        if not available_mask:
            if current_score > best_score:
                best_score = current_score
                best_S_mask = current_mask
            return

        # Upper bound check
        ub = better_upper_bound(available_mask, current_score)
        if ub <= best_score + 1e-12:
            return

        # Find the highest weight available vertex
        pivot = 0
        pivot_w = -1.0
        for a in range(N, 0, -1):
            if available_mask & (1 << a):
                if w[a] > pivot_w:
                    pivot_w = w[a]
                    pivot = a
                    break  # First one in descending order is highest

        if pivot == 0:
            return

        # Branch 1: Include pivot
        # Check if pivot can be added (it's in available_mask, so by definition no conflicts with current_mask)
        new_mask_include = current_mask | (1 << pivot)
        # New available: remove pivot and all vertices that conflict with pivot
        new_available_include = available_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound(new_mask_include, new_available_include, current_score + w[pivot], depth + 1)

        # Branch 2: Exclude pivot
        new_available_exclude = available_mask & ~(1 << pivot)
        branch_and_bound(current_mask, new_available_exclude, current_score, depth + 1)

    # Run branch and bound
    initial_available = 0
    for a in range(1, N + 1):
        initial_available |= (1 << a)

    branch_and_bound(0, initial_available, 0.0, 0)

    # Convert best_S_mask to list
    result = []
    for a in range(1, N + 1):
        if best_S_mask & (1 << a):
            result.append(a)

    return result
