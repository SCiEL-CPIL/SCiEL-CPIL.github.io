"""Maximize reciprocal sum of subset of [2,256] with pairwise lcm > 256."""
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    val = [0] * n
    for i in range(n):
        val[i] = numbers[i]

    # Precompute conflict masks (bit i conflicts with bit j)
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = val[i], val[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = [1.0 / val[i] for i in range(n)]

    def sum_of(mask):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += weights[i]
        return s

    def can_add(i, mask):
        return (conflict[i] & mask) == 0

    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if conflict[i] & (1 << j):
                            return False
        return True

    # --- Build initial solution: all of [129,256] (mutually compatible) ---
    base_mask = 0
    for i in range(n):
        if val[i] >= 129:
            base_mask |= (1 << i)

    # Try adding small numbers greedily to base
    def greedy_add(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)) and can_add(i, new_mask):
                new_mask |= (1 << i)
        return new_mask

    candidates = []
    candidates.append(base_mask)
    candidates.append(greedy_add(base_mask))

    # Alternative: start empty, greedy by weight
    def greedy_from_empty(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask

    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    candidates.append(greedy_from_empty(order_w))

    # Greedy by weight but prefer large numbers (to keep [129,256] core)
    order_large = list(range(n))
    order_large.sort(key=lambda i: -val[i])
    candidates.append(greedy_from_empty(order_large))

    # Mixed: large first then small
    order_mixed = order_large + order_w
    candidates.append(greedy_from_empty(order_mixed))

    best_mask = 0
    best_sum = -1.0
    for c in candidates:
        s = sum_of(c)
        if s > best_sum:
            best_sum = s
            best_mask = c

    # --- Branch and bound for maximum weight independent set ---
    # We solve this using a recursive branch-and-bound with LP relaxation bounds.
    
    # Sort vertices by weight descending for better pruning
    order = list(range(n))
    order.sort(key=lambda i: -weights[i])
    
    # Precompute adjacency in sorted order
    adj = [0] * n  # conflict masks
    
    def lp_bound(candidates_mask, current_weight):
        """Compute LP relaxation bound for remaining candidates."""
        # Simple bound: sum of all weights in candidates that don't conflict with each other
        # Better: use greedy coloring on the conflict subgraph
        bound = current_weight
        # For a quick bound, just add all remaining weights (loose but fast)
        # For tighter bound, we need to solve the fractional packing
        # Use a simple greedy bound: process vertices, add if no conflict with already-added
        added = 0
        for i in range(n):
            if candidates_mask & (1 << i):
                if can_add(i, added):
                    added |= (1 << i)
                    bound += weights[i]
        return bound

    def branch_and_bound(candidates_mask, current_mask, current_weight, depth=0):
        """Recursive branch and bound for MWIS."""
        nonlocal best_mask, best_sum
        
        if elapsed() > DEADLINE:
            return
        
        # Update best if current is better
        if current_weight > best_sum:
            best_sum = current_weight
            best_mask = current_mask
        
        if candidates_mask == 0:
            return
        
        # Compute upper bound
        # Tighter bound: sum of weights of candidates that can be added
        # without conflicting with each other (greedy approximation)
        upper_bound = current_weight
        temp_added = 0
        for i in range(n):
            if candidates_mask & (1 << i):
                if can_add(i, temp_added):
                    temp_added |= (1 << i)
                    upper_bound += weights[i]
        
        if upper_bound <= best_sum + 1e-15:
            return
        
        # Choose branch vertex: pick the one with highest weight
        branch_i = -1
        max_w = -1
        for i in range(n):
            if candidates_mask & (1 << i):
                if weights[i] > max_w:
                    max_w = weights[i]
                    branch_i = i
        
        if branch_i < 0:
            return
        
        # Branch 1: include branch_i
        if can_add(branch_i, current_mask):
            new_mask = current_mask | (1 << branch_i)
            new_cand = candidates_mask & ~(1 << branch_i)
            # Remove all conflicts of branch_i from candidates
            new_cand &= ~conflict[branch_i]
            branch_and_bound(new_cand, new_mask, current_weight + weights[branch_i], depth + 1)
        
        # Branch 2: exclude branch_i
        new_cand2 = candidates_mask & ~(1 << branch_i)
        branch_and_bound(new_cand2, current_mask, current_weight, depth + 1)

    # Run branch and bound
    all_mask = (1 << n) - 1
    branch_and_bound(all_mask, 0, 0.0)

    # Final verification
    assert is_valid(best_mask), "Result is invalid!"

    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(val[i])
    result.sort()

    if not result:
        result = [2]

    return result
