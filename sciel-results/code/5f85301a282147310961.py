"""Optimized construction of 3-AP-free subset of [1,256] maximizing sum of reciprocals.
Uses branch-and-bound with bitset acceleration on the conflict graph."""
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256

    # Precompute conflict graph using bitsets
    # conflict[i] is a bitset of all j != i such that {i, j} is part of a 3-AP
    # Two elements i, j conflict if there exists k such that {i, j, k} forms a 3-AP
    # For a 3-AP {a, b, c} with a < b < c: b = (a+c)/2, c = 2b-a, a = 2b-c
    # So i and j conflict if:
    #   - j = 2*i - k for some k in [1,n] (i is middle or largest)
    #   - i = 2*j - k for some k in [1,n]
    #   - there exists k such that i, j, k form an AP (one is the middle of the other two)

    # Two numbers a, b (a < b) are in conflict if there exists c in [1,n] such that
    # {a, b, c} is a 3-AP. This happens when:
    #   c = 2*b - a (if c <= n) -> a, b, c form AP with b as middle
    #   a = 2*b - c (if a >= 1) -> c, b, a form AP, i.e., a + c = 2*b
    #   b = 2*a - c (if b >= 1) -> c, a, b form AP, i.e., a + b = 2*? No...
    # Let me reconsider. Three numbers x, y, z form a 3-AP if one is the average of the other two.
    # For pair (i, j) with i < j, they conflict if:
    #   Case 1: i is smallest, j is middle: need k > j such that j - i = k - j, so k = 2*j - i <= n
    #   Case 2: i is smallest, j is largest: need k between i and j such that j - k = k - i, so k = (i+j)/2, requires i+j even
    #   Case 3: i is middle, j is largest: need k < i such that i - k = j - i, so k = 2*i - j >= 1

    conflict = [0] * (n + 1)  # 1-indexed, conflict[i] is integer bitset
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Check if i and j can be part of a 3-AP
            # Case 1: i smallest, j middle: k = 2*j - i
            k = 2 * j - i
            if k <= n:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            # Case 2: i smallest, j largest: k = (i+j)/2
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= n:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
            # Case 3: i middle, j largest: k = 2*i - j
            k = 2 * i - j
            if k >= 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                continue

    # Weights: w[i] = 1/i
    weights = np.array([0.0] + [1.0 / i for i in range(1, n + 1)])

    # Greedy initialization: add elements in order of decreasing weight (smallest i first)
    # But check conflicts
    selected_bits = 0  # bitset of selected elements
    selected_set = set()
    for i in range(1, n + 1):
        # Check if i conflicts with any selected element
        if conflict[i] & selected_bits:
            continue
        selected_bits |= (1 << i)
        selected_set.add(i)

    current_sum = sum(weights[i] for i in selected_set)
    best_set = set(selected_set)
    best_sum = current_sum
    best_bits = selected_bits

    def check_conflict(bits, x):
        """Check if adding x to bits creates a conflict."""
        return (conflict[x] & bits) != 0

    # Local search with bitsets
    def local_search(bits, total_sum, max_time):
        """Improve the solution using local search with bitset operations."""
        nonlocal best_set, best_sum, best_bits
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try adding each unselected element
            for x in range(1, n + 1):
                if (bits >> x) & 1:
                    continue
                if not check_conflict(bits, x):
                    new_bits = bits | (1 << x)
                    new_sum = total_sum + weights[x]
                    if new_sum > best_sum + 1e-15:
                        best_sum = new_sum
                        best_bits = new_bits
                        best_set = set(i for i in range(1, n+1) if (new_bits >> i) & 1)
                        bits = new_bits
                        total_sum = new_sum
                        improved = True
            # Try removing one element and adding another
            sel_list = [i for i in range(1, n + 1) if (bits >> i) & 1]
            for x in sel_list:
                if time.time() > max_time:
                    break
                reduced_bits = bits & ~(1 << x)
                reduced_sum = total_sum - weights[x]
                # Try adding each unselected element to reduced
                best_new = None
                best_new_sum = reduced_sum
                for y in range(1, n + 1):
                    if (reduced_bits >> y) & 1:
                        continue
                    if not check_conflict(reduced_bits, y):
                        new_sum = reduced_sum + weights[y]
                        if new_sum > best_new_sum + 1e-15:
                            best_new_sum = new_sum
                            best_new = reduced_bits | (1 << y)
                if best_new is not None and best_new_sum > total_sum + 1e-15:
                    bits = best_new
                    total_sum = best_new_sum
                    if total_sum > best_sum + 1e-15:
                        best_sum = total_sum
                        best_bits = bits
                        best_set = set(i for i in range(1, n+1) if (bits >> i) & 1)
                    improved = True
        return bits, total_sum

    # Run local search
    local_search(best_bits, best_sum, start_time + DEADLINE - 30)

    # Branch and bound for further improvement
    # Sort elements by weight descending for branching
    order = list(range(1, n + 1))
    order.sort(key=lambda x: -weights[x])

    def branch_and_bound(bits, total_sum, idx, max_time):
        """Branch and bound on the conflict graph."""
        nonlocal best_set, best_sum, best_bits
        if time.time() > max_time:
            return
        if idx == len(order):
            if total_sum > best_sum + 1e-15:
                best_sum = total_sum
                best_bits = bits
                best_set = set(i for i in range(1, n+1) if (bits >> i) & 1)
            return

        # Upper bound: current sum + sum of remaining weights (relaxation)
        remaining_sum = sum(weights[order[k]] for k in range(idx, len(order)))
        if total_sum + remaining_sum <= best_sum + 1e-15:
            return

        x = order[idx]

        # Option 1: Don't select x
        branch_and_bound(bits, total_sum, idx + 1, max_time)

        # Option 2: Select x (if no conflict)
        if not check_conflict(bits, x):
            new_bits = bits | (1 << x)
            # Remove all conflicts of x from consideration
            branch_and_bound(new_bits, total_sum + weights[x], idx + 1, max_time)

    # Run B&B with a time limit
    bb_deadline = start_time + DEADLINE - 5
    if time.time() < bb_deadline:
        branch_and_bound(best_bits, best_sum, 0, bb_deadline)

    # Final local search
    local_search(best_bits, best_sum, start_time + DEADLINE - 2)

    return sorted(best_set)
