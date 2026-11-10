import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    # Set a hard deadline to ensure we return within 60s (evaluator limit)
    # We use 55s to be safe, allowing for final cleanup/return.
    deadline = start_time + 55
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: for each element x, which pairs (a,b) with a,b != x
    # form a bad triple with x
    # conflict_pairs[x] = set of frozenset({a,b}) where {a,b,x} is bad
    conflict_pairs = [set() for _ in range(n + 1)]
    
    # Enumerate all triples to find bad ones
    # Optimization: Only check c > b > a
    for a in range(1, n + 1):
        if time.time() > deadline - 10:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # Bad triple {a,b,c}
                    conflict_pairs[a].add(frozenset((b, c)))
                    conflict_pairs[b].add(frozenset((a, c)))
                    conflict_pairs[c].add(frozenset((a, b)))

    def count_conflicts(x, selected_set):
        # Check if x conflicts with any pair in selected_set
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return 1
        return 0

    def count_conflicts_excluding(x, selected_set, excluded_y):
        # Check if x conflicts with any pair in selected_set, 
        # ignoring pairs that contain excluded_y
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                # If the conflicting pair does NOT contain the excluded element, 
                # the conflict is still active.
                if a != excluded_y and b != excluded_y:
                    return 1
        return 0

    best_result = set()
    
    # Define various orderings for greedy construction
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        # Sort by degree in conflict graph (fewer conflicts first is often better for packing)
        sorted(range(1, n + 1), key=lambda x: len(conflict_pairs[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict_pairs[x])),
    ]

    # Run greedy for each order
    for order in orders:
        if time.time() > deadline - 15:
            break
        selected = set()
        for x in order:
            if x in selected:
                continue
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
            if time.time() > deadline - 15:
                break

    # Local search improvement
    current = best_result.copy()
    
    # Phase 1: Try to add any element that fits
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
        if not improved and time.time() > deadline - 10:
            break

    # Phase 2: 1-for-1 Swaps
    # We try to remove one element y and add one element x
    # To speed this up, we limit the number of swap iterations
    swap_attempts = 0
    max_swap_attempts = 5000 # Cap to prevent infinite loop or timeout
    
    while time.time() < deadline - 5 and swap_attempts < max_swap_attempts:
        swap_found = False
        # Iterate through potential additions
        for x in range(1, n + 1):
            if x in current:
                continue
            # Iterate through potential removals
            for y in list(current):
                swap_attempts += 1
                if swap_attempts > max_swap_attempts or time.time() > deadline - 5:
                    break
                
                # Check if x can be added to current \ {y}
                # This is equivalent to: no pair in conflict_pairs[x] is fully contained in current \ {y}
                if count_conflicts_excluding(x, current, y) == 0:
                    # Perform swap
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found or swap_attempts > max_swap_attempts:
                break
        if not swap_found:
            break

    return list(current) if current else list(best_result)
