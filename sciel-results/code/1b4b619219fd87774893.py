import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Compute LCM to bound the search period
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    L = 1
    for d in divisors:
        L = lcm(L, d)

    # Generate all multiples for each divisor up to a reasonable limit
    # We'll search within [1, 3*L] to be safe
    search_limit = 3 * L

    # For each divisor index, store sorted list of (value, divisor_index)
    # We'll create a merged sorted list of all multiples
    all_multiples = []  # (value, divisor_idx)
    for i, d in enumerate(divisors):
        for k in range(1, search_limit // d + 1):
            val = k * d
            all_multiples.append((val, i))

    # Sort by value
    all_multiples.sort()

    # For bipartite matching: divisors (n nodes) to values in window
    # We'll use a sliding window and check for perfect matching
    # Since n is small (10), we can use simple DFS-based matching

    def has_perfect_matching(window_multiples):
        """Check if there's a perfect matching from divisors to distinct values in window_multiples.
        window_multiples is a list of (value, divisor_idx) where value is in the window.
        We need to assign each divisor a distinct value from the window that is a multiple of it.
        """
        # Build adjacency: for each divisor, which values in the window are multiples
        # But values must be distinct, so it's a bipartite matching
        # Left side: divisors (0..n-1)
        # Right side: distinct values in the window
        # Edge: divisor i connected to value v if v is a multiple of divisors[i]

        # Collect distinct values in window and which divisors they can serve
        value_to_divisors = defaultdict(list)
        for val, div_idx in window_multiples:
            value_to_divisors[val].append(div_idx)

        # For matching, we try to assign each divisor to a value
        # Use DFS-based bipartite matching (Kuhn's algorithm)
        # Right side is values, left side is divisors
        # match[div_idx] = value assigned to divisor div_idx, or -1
        match = [-1] * n

        def dfs(div_idx, visited):
            for val, candidate_div_idx in window_multiples:
                if candidate_div_idx != div_idx:
                    continue
                if visited[val]:
                    continue
                visited[val] = True
                if match[div_idx] == -1 or dfs(match[div_idx], visited):
                    match[div_idx] = val
                    return True
            return False

        # Try to match each divisor
        for div_idx in range(n):
            visited = {}
            if not dfs(div_idx, visited):
                return False
        return True

    # Sliding window approach
    # left and right pointers into all_multiples
    # For each left, extend right until we have a valid matching, then record and try to shrink
    best_range = float('inf')
    best_solution = None

    # To speed up, we'll use a different strategy:
    # For each possible minimum value (from the sorted list), try to find the best assignment
    # Greedy: for the minimum, pick a divisor that has this value as a multiple
    # Then for remaining divisors, pick the smallest multiple >= min that is unused

    # Actually, let's use a smarter approach:
    # Iterate over each possible "anchor" as the minimum value
    # For each anchor, greedily assign the smallest possible multiple >= anchor for each divisor
    # But we need to handle conflicts (distinctness)

    # Let's use local search / simulated annealing style
    # Start with a feasible solution, then try to improve

    # First, get an initial feasible solution using greedy
    def greedy_solution(start_min):
        """Given a minimum value, greedily assign multiples >= start_min for each divisor."""
        assigned = {}
        used_values = set()
        for i, d in enumerate(divisors):
            # Find smallest multiple of d >= start_min not in used_values
            k = (start_min + d - 1) // d
            if k < 1:
                k = 1
            found = False
            for attempt in range(100):
                val = (k + attempt) * d
                if val > max_val:
                    break
                if val not in used_values:
                    assigned[i] = val
                    used_values.add(val)
                    found = True
                    break
            if not found:
                return None
        return assigned

    # Get initial solution
    initial = greedy_solution(1)
    if initial is None:
        return divisors[:]

    best_range = max(initial.values()) - min(initial.values())
    best_solution = [initial[i] for i in range(n)]

    # Now try to improve by shifting the window
    # Try different starting points
    # For each divisor, try each of its multiples as a potential minimum
    # and see if we can get a better range

    # More efficient: use the sliding window with matching check
    # But that might be slow. Let's try a randomized local search

    # Randomized local search
    import random
    random.seed(42)

    current = {i: initial[i] for i in range(n)}
    current_range = max(current.values()) - min(current.values())

    for iteration in range(100000):
        if time.time() > deadline:
            break

        # Pick a random divisor and try to change its multiple
        div_idx = random.randint(0, n - 1)
        d = divisors[div_idx]
        old_val = current[div_idx]

        # Try nearby multiples
        k_old = old_val // d
        tried = set()
        for delta in range(-5, 6):
            if delta == 0:
                continue
            k_new = k_old + delta
            if k_new < 1:
                continue
            new_val = k_new * d
            if new_val > max_val or new_val in tried:
                continue
            tried.add(new_val)

            # Check if new_val is already used by another divisor
            conflict = False
            for j in range(n):
                if j != div_idx and current[j] == new_val:
                    conflict = True
                    break
            if conflict:
                continue

            # Try this assignment
            current[div_idx] = new_val
            new_range = max(current.values()) - min(current.values())

            if new_range < current_range:
                current_range = new_range
                if new_range < best_range:
                    best_range = new_range
                    best_solution = [current[i] for i in range(n)]
            else:
                # Accept with some probability (simulated annealing)
                if random.random() < 0.1:
                    current_range = new_range
                else:
                    current[div_idx] = old_val

    # Return best solution found
    if best_solution is None:
        return divisors[:]

    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
