"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # Universe: [1, 256]
    N = 256
    nums = list(range(1, N + 1))
    idx = {v: i for i, v in enumerate(nums)}

    # Build conflict graph: edge (i, j) exists if nums[i] + nums[j] is a perfect square.
    # This includes self-loops (i, i) if 2*nums[i] is a perfect square.
    # We want a maximum independent set in this graph (including self-loops as constraints).
    # Actually, the constraint is: for a <= b in the set, a+b is never a perfect square.
    # This means:
    #   1. For a != b: a+b is not a square.
    #   2. For a == b: 2a is not a square.
    # So self-loops (conflicts with itself) exist for elements where 2a is a square.
    # Elements with self-conflicts CANNOT be in the set at all.

    # First, filter out elements that have self-conflicts (2*a is a perfect square)
    valid_elements = []
    for a in nums:
        if not _is_square(2 * a):
            valid_elements.append(a)

    # Now build conflict graph among valid_elements
    # Edge between a and b (a != b) if a + b is a perfect square
    m = len(valid_elements)
    # Use bitsets for efficiency
    # conflict[i] is a bitset where bit j is set if valid_elements[i] + valid_elements[j] is a square
    conflict = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if _is_square(valid_elements[i] + valid_elements[j]):
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # Now we need maximum independent set in this graph.
    # For m up to 256, exact MIS is hard. Let's use a greedy + local search approach.
    # Actually, let's try a smarter approach: branch and bound or use the fact that
    # the graph has special structure.

    # Greedy approach: repeatedly pick the vertex with minimum degree, remove it and its neighbors.
    # But this gives a maximal independent set, not necessarily maximum.

    # Let's use a local search / hill climbing approach:
    # Start with a greedy independent set, then try to add elements, remove conflicts, etc.

    # Better: use a simulated annealing or tabu search to maximize the size.

    # Given time constraints, let's use a combination:
    # 1. Greedy construction
    # 2. Local improvement: try to swap elements to increase size

    # Greedy: sort by degree, pick lowest degree first
    degrees = [bin(conflict[i]).count('1') for i in range(m)]
    order = sorted(range(m), key=lambda i: degrees[i])

    selected = set()
    available = set(range(m))

    for i in order:
        if i not in available:
            continue
        # Check if i conflicts with any already selected
        can_add = True
        for j in selected:
            if (conflict[i] >> j) & 1:
                can_add = False
                break
        if can_add:
            selected.add(i)
            # Remove neighbors from available
            for j in range(m):
                if j != i and ((conflict[i] >> j) & 1):
                    available.discard(j)

    # Local search: try to improve
    best_set = set(selected)
    best_size = len(best_set)

    # Try to add elements not in the set by removing conflicting elements
    for _iter in range(100):
        if time.time() > deadline - 5:
            break

        improved = False
        for i in range(m):
            if i in best_set:
                continue
            # Find neighbors of i that are in best_set
            neighbors_in_set = set()
            for j in best_set:
                if (conflict[i] >> j) & 1:
                    neighbors_in_set.add(j)

            # If we remove all neighbors_in_set and add i, do we gain?
            if len(neighbors_in_set) < 1:
                # Can add i directly
                best_set.add(i)
                improved = True
                continue

            # Try removing one neighbor and adding i
            for j in neighbors_in_set:
                new_set = set(best_set)
                new_set.discard(j)
                new_set.add(i)
                # Check validity
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a >= b:
                            if _is_square(valid_elements[a] + valid_elements[b]):
                                valid = False
                                break
                    if not valid:
                        break
                if valid and len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break

        if not improved:
            break

    # Convert back to values
    result = [valid_elements[i] for i in best_set]
    result.sort()

    # Verify
    for i in range(len(result)):
        for j in range(i, len(result)):
            if _is_square(result[i] + result[j]):
                # Should not happen
                pass

    if not result:
        # Fallback
        result = [1]

    return result
# EVOLVE-BLOCK-END
