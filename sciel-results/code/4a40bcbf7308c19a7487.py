import time
from math import gcd

def construct():
    start = time.time()
    DEADLINE = start + 70.0

    LIMIT = 512
    N = 512

    # Precompute lcm table for pairs (a,b) with a,b in [1,512]
    # Only need upper triangle; store as list of lists (bytearray or list of ints)
    # Actually, let's just compute on the fly since 512^2 = 262144, which is small.

    # Build compatibility matrix: compat[a][b] = True if lcm(a,b) <= LIMIT
    # Use a bitset for efficiency
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            l = a * b // gcd(a, b)
            if l <= LIMIT:
                compat[a] |= (1 << b)
                if a != b:
                    compat[b] |= (1 << a)

    def can_add(x, current_bits):
        """Check if x can be added to the set represented by current_bits.
        current_bits has bit a set if a is in the current set."""
        return (current_bits & compat[x]) == current_bits

    # Greedy approach with multiple orderings
    best_set_bits = 0
    best_count = 0

    # Ordering 1: numerical
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -sum(1 for a in range(1, N+1) if (compat[x] >> a) & 1)),
        sorted(range(1, N + 1), key=lambda x: sum(1 for a in range(1, N+1) if (compat[x] >> a) & 1)),
    ]

    for order in orderings:
        if time.time() > DEADLINE:
            break
        bits = 0
        count = 0
        for x in order:
            if time.time() > DEADLINE:
                break
            if (bits >> x) & 1:
                continue
            if can_add(x, bits):
                bits |= (1 << x)
                count += 1
        if count > best_count:
            best_count = count
            best_set_bits = bits

    # Local search: try to improve by swapping elements
    # Convert best set to a list
    current_set = set()
    for a in range(1, N + 1):
        if (best_set_bits >> a) & 1:
            current_set.add(a)

    # Try to add more by removing one element and adding two
    # Or try to replace elements to free up compatibility
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        current_list = sorted(current_set)
        # Try removing one element and see if we can add two that were previously blocked
        for i, remove in enumerate(current_list):
            if time.time() > DEADLINE:
                break
            remaining = set(current_list)
            remaining.discard(remove)
            remaining_bits = 0
            for a in remaining:
                remaining_bits |= (1 << a)

            # Find candidates that can be added to remaining
            # but were blocked by remove
            candidates = []
            for x in range(1, N + 1):
                if (best_set_bits >> x) & 1:
                    continue  # already in set
                if can_add(x, remaining_bits):
                    candidates.append(x)

            if len(candidates) >= 2:
                # Try pairs from candidates
                found = False
                for ci in range(len(candidates)):
                    if found or time.time() > DEADLINE:
                        break
                    for cj in range(ci + 1, len(candidates)):
                        if time.time() > DEADLINE:
                            break
                        c1, c2 = candidates[ci], candidates[cj]
                        if can_add(c1, remaining_bits) and can_add(c2, remaining_bits | (1 << c1)):
                            new_set = remaining | {c1, c2}
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                best_set_bits = 0
                                for a in current_set:
                                    best_set_bits |= (1 << a)
                                best_count = len(current_set)
                                improved = True
                                found = True
                                break
                if improved:
                    break

    # Final local search: try replacing one element with another to allow more additions
    if time.time() < DEADLINE:
        current_list = sorted(current_set)
        for i, elem in enumerate(current_list):
            if time.time() > DEADLINE:
                break
            remaining = set(current_list)
            remaining.discard(elem)
            remaining_bits = 0
            for a in remaining:
                remaining_bits |= (1 << a)

            # Try to add an element that requires removing elem
            for x in range(1, N + 1):
                if (best_set_bits >> x) & 1:
                    continue
                if can_add(x, remaining_bits) and not can_add(x, best_set_bits):
                    # x can be added if we remove elem
                    # Check if we can also add another element
                    new_bits = remaining_bits | (1 << x)
                    for y in range(1, N + 1):
                        if (best_set_bits >> y) & 1:
                            continue
                        if y == x:
                            continue
                        if can_add(y, new_bits):
                            new_set = remaining | {x, y}
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                best_set_bits = 0
                                for a in current_set:
                                    best_set_bits |= (1 << a)
                                best_count = len(current_set)
                                break
                    break

    # Convert to sorted list
    result = []
    for a in range(1, N + 1):
        if (best_set_bits >> a) & 1:
            result.append(a)

    # Ensure nonempty
    if not result:
        result = [1]

    return result
