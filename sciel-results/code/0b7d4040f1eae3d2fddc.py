"""Construct maximum-size set with cardinality-separated subset sums."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    max_val = 256
    max_size = 60

    def try_add(elements, x, dp, current_size):
        """Try to add x to elements, check if cardinality separation holds."""
        if x in elements:
            return None

        new_dp = [dp[k] for k in range(len(dp))]
        # Extend dp if needed
        while len(new_dp) <= current_size + 1:
            new_dp.append(0)

        # Update: for k from current_size down to 0
        for k in range(current_size, -1, -1):
            new_dp[k+1] |= (dp[k] << x)

        # Check cardinality separation: for each pair (i,j) with i≠j, 
        # dp[i] & dp[j] must be 0 (no common sums)
        n_k = len(new_dp)
        for i in range(n_k):
            for j in range(i+1, n_k):
                if new_dp[i] & new_dp[j]:
                    return None

        return elements + [x], new_dp

    def verify_disjoint(dp, n):
        """Check that all dp[i] for i in 0..n are pairwise disjoint."""
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    def count_bits(dp, n):
        """Count total number of achievable sums across all cardinalities."""
        return sum(bin(dp[i]).count('1') for i in range(n + 1))

    # Greedy construction with bitset DP
    best_elements = []
    best_dp = [1]

    # Try multiple orderings to find a good initial solution
    orderings = [
        ("inc", list(range(1, max_val + 1))),
        ("dec", list(range(max_val, 0, -1))),
        ("sparse", [1, 3, 7, 15, 31, 63, 127, 255, 256] + list(range(2, 128))),
        ("prime", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251] + list(range(4, 257, 2))),
    ]

    for ordering_name, ordering in orderings:
        if time.time() > deadline:
            break

        elements = []
        dp = [1]

        for x in ordering:
            if time.time() > deadline:
                break
            if len(elements) >= max_size:
                break
            result = try_add(elements, x, dp, len(elements))
            if result:
                elements, dp = result

        if len(elements) > len(best_elements):
            best_elements = list(elements)
            best_dp = dp

    # Now try to improve with local search: swap elements
    if len(best_elements) < max_size and time.time() < deadline:
        in_set = set(best_elements)
        # Try adding elements not in best_elements
        for x in range(1, max_val + 1):
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            result = try_add(best_elements, x, best_dp, len(best_elements))
            if result:
                best_elements, best_dp = result
                in_set.add(x)
                if len(best_elements) >= max_size:
                    break

    # Local search: try swapping elements to potentially allow more additions
    if len(best_elements) < max_size and time.time() < deadline:
        for _ in range(50):  # limited number of swap attempts
            if time.time() > deadline:
                break
            if len(best_elements) >= max_size:
                break

            # Try swapping one element with another not in the set
            i = random.randint(0, len(best_elements) - 1)
            old_elem = best_elements[i]

            # Find a candidate to swap in
            candidates = [x for x in range(1, max_val + 1) if x not in in_set]
            if not candidates:
                break

            # Try a few random candidates
            for _ in range(10):
                if time.time() > deadline:
                    break
                if not candidates:
                    break
                new_elem = random.choice(candidates)
                candidates.remove(new_elem)

                # Try the swap
                new_elements = best_elements[:i] + best_elements[i+1:] + [new_elem]
                # Rebuild dp for the new set
                n = len(new_elements)
                new_dp = [0] * (n + 1)
                new_dp[0] = 1
                for x in new_elements:
                    for k in range(n, 0, -1):
                        new_dp[k] |= (new_dp[k-1] << x)

                if verify_disjoint(new_dp, n):
                    # Check if this swap allows adding more elements
                    in_set_new = set(new_elements)
                    can_add = False
                    for x in range(1, max_val + 1):
                        if x in in_set_new:
                            continue
                        result = try_add(new_elements, x, new_dp, n)
                        if result:
                            can_add = True
                            break

                    if can_add or len(new_elements) > len(best_elements):
                        best_elements = new_elements
                        best_dp = new_dp
                        in_set = set(best_elements)
                        break

    # Final verification
    if not best_elements:
        return [1]

    # Verify: check all subset sums by cardinality
    n = len(best_elements)
    verify_dp = [0] * (n + 1)
    verify_dp[0] = 1
    for x in best_elements:
        for k in range(n, 0, -1):
            verify_dp[k] |= (verify_dp[k-1] << x)

    # Check disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if verify_dp[i] & verify_dp[j]:
                # Invalid! Fall back to a known-valid small set
                return [1, 3, 7]

    # Ensure all are Python ints
    return [int(x) for x in best_elements]
