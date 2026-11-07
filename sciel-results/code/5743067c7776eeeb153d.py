import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s margin for evaluator's 90s limit

    def get_subset_sums(elements):
        """Return set of all nonempty subset sums."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def has_divisibility(sums):
        """Check if any two distinct sums divide each other."""
        sum_list = sorted(sums)
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                if sum_list[j] % sum_list[i] == 0:
                    return True
        return False

    # Candidate pool
    candidates = list(range(1, 257))

    best_set = []
    best_size = 0

    # Strategy 1: Superincreasing-like construction using large primes
    # Elements that are pairwise co-prime and large enough
    # Try to build a set where subset sums don't divide each other

    # First, try a deterministic approach with carefully chosen elements
    # Key insight: if we use elements that are all > 128, then:
    # - Any single element is > 128
    # - Any pair sum is > 256 (but max is 512)
    # - We need to ensure no sum divides another

    # Better strategy: use a greedy randomized search with incremental subset sum tracking

    def try_add_element(current, current_sums, c):
        """Try adding element c to current set. Return (new_sums, valid) or (None, False)."""
        new_sums = set()
        for s in current_sums:
            new_sums.add(s)
            new_sums.add(s + c)
        new_sums.add(c)

        # Check divisibility
        sum_list = sorted(new_sums)
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                if sum_list[j] % sum_list[i] == 0:
                    return None, False
        return new_sums, True

    # Multiple restarts with different random orderings
    for attempt in range(5000):
        if time.time() > deadline - 5.0:
            break

        random.shuffle(candidates)
        current = []
        current_sums = set()

        for c in candidates:
            if len(current) >= 30:
                break
            new_sums, valid = try_add_element(current, current_sums, c)
            if valid:
                current = current + [c]
                current_sums = new_sums

        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # If we haven't found a large set, try a more structured approach
    if best_size < 15:
        # Try using elements from specific ranges that are less likely to cause divisibility
        # Elements in (128, 256] have subset sums that are large
        large_candidates = [c for c in range(129, 257)]
        random.shuffle(large_candidates)

        for attempt in range(2000):
            if time.time() > deadline - 5.0:
                break

            current = []
            current_sums = set()

            for c in large_candidates:
                if len(current) >= 30:
                    break
                new_sums, valid = try_add_element(current, current_sums, c)
                if valid:
                    current = current + [c]
                    current_sums = new_sums

            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                if best_size >= 30:
                    break

    # Final attempt: try mixed approach with some small and some large elements
    if best_size < 20 and time.time() < deadline - 10.0:
        # Use a mix: some elements from 100-256, avoid very small ones
        mixed = [c for c in range(50, 257)]
        random.shuffle(mixed)

        for attempt in range(3000):
            if time.time() > deadline - 5.0:
                break

            current = []
            current_sums = set()

            for c in mixed:
                if len(current) >= 30:
                    break
                new_sums, valid = try_add_element(current, current_sums, c)
                if valid:
                    current = current + [c]
                    current_sums = new_sums

            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                if best_size >= 30:
                    break

    # Ensure we return a valid list of Python ints
    result = [int(x) for x in best_set]
    return result
