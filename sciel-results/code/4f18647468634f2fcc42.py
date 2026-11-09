import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin

    best_set = [31, 32, 34]
    best_size = 3

    # Precompute divisibility relationships for all possible subset sums
    MAX_SUM = 8000

    # divisors[i] = set of j < i such that i % j == 0
    # multiples[i] = set of j > i such that j % i == 0
    divisors = [set() for _ in range(MAX_SUM + 1)]
    multiples = [set() for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i * 2, MAX_SUM + 1, i):
            divisors[j].add(i)
            multiples[i].add(j)

    def is_valid_full(S):
        """Check if set S is valid using full subset sum enumeration."""
        if not S:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums

        non_empty_set = sums - {0}
        for u in non_empty_set:
            for m in multiples[u]:
                if m in non_empty_set:
                    return False
            for d in divisors[u]:
                if d in non_empty_set:
                    return False
        return True

    def try_add(S, x, sums_set):
        """
        Try to add x to S. S is assumed valid.
        sums_set is the set of all subset sums of S (including 0).
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        # New sums are x + s for s in sums_set
        new_sums = set()
        for s in sums_set:
            ns = s + x
            if ns in sums_set:
                return (False, None)
            new_sums.add(ns)

        # Check divisibility: for each new sum u, check if any divisor or multiple is in sums_set
        # Also check pairs among new_sums themselves
        for u in new_sums:
            # Check against existing sums
            for m in multiples[u]:
                if m in sums_set:
                    return (False, None)
            for d in divisors[u]:
                if d in sums_set:
                    return (False, None)

        # Check divisibility among new_sums themselves
        new_sums_list = list(new_sums)
        for i, u in enumerate(new_sums_list):
            for v in new_sums_list[i+1:]:
                if u < v:
                    if v % u == 0:
                        return (False, None)
                else:
                    if u % v == 0:
                        return (False, None)

        all_sums = sums_set | new_sums
        return (True, all_sums)

    def greedy_build(candidates, start_S=None, start_sums=None):
        """Greedy build: iterate through candidates, add if valid."""
        if start_S is None:
            S = []
            sums_set = {0}
        else:
            S = list(start_S)
            sums_set = set(start_sums)

        for x in candidates:
            if x in S:
                continue
            ok, new_sums = try_add(S, x, sums_set)
            if ok:
                S.append(x)
                sums_set = new_sums

        return S, sums_set

    # Strategy 1: Binary signature approach
    def construct_binary_signature():
        by_highest_bit = {}
        for n in range(1, 257):
            hb = n.bit_length() - 1
            if hb not in by_highest_bit:
                by_highest_bit[hb] = []
            by_highest_bit[hb].append(n)

        best_local = []

        for attempt in range(80):
            if time.time() - start_time > time_limit:
                break

            bit_positions = sorted(by_highest_bit.keys(), reverse=True)
            random.shuffle(bit_positions)

            S = []
            sums_set = {0}

            for bp in bit_positions:
                if time.time() - start_time > time_limit:
                    break

                candidates = by_highest_bit[bp][:]
                random.shuffle(candidates)

                for x in candidates:
                    ok, new_sums = try_add(S, x, sums_set)
                    if ok:
                        S.append(x)
                        sums_set = new_sums
                        break

            if len(S) > len(best_local):
                best_local = S[:]

        return best_local

    binary_result = construct_binary_signature()
    if len(binary_result) > best_size:
        best_size = len(binary_result)
        best_set = binary_result[:]

    # Strategy 2: Randomized greedy with various orderings
    max_restarts = 3000

    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break

        candidates = list(range(1, 257))
        random.shuffle(candidates)

        # Vary the ordering strategy
        strategy = restart % 4
        if strategy == 0:
            pass  # random order
        elif strategy == 1:
            candidates.sort(reverse=True)  # descending
        elif strategy == 2:
            candidates.sort()  # ascending
        else:
            # Sort by some hash-like value
            candidates.sort(key=lambda x: (x * 7919 + restart) % 256)

        S, sums_set = greedy_build(candidates)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

            # Try to extend with remaining numbers
            remaining = [x for x in range(1, 257) if x not in S]
            random.shuffle(remaining)
            S2, sums_set2 = greedy_build(remaining, S, sums_set)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2[:]

    # Strategy 3: Local search - swap elements
    if best_size < 30 and time.time() - start_time < time_limit - 5:
        current_S = list(best_set)
        sums_set = {0}
        for x in current_S:
            new_sums = set()
            for s in sums_set:
                new_sums.add(s + x)
            sums_set = sums_set | new_sums

        for i in range(len(current_S)):
            if time.time() - start_time > time_limit - 2:
                break

            removed = current_S[i]
            S_reduced = current_S[:i] + current_S[i+1:]

            sums_reduced = {0}
            for x in S_reduced:
                new_sums = set()
                for s in sums_reduced:
                    new_sums.add(s + x)
                sums_reduced = sums_reduced | new_sums

            for x in range(1, 257):
                if x in S_reduced:
                    continue
                ok, new_sums = try_add(S_reduced, x, sums_reduced)
                if ok:
                    new_S = S_reduced + [x]
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = new_S[:]

    # Final verification
    if len(best_set) > 0:
        if not is_valid_full(best_set):
            best_set = [31, 32, 34]
            best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
