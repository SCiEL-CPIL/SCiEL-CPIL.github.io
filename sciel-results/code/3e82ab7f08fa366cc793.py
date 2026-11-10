import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s limit

    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        reciprocals = [Fraction(1, e) for e in elements]
        sums = [Fraction(0, 1)]
        seen = {Fraction(0, 1)}
        for r in reciprocals:
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True

    all_elements = list(range(1, 23))

    # Multiple diverse greedy orders
    orders = [
        list(range(22, 0, -1)),
        list(range(1, 23)),
        list(range(11, 0, -1)) + list(range(22, 11, -1)),
        list(range(1, 23, 2)) + list(range(22, 0, -2)),
        list(range(22, 0, -2)) + list(range(1, 23, 2)),
        [22, 21, 1, 2, 20, 19, 3, 4, 18, 17, 5, 6, 16, 15, 7, 8, 14, 13, 9, 10, 12, 11],
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
    ]

    best_set = []

    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for e in order:
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        if len(current) > len(best_set):
            best_set = current[:]

    # Greedy fill: try to add remaining elements
    if time.time() < deadline:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                candidate = best_set + [e]
                if is_dissociated(candidate):
                    best_set = candidate
                    remaining.remove(e)
                    improved = True

    # Local search: try swapping elements to enable adding more
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_len = len(best_set)
            best_set_set = set(best_set)
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(len(best_set)):
                    if i == j:
                        continue
                    for e in all_elements:
                        if e in best_set_set:
                            continue
                        candidate = best_set[:j] + [e] + best_set[j+1:]
                        if is_dissociated(candidate):
                            # Try to add more elements after the swap
                            remaining_after = [x for x in all_elements if x not in set(candidate)]
                            improved_candidate = candidate[:]
                            for x in remaining_after:
                                if time.time() > deadline:
                                    break
                                test = improved_candidate + [x]
                                if is_dissociated(test):
                                    improved_candidate = test
                            if len(improved_candidate) > len(best_set):
                                best_set = improved_candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if len(best_set) <= best_len:
                break

    # Fallback
    if not best_set or len(best_set) <= 9:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    # Final verification
    if not is_dissociated(best_set):
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
