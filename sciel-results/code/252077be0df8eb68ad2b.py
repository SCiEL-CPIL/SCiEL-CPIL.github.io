"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit

    # Generate 2-3-5 smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)

    smooth.sort()
    smooth_set = set(smooth)

    def is_antichain(terms):
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = terms[i], terms[j]
                if a % b == 0 or b % a == 0:
                    return False
        return True

    answers = []

    for n in range(1, 129):
        # Check time budget
        if time.time() > deadline:
            # Fallback: use a simple valid solution if available
            # For now, we'll try to find any valid solution quickly
            answer = None
            # Try small counts with quick check
            for count in range(1, 6):
                candidates = [x for x in smooth if x <= n]
                if len(candidates) < count:
                    break
                if count == 1:
                    if n in smooth_set:
                        answer = [n]
                        break
                elif count == 2:
                    for a in candidates:
                        if a >= n // 2:
                            break
                        b = n - a
                        if b in smooth_set and b != a and b > a:
                            if a % b != 0 and b % a != 0:
                                answer = [a, b]
                                break
                    if answer:
                        break
                else:
                    # For count >= 3, use combinations with early termination
                    for terms in combinations(candidates, count):
                        if sum(terms) == n and is_antichain(terms):
                            answer = list(terms)
                            break
                    if answer:
                        break
                if answer:
                    break
            if answer is None:
                # Last resort: just use [n] if n is smooth, otherwise fail
                if n in smooth_set:
                    answer = [n]
                else:
                    # This should not happen, but to be safe
                    answer = []
            answers.append(answer)
            continue

        answer = None
        candidates = [x for x in smooth if x <= n]

        # Try count = 1
        if n in smooth_set:
            answer = [n]
        else:
            # Try count = 2
            found = False
            for a in candidates:
                if a >= n // 2:
                    break
                b = n - a
                if b in smooth_set and b != a and b > a:
                    if a % b != 0 and b % a != 0:
                        answer = [a, b]
                        found = True
                        break
            if found:
                pass
            else:
                # Try count = 3, 4, ...
                for count in range(3, 31):
                    if time.time() > deadline:
                        break
                    if len(candidates) < count:
                        break

                    # Pruning: min and max possible sums
                    min_sum = sum(candidates[:count])
                    max_sum = sum(candidates[-count:])
                    if min_sum > n or max_sum < n:
                        continue

                    for terms in combinations(candidates, count):
                        if sum(terms) == n and is_antichain(terms):
                            answer = list(terms)
                            break
                    if answer is not None:
                        break

        if answer is None:
            # Fallback: should not happen if all targets are representable
            # But to be safe, try a larger count or use a known valid pattern
            # For now, mark as empty list which would be invalid
            # We need to ensure all targets have a representation.
            # Let's try a more exhaustive search for this specific n
            answer = []
            for count in range(1, 31):
                if len(candidates) < count:
                    break
                if count == 1:
                    if n in smooth_set:
                        answer = [n]
                        break
                elif count == 2:
                    for a in candidates:
                        if a >= n // 2:
                            break
                        b = n - a
                        if b in smooth_set and b != a and b > a:
                            if a % b != 0 and b % a != 0:
                                answer = [a, b]
                                break
                    if answer:
                        break
                else:
                    for terms in combinations(candidates, count):
                        if sum(terms) == n and is_antichain(terms):
                            answer = list(terms)
                            break
                    if answer:
                        break
                if answer:
                    break

        answers.append(answer)

    return answers
# EVOLVE-BLOCK-END
