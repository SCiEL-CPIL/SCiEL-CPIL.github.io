"""B2[2] set in 1..1000: maximize size with every sum having <= 2 representations a+b, a<=b."""
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def sum_counts(chosen):
        """Return dict of sum -> count of representations a+b, a<=b, a,b in chosen."""
        counts = {}
        n = len(chosen)
        for i in range(n):
            a = chosen[i]
            for j in range(i, n):
                b = chosen[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def insertion_sums(x, chosen):
        """New sums added by inserting x: x+y for y in chosen, plus 2x."""
        return [x + y for y in chosen] + [2 * x]

    def is_valid_insertion(x, chosen, counts):
        """Check if inserting x keeps every sum <= 2."""
        for s in insertion_sums(x, chosen):
            if counts.get(s, 0) + 1 > 2:
                return False
        return True

    def apply_insertion(x, chosen, counts):
        """Insert x, update counts. Assumes valid."""
        chosen.append(x)
        for s in insertion_sums(x, chosen[:-1]):
            counts[s] = counts.get(s, 0) + 1
        return chosen, counts

    def remove_element(x, chosen, counts):
        """Remove x, update counts."""
        chosen.remove(x)
        # subtract sums x+y for y in chosen (now without x), plus 2x
        for y in chosen:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        return chosen, counts

    # ---- Phase 1: probabilistic greedy ----
    random.seed(42)
    chosen = []
    counts = {}
    for x in range(1, 1001):
        if elapsed() > DEADLINE - 60:
            break
        new_sums = insertion_sums(x, chosen)
        # slack: number of new sums that would have count < 2 after insertion
        slack = 0
        valid = True
        for s in new_sums:
            c = counts.get(s, 0) + 1
            if c > 2:
                valid = False
                break
            if c < 2:
                slack += 1
        if not valid:
            continue
        # accept probability proportional to slack (more slack = better balance)
        # base acceptance with a temperature-like factor
        p = min(1.0, slack / 3.0 + 0.5)
        if random.random() < p:
            apply_insertion(x, chosen, counts)

    # ---- Phase 2: local search — add-only + swaps ----
    def best_add(chosen, counts):
        """Find an unchosen x that can be added; return (x, slack) for best, or None."""
        best_x = None
        best_slack = -1
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            new_sums = insertion_sums(x, chosen)
            slack = 0
            valid = True
            for s in new_sums:
                c = counts.get(s, 0) + 1
                if c > 2:
                    valid = False
                    break
            if not valid:
                continue
            for s in new_sums:
                if counts.get(s, 0) + 1 < 2:
                    slack += 1
            if slack > best_slack:
                best_slack = slack
                best_x = x
        if best_x is None:
            return None
        return best_x, best_slack

    chosen_set = set(chosen)
    improved = True
    while improved and elapsed() < DEADLINE - 30:
        improved = False
        res = best_add(chosen, counts)
        if res is not None:
            x, _ = res
            apply_insertion(x, chosen, counts)
            chosen_set.add(x)
            improved = True
            continue
        # No add possible; try swaps to enable future adds
        # Try removing a random element and see if an add becomes possible
        if chosen:
            # try a few random removals
            for _ in range(20):
                if elapsed() > DEADLINE - 30:
                    break
                a = random.choice(chosen)
                chosen2, counts2 = remove_element(a, list(chosen), dict(counts))
                chosen2_set = set(chosen2)
                # find best add in this reduced set
                best_x = None
                best_slack = -1
                for x in range(1, 1001):
                    if x in chosen2_set:
                        continue
                    new_sums = insertion_sums(x, chosen2)
                    valid = True
                    slack = 0
                    for s in new_sums:
                        c = counts2.get(s, 0) + 1
                        if c > 2:
                            valid = False
                            break
                    if not valid:
                        continue
                    for s in new_sums:
                        if counts2.get(s, 0) + 1 < 2:
                            slack += 1
                    if slack > best_slack:
                        best_slack = slack
                        best_x = x
                if best_x is not None and best_slack >= 0:
                    # swap: removed a, added best_x -> size same, but maybe better config
                    # only accept if it enables a subsequent add (check one more add)
                    chosen3, counts3 = apply_insertion(best_x, list(chosen2), dict(counts2))
                    chosen3_set = set(chosen3)
                    # check if an add is now possible
                    add_found = False
                    for x in range(1, 1001):
                        if x in chosen3_set:
                            continue
                        new_sums = insertion_sums(x, chosen3)
                        valid = True
                        for s in new_sums:
                            if counts3.get(s, 0) + 1 > 2:
                                valid = False
                                break
                        if valid:
                            add_found = True
                            break
                    if add_found:
                        chosen, counts = chosen3, counts3
                        chosen_set = set(chosen)
                        improved = True
                        break

    # ---- Phase 3: final polish — exhaustive add pass ----
    while elapsed() < DEADLINE - 5:
        res = best_add(chosen, counts)
        if res is None:
            break
        x, _ = res
        apply_insertion(x, chosen, counts)
        chosen_set.add(x)

    return chosen
