"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave 10s margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def count_violations(S):
        Sset = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in Sset:
                    count += 1
        return count

    def greedy_add(S, candidates):
        """Try to add each candidate to S if it doesn't create a violation."""
        Sset = set(S)
        for x in candidates:
            if x in Sset:
                continue
            # Check if adding x creates a violation
            # Violation if: exists a,b in Sset such that a+b=x, or x+a=b, or x+b=a for a,b in Sset
            # i.e., x = a+b for some a,b in Sset, or a = x+b for some b in Sset (i.e., a-b=x), or b = x+a
            # Simpler: check all pairs
            valid = True
            for a in Sset:
                if a + x in Sset:  # a + x = c where c in Sset
                    valid = False
                    break
                if x + a in Sset and x + a != x:  # same as above
                    valid = False
                    break
            if valid:
                # Also need: for any a in Sset, a+x not in Sset (covered above)
                # And: x + a = b means b - a = x, so check if there exist a,b in Sset with b-a=x
                # This is equivalent to: for any a in Sset, a+x in Sset (already checked)
                Sset.add(x)
                S.append(x)
        return S

    def local_search(S, all_elements, max_time=10):
        """Improve S by trying to add excluded elements."""
        Sset = set(S)
        excluded = [x for x in all_elements if x not in Sset]
        S = list(S)
        best = len(S)
        while time.time() - start_time < DEADLINE - 5:
            improved = False
            for x in excluded:
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Try adding x
                new_Sset = Sset | {x}
                # Check if x creates violations
                # A violation is a+b=c where a,b,c in new_Sset
                # New violations must involve x
                # Case 1: x = a+b for a,b in Sset
                # Case 2: a+x = c for a,c in Sset (i.e., c-a=x)
                # Case 3: x+a = c same as case 2
                # Case 4: a+x = b and b+x = a... no, just a+b=c
                # Actually: violations involving x are:
                #   x = a+b (a,b in Sset)
                #   a+x = c (a,c in Sset) => c - a = x
                #   x+a = c (same)
                #   a+b = x (same as first)
                # So check: does any a in Sset have a+x in Sset? Or does any pair a,b in Sset have a+b=x?
                has_violation = False
                for a in Sset:
                    if a + x in Sset:
                        has_violation = True
                        break
                if not has_violation:
                    # Check if x = a+b for any a,b in Sset
                    # This is equivalent to: for any a in Sset, x-a in Sset
                    for a in Sset:
                        if x - a in Sset:
                            has_violation = True
                            break
                if not has_violation:
                    Sset.add(x)
                    S.append(x)
                    excluded.remove(x)
                    improved = True
                    best = len(S)
            if not improved:
                break
        return S

    def solve_instance(elements):
        n = len(elements)
        # Multiple initializations, pick best
        best_S = None
        best_len = 0

        # Init 1: all odds
        S1 = [x for x in elements if x % 2 != 0]
        # Init 2: all positives > max/2
        max_val = max(elements)
        S2 = [x for x in elements if x > max_val / 2]
        # Init 3: all negatives < min/2
        min_val = min(elements)
        S3 = [x for x in elements if x < min_val / 2]
        # Init 4: top n/3 largest
        S4 = sorted(elements, reverse=True)[:max(1, n // 3)]
        # Init 5: bottom n/3 smallest
        S5 = sorted(elements)[:max(1, n // 3)]
        # Init 6: all positives
        S6 = [x for x in elements if x > 0]
        # Init 7: all negatives
        S7 = [x for x in elements if x < 0]

        for S0 in [S1, S2, S3, S4, S5, S6, S7]:
            if not S0:
                continue
            # Verify sum-free
            S0_set = set(S0)
            valid = True
            for a in S0:
                for b in S0:
                    if a + b in S0_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                # Fix by removing violators
                S0_set = set(S0)
                changed = True
                while changed:
                    changed = False
                    to_remove = set()
                    for a in list(S0_set):
                        for b in S0_set:
                            if a + b in S0_set:
                                to_remove.add(a)
                                to_remove.add(b)
                                to_remove.add(a + b)
                    if to_remove:
                        S0_set -= to_remove
                        changed = True
                S0 = list(S0_set)

            # Greedy add
            S = list(S0)
            S = greedy_add(S, [x for x in elements if x not in set(S)])

            # Local search
            S = local_search(S, elements)

            if len(S) > best_len:
                best_len = len(S)
                best_S = S[:]

        return best_S if best_S else []

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 10:
            # Fallback: just odds
            elements = instance(size, seed)
            results.append([x for x in elements if x % 2 != 0])
            continue
        elements = instance(size, seed)
        S = solve_instance(elements)
        # Final verification
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    # Shouldn't happen, but fix if it does
                    S = [x for x in S if x not in (a, b, a+b)]
                    Sset = set(S)
                    break
        results.append(S)

    return results
# EVOLVE-BLOCK-END
