import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s wall-clock

    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))

    # Precompute gcd table for speed
    gcd_table = {}
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            gcd_table[(i, j)] = gcd(i, j)

    def get_gcd(a, b):
        if a <= b:
            return gcd_table.get((a, b), gcd(a, b))
        else:
            return gcd_table.get((b, a), gcd(a, b))

    def find_residue(m, selected):
        """
        Find a residue r in [0, m-1] such that (r, m) is disjoint from all (ri, mi) in selected.
        Returns r or None.
        """
        for r in range(m):
            valid = True
            for ri, mi in selected:
                g = get_gcd(m, mi)
                if g > 1:
                    if (r - ri) % g == 0:
                        valid = False
                        break
                else:
                    # gcd == 1 means they always intersect, so this m is incompatible with mi
                    valid = False
                    break
            if valid:
                return r
        return None

    def try_add(m, selected):
        r = find_residue(m, selected)
        if r is not None:
            selected.append((r, m))
            return True
        return False

    best_size = 0
    best_result = []

    # Strategy 1: Greedy with random restarts
    # Bias towards even numbers as they form a large non-coprime clique
    num_iterations = 5000

    for i in range(num_iterations):
        if time.time() > deadline:
            break

        # Randomize the order of moduli
        random.shuffle(all_moduli)

        selected = []
        for m in all_moduli:
            if time.time() > deadline:
                break
            if try_add(m, selected):
                pass
            # If it fails, we just skip it.
            # Note: A modulus that fails now might succeed later if the order was different?
            # No, adding more elements makes the constraint tighter.
            # So if it fails against the current set, it will fail against any superset.
            # So skipping is correct for this specific greedy path.

        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected[:]

    # Strategy 2: Local Search / Improvement on the best found
    if best_result:
        used_moduli = set(m for _, m in best_result)
        remaining = [m for m in all_moduli if m not in used_moduli]
        improved = list(best_result)

        # Try to add remaining moduli one by one
        for m in remaining:
            if time.time() > deadline:
                break
            if try_add(m, improved):
                pass

        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Strategy 3: If we are stuck, try specific families
    # Family: All Even
    evens = [m for m in all_moduli if m % 2 == 0]
    random.shuffle(evens)
    sel_even = []
    for m in evens:
        if time.time() > deadline:
            break
        if try_add(m, sel_even):
            pass

    if len(sel_even) > best_size:
        best_size = len(sel_even)
        best_result = sel_even

    # Family: Multiples of 3
    mult3 = [m for m in all_moduli if m % 3 == 0]
    random.shuffle(mult3)
    sel_3 = []
    for m in mult3:
        if time.time() > deadline:
            break
        if try_add(m, sel_3):
            pass

    if len(sel_3) > best_size:
        best_size = len(sel_3)
        best_result = sel_3

    # Family: Multiples of 5
    mult5 = [m for m in all_moduli if m % 5 == 0]
    random.shuffle(mult5)
    sel_5 = []
    for m in mult5:
        if time.time() > deadline:
            break
        if try_add(m, sel_5):
            pass

    if len(sel_5) > best_size:
        best_size = len(sel_5)
        best_result = sel_5

    # Convert to list of lists
    result = [[r, m] for r, m in best_result]

    if not result:
        result = [[1, 2]]

    return result
