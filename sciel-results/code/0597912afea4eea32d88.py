import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Base: upper tail [171, 256] is valid
    s = set(range(171, 257))

    def can_add(x, current_set):
        """Check if x can be added to current_set without violating the property."""
        elems = np.array(sorted(current_set), dtype=np.int32)
        m = len(elems)
        if m == 0:
            return True

        # Case 1: x is the smallest (a=x, b,c in current_set with x<b<=c)
        # Need: NOT exists b,c in S with x < b <= c and (b+c) % x == 0
        mask1 = elems > x
        if np.any(mask1):
            b_arr = elems[mask1]
            # For each b in b_arr, check if there exists c >= b with (b+c) % x == 0
            # (b+c) % x == 0 means c % x == (-b) % x
            # c must be >= b and c in S
            # For each b, we need to check if any element in S that is >= b has the right residue
            for b in b_arr:
                target_res = (-b) % x
                # Check if any element in S that is >= b has residue target_res mod x
                # Elements >= b: elems[elems >= b]
                ge_b = elems[elems >= b]
                if np.any(ge_b % x == target_res):
                    return False

        # Case 2: x is the middle (b=x, a in current_set with a<x, c in current_set with x<=c)
        # Need: NOT exists a,c in S with a < x <= c and (x+c) % a == 0
        mask_a = elems < x
        mask_c = elems >= x
        if np.any(mask_a) and np.any(mask_c):
            a_arr = elems[mask_a]
            c_arr = elems[mask_c]
            # For each a, check if exists c with (x+c) % a == 0
            # (x+c) % a == 0 means c % a == (-x) % a
            for a in a_arr:
                target_res = (-x) % a
                if np.any(c_arr % a == target_res):
                    return False

        # Case 3: x is the largest (c=x, a,b in current_set with a<b<=x)
        # Need: NOT exists a,b in S with a < b <= x and (b+x) % a == 0
        mask3 = elems <= x
        if np.any(mask3):
            le_x = elems[mask3]
            # For each pair a < b in le_x, check (b+x) % a == 0
            # (b+x) % a == 0 means b % a == (-x) % a
            for i in range(len(le_x)):
                a = le_x[i]
                # b must be > a and <= x
                b_arr = le_x[le_x > a]
                if len(b_arr) > 0:
                    target_res = (-x) % a
                    if np.any(b_arr % a == target_res):
                        return False

        return True

    # Phase 1: Greedy addition from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline - 15:
            break
        if x not in s and can_add(x, s):
            s.add(x)

    best_set = set(s)
    best_size = len(s)

    # Phase 2: Try different greedy orderings
    orderings = [
        list(range(170, 0, -1)),
        list(range(1, 171)),
        list(range(1, 171, 2)),  # odd numbers first
        list(range(2, 171, 2)),  # even numbers first
        list(range(170, 0, -2)),  # even from top
        list(range(169, 0, -2)),  # odd from top
    ]
    # Add random orderings
    rng = random.Random(42)
    for _ in range(10):
        orderings.append(rng.sample(range(1, 171), 170))

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        s_trial = set(range(171, 257))
        for x in ordering:
            if x not in s_trial and can_add(x, s_trial):
                s_trial.add(x)
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Phase 3: Local search - try swaps to improve
    # Try removing one element and adding another to increase size
    for iteration in range(200):
        if time.time() > deadline - 5:
            break

        s_trial = set(best_set)
        improved = False

        # Try to remove a small element and add a different one
        small_elements = sorted([x for x in s_trial if x < 171])
        if not small_elements:
            break

        for rem in small_elements:
            if time.time() > deadline - 5:
                break
            s_trial.discard(rem)
            # Try to add elements not in s_trial
            candidates = [x for x in range(1, 171) if x not in s_trial]
            for add in candidates:
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                        break
                    s_trial.discard(add)
            # Restore
            s_trial.add(rem)
            if improved:
                break

        if not improved:
            break

    # Phase 4: More aggressive local search - try multiple removals
    for iteration in range(20):
        if time.time() > deadline - 3:
            break

        s_trial = set(best_set)
        small_elements = sorted([x for x in s_trial if x < 171])
        if len(s_trial) < 10:
            break

        # Try removing 1-2 small elements
        removals = []
        for _ in range(random.randint(1, 2)):
            if small_elements:
                removals.append(random.choice(small_elements))
                small_elements.remove(removals[-1])

        for rem in removals:
            s_trial.discard(rem)

        # Try to add as many as possible
        candidates = sorted([x for x in range(1, 171) if x not in s_trial], reverse=True)
        for add in candidates:
            if time.time() > deadline - 3:
                break
            if can_add(add, s_trial):
                s_trial.add(add)

        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Phase 5: Final polish - try to add any remaining elements
    s_final = set(best_set)
    for x in range(1, 171):
        if time.time() > deadline - 1:
            break
        if x not in s_final and can_add(x, s_final):
            s_final.add(x)

    if len(s_final) > best_size:
        best_set = s_final

    # Phase 6: Final local search pass with the polished set
    for iteration in range(50):
        if time.time() > deadline - 0.5:
            break
        s_trial = set(best_set)
        small_elements = sorted([x for x in s_trial if x < 171])
        if not small_elements:
            break
        improved = False
        for rem in small_elements:
            if time.time() > deadline - 0.5:
                break
            s_trial.discard(rem)
            candidates = [x for x in range(1, 171) if x not in s_trial]
            for add in candidates:
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                        break
                    s_trial.discard(add)
            s_trial.add(rem)
            if improved:
                break
        if not improved:
            break

    return sorted(best_set)
