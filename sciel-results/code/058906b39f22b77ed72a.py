import time
import math
import random

def construct():
    t_start = time.time()
    DEADLINE = 85.0
    n = 256

    def is_valid(S_list):
        S_list = sorted(S_list)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c in S_list[j:]:
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S_sorted, x):
        """Check if x can be added to S_sorted without violating constraints."""
        residues = {}
        for v in S_sorted:
            r = v % x
            residues[r] = residues.get(r, 0) + 1
        for r, cnt in residues.items():
            complement = (x - r) % x
            if r == 0:
                if cnt >= 2:
                    return False
            else:
                if complement in residues and residues[complement] >= 1:
                    return False
        return True

    base_S = list(range(171, 257))
    best_solution = list(base_S)
    best_size = len(base_S)

    # Strategy 1: Greedy add from 170 down to 1
    S1 = list(base_S)
    S1_sorted = sorted(S1)
    for x in range(170, 0, -1):
        if time.time() - t_start > DEADLINE:
            break
        if can_add(S1_sorted, x):
            S1.append(x)
            S1_sorted = sorted(S1)

    if len(S1) > best_size:
        best_size = len(S1)
        best_solution = list(S1)

    # Strategy 2: Try different base intervals
    for k in range(155, 178):
        if time.time() - t_start > DEADLINE:
            break
        S2 = list(range(k, n + 1))
        S2_sorted = sorted(S2)
        for x in range(k - 1, 0, -1):
            if time.time() - t_start > DEADLINE:
                break
            if can_add(S2_sorted, x):
                S2.append(x)
                S2_sorted = sorted(S2)
        if len(S2) > best_size:
            best_size = len(S2)
            best_solution = list(S2)

    # Strategy 3: Local search with swaps and random restarts
    random.seed(42)
    for restart in range(40):
        if time.time() - t_start > DEADLINE:
            break

        # Start from best or a perturbation
        S_cur = list(best_solution)
        if restart > 0 and len(S_cur) > 15:
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, len(S_cur) - 1)
                S_cur.pop(idx)

        S_cur_sorted = sorted(S_cur)

        # Phase 1: try to add any missing element
        changed = True
        while changed and time.time() - t_start < DEADLINE:
            changed = False
            S_cur_set = set(S_cur)
            for x in range(1, n + 1):
                if time.time() - t_start > DEADLINE:
                    break
                if x not in S_cur_set:
                    if can_add(S_cur_sorted, x):
                        S_cur.append(x)
                        S_cur_set.add(x)
                        S_cur_sorted = sorted(S_cur)
                        changed = True
                        break

        # Phase 2: try swaps to improve
        for _ in range(25):
            if time.time() - t_start > DEADLINE:
                break
            S_cur_set = set(S_cur)
            S_cur_sorted = sorted(S_cur)

            if len(S_cur) <= 10:
                break

            # Pick a random element to remove
            y = random.choice(S_cur)
            S_temp = [v for v in S_cur if v != y]
            S_temp_set = set(S_temp)
            S_temp_sorted = sorted(S_temp)

            # Try to add an element
            best_add = None
            for x in range(1, n + 1):
                if time.time() - t_start > DEADLINE:
                    break
                if x not in S_temp_set and x != y:
                    if can_add(S_temp_sorted, x):
                        best_add = x
                        break

            if best_add is not None and len(S_temp) + 1 >= len(S_cur):
                S_cur = S_temp + [best_add]

        if len(S_cur) > best_size:
            best_size = len(S_cur)
            best_solution = list(S_cur)

    S_final = sorted(best_solution)
    if not is_valid(S_final):
        S_final = list(base_S)

    return S_final
