import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def is_valid_set(S):
        S_list = sorted(S)
        size = len(S_list)
        for i in range(size):
            a = S_list[i]
            for j in range(i + 1, size):
                b = S_list[j]
                for k in range(j, size):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def check_add(S_set, S_sorted, x):
        if not S_sorted:
            return True

        size = len(S_sorted)

        # Role 1: x is 'a'
        b_start = 0
        while b_start < size and S_sorted[b_start] <= x:
            b_start += 1

        for i in range(b_start, size):
            b = S_sorted[i]
            target_residue = (-b) % x
            for j in range(i, size):
                c = S_sorted[j]
                if c % x == target_residue:
                    return False

        # Role 2: x is 'b'
        for i in range(size):
            a = S_sorted[i]
            if a >= x:
                break
            for j in range(size):
                c = S_sorted[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Role 3: x is 'c'
        for i in range(size):
            a = S_sorted[i]
            for j in range(i + 1, size):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False

        return True

    best_set = set()
    best_size = 0

    # Try multiple starting thresholds
    for t in range(171, 130, -1):
        if time.time() > deadline - 15:
            break

        current_set = set(range(t, n + 1))
        if not is_valid_set(current_set):
            continue

        S_sorted = sorted(current_set)

        # Greedy addition
        candidates = list(range(t - 1, 0, -1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline - 5:
                break
            if check_add(current_set, S_sorted, x):
                current_set.add(x)
                S_sorted = sorted(current_set)

        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False

            # 1. Try to add any element
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if check_add(current_set, S_sorted, x):
                    current_set.add(x)
                    S_sorted = sorted(current_set)
                    improved = True
                    break

            if improved:
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                continue

            # 2. Try to swap
            # Prefer removing smaller elements and adding larger ones
            S_list = sorted(current_set)
            # Sort by value, we want to remove small ones first?
            # Let's try a few random swaps biased towards removing small y
            for _ in range(20):  # Limit swaps per iteration
                if time.time() > deadline - 5:
                    break
                
                y = random.choice(S_list[:min(20, len(S_list))]) # Pick a small element to remove
                temp_set = set(current_set)
                temp_set.remove(y)
                temp_sorted = sorted(temp_set)
                
                # Find a valid x to add
                # Prefer x > y
                candidates_for_swap = [x for x in range(1, n + 1) if x not in temp_set and x > y]
                if not candidates_for_swap:
                    candidates_for_swap = [x for x in range(1, n + 1) if x not in temp_set]
                
                random.shuffle(candidates_for_swap)
                
                for x in candidates_for_swap:
                    if check_add(temp_set, temp_sorted, x):
                        current_set = temp_set
                        current_set.add(x)
                        S_sorted = sorted(current_set)
                        improved = True
                        break
                
                if improved:
                    break

    if best_set and not is_valid_set(best_set):
        best_set = set(range(171, n + 1))
        if not is_valid_set(best_set):
            best_set = {256}

    return sorted(list(best_set))
