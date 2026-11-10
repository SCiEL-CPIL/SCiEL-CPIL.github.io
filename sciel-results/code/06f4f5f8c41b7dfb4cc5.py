import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute all 4-term arithmetic progressions
    # ap_list: list of tuples (a, b, c, d)
    # aps_containing[i]: list of AP indices containing element i
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]

    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            for v in (a, b, c, e):
                aps_containing[v].append(ap_idx)

    num_aps = len(ap_list)

    # Precompute degree of each element (number of APs it belongs to)
    degree = [len(aps_containing[i]) for i in range(N + 1)]

    best_set = set()
    best_size = 0

    def is_valid_addition(ap_count, x):
        for ap_idx in aps_containing[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True

    def add_element(S, ap_count, x):
        S.add(x)
        for ap_idx in aps_containing[x]:
            ap_count[ap_idx] += 1

    def remove_element(S, ap_count, x):
        S.discard(x)
        for ap_idx in aps_containing[x]:
            ap_count[ap_idx] -= 1

    def greedy_init(order):
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            if is_valid_addition(ap_count, x):
                add_element(S, ap_count, x)
        return S, ap_count

    def local_search(S, ap_count, max_steps=30000):
        current_size = len(S)
        steps_since_improvement = 0

        while steps_since_improvement < max_steps and time.time() < deadline:
            # Try to add a valid element
            valids = []
            for x in range(1, N + 1):
                if x not in S and is_valid_addition(ap_count, x):
                    valids.append(x)

            if valids:
                x = random.choice(valids)
                add_element(S, ap_count, x)
                if len(S) > current_size:
                    current_size = len(S)
                    steps_since_improvement = 0
            else:
                # Stuck: try 1-exchange (remove one, add one)
                found_swap = False
                if S:
                    removable = list(S)
                    random.shuffle(removable)
                    # Limit the number of removals we try per iteration
                    for x_remove in removable[:30]:
                        remove_element(S, ap_count, x_remove)

                        # Try to add any valid element
                        for y_add in range(1, N + 1):
                            if y_add not in S and is_valid_addition(ap_count, y_add):
                                add_element(S, ap_count, y_add)
                                found_swap = True
                                break

                        if found_swap:
                            if len(S) > current_size:
                                current_size = len(S)
                                steps_since_improvement = 0
                            break
                        else:
                            # Restore x_remove
                            add_element(S, ap_count, x_remove)

                if not found_swap:
                    # Try 2-exchange: remove two, add two
                    found_2swap = False
                    if len(S) >= 2:
                        removable = list(S)
                        random.shuffle(removable)
                        # Try more pairs
                        max_i = min(20, len(removable))
                        for i in range(max_i):
                            for j in range(i + 1, min(i + 20, len(removable))):
                                x1 = removable[i]
                                x2 = removable[j]
                                remove_element(S, ap_count, x1)
                                remove_element(S, ap_count, x2)

                                # Try to add two valid elements
                                added = []
                                for y_add in range(1, N + 1):
                                    if y_add not in S and len(added) < 2 and is_valid_addition(ap_count, y_add):
                                        add_element(S, ap_count, y_add)
                                        added.append(y_add)
                                    if len(added) == 2:
                                        break

                                if len(added) == 2:
                                    found_2swap = True
                                    if len(S) > current_size:
                                        current_size = len(S)
                                        steps_since_improvement = 0
                                    break
                                else:
                                    # Restore
                                    for y in added:
                                        remove_element(S, ap_count, y)
                                    add_element(S, ap_count, x1)
                                    add_element(S, ap_count, x2)

                            if found_2swap:
                                break

                    if not found_swap and not found_2swap:
                        # Try 3-exchange: remove three, add three
                        found_3swap = False
                        if len(S) >= 3:
                            removable = list(S)
                            random.shuffle(removable)
                            max_i = min(10, len(removable))
                            for i in range(max_i):
                                for j in range(i + 1, min(i + 10, len(removable))):
                                    for k in range(j + 1, min(j + 10, len(removable))):
                                        x1 = removable[i]
                                        x2 = removable[j]
                                        x3 = removable[k]
                                        remove_element(S, ap_count, x1)
                                        remove_element(S, ap_count, x2)
                                        remove_element(S, ap_count, x3)

                                        added = []
                                        for y_add in range(1, N + 1):
                                            if y_add not in S and len(added) < 3 and is_valid_addition(ap_count, y_add):
                                                add_element(S, ap_count, y_add)
                                                added.append(y_add)
                                            if len(added) == 3:
                                                break

                                        if len(added) == 3:
                                            found_3swap = True
                                            if len(S) > current_size:
                                                current_size = len(S)
                                                steps_since_improvement = 0
                                            break
                                        else:
                                            for y in added:
                                                remove_element(S, ap_count, y)
                                            add_element(S, ap_count, x1)
                                            add_element(S, ap_count, x2)
                                            add_element(S, ap_count, x3)

                                        if found_3swap:
                                            break
                                    if found_3swap:
                                        break
                                if found_3swap:
                                    break

                        if not found_swap and not found_2swap and not found_3swap:
                            steps_since_improvement += 1
                            # If stuck for too long, remove a random element to break deadlock
                            if steps_since_improvement > 150 and S:
                                x_remove = random.choice(list(S))
                                remove_element(S, ap_count, x_remove)
                                steps_since_improvement = 0

        return S, ap_count

    # Multiple restarts with different initialization strategies
    while time.time() < deadline - 5.0:
        strategy = random.random()

        if strategy < 0.25:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy < 0.5:
            # Degree-ordered: prefer elements with fewer APs (less constrained)
            order = list(range(1, N + 1))
            order.sort(key=lambda x: degree[x])
            # Slight randomization
            random.shuffle(order[:50])
        elif strategy < 0.75:
            # Reverse degree order: prefer more constrained elements first
            order = list(range(1, N + 1))
            order.sort(key=lambda x: -degree[x])
            random.shuffle(order[:50])
        else:
            # Random with bias toward middle
            order = list(range(1, N + 1))
            random.shuffle(order)

        S, ap_count = greedy_init(order)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

        # Local search
        S, ap_count = local_search(S, ap_count, max_steps=15000)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

        # If we have time, do another round of local search
        if time.time() < deadline - 10.0:
            S, ap_count = local_search(S, ap_count, max_steps=15000)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()

    # Final polish: deterministic pass
    if time.time() < deadline - 2.0:
        S = set(best_set)
        ap_count = [0] * num_aps
        for x in S:
            for ap_idx in aps_containing[x]:
                ap_count[ap_idx] += 1

        # Try to add every valid element in order
        for x in range(1, N + 1):
            if x not in S and is_valid_addition(ap_count, x):
                add_element(S, ap_count, x)

        # Try systematic 1-swaps
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for x_remove in range(1, N + 1):
                if x_remove not in S:
                    continue
                remove_element(S, ap_count, x_remove)
                for y_add in range(1, N + 1):
                    if y_add not in S and y_add != x_remove and is_valid_addition(ap_count, y_add):
                        add_element(S, ap_count, y_add)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S.copy()
                        improved = True
                        break
                if not improved:
                    # Restore x_remove if we didn't add anything
                    if x_remove not in S:
                        add_element(S, ap_count, x_remove)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return sorted(best_set)
