import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 82  # 8s buffer before 90s limit

    # Pre-compute all 4-APs
    aps = []
    element_to_aps = [[] for _ in range(N)]

    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            element_to_aps[a].append(ap_idx)
            element_to_aps[b].append(ap_idx)
            element_to_aps[c].append(ap_idx)
            element_to_aps[e].append(ap_idx)

    num_aps = len(aps)

    # Helper functions for state management
    def init_state():
        in_set = np.zeros(N, dtype=np.bool_)
        ap_count = np.zeros(num_aps, dtype=np.int32)
        return in_set, ap_count

    def add_element(x, in_set, ap_count):
        for ap_idx in element_to_aps[x]:
            if ap_count[ap_idx] == 3:
                return False
        in_set[x] = True
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] += 1
        return True

    def remove_element(x, in_set, ap_count):
        in_set[x] = False
        for ap_idx in element_to_aps[x]:
            ap_count[ap_idx] -= 1

    def get_size(in_set):
        return np.sum(in_set)

    def try_add_all(in_set, ap_count):
        """Try to add all possible elements greedily"""
        added = 0
        for x in range(N):
            if not in_set[x]:
                if add_element(x, in_set, ap_count):
                    added += 1
        return added

    # 1. Initial Greedy Pass with multiple restarts
    best_in_set = None
    best_ap_count = None
    best_size = 0

    # Run multiple greedy passes with different random orders
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        in_set, ap_count = init_state()
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if add_element(x, in_set, ap_count):
                pass

        size = get_size(in_set)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            best_ap_count = ap_count.copy()

    # 2. Local Search with Kicks, Swaps, and Systematic Improvement
    current_in_set = best_in_set
    current_ap_count = best_ap_count
    current_size = best_size

    no_improve_count = 0
    max_no_improve = 100
    iteration = 0

    while time.time() < deadline - 5:
        iteration += 1

        # Try to add any available element (greedy improvement)
        improved = try_add_all(current_in_set, current_ap_count) > 0
        if improved:
            current_size = get_size(current_in_set)
            no_improve_count = 0
            continue

        no_improve_count += 1
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            max_no_improve = min(max_no_improve + 30, 500)

        # Choose operation: kick, swap, or pair removal
        op = random.random()
        current_elements = np.where(current_in_set)[0]
        non_elements = np.where(~current_in_set)[0]

        if op < 0.4 and len(current_elements) >= 2:
            # Kick: remove k random elements, then greedily add back
            k = random.choice([1, 2, 3])
            if len(current_elements) < k:
                continue
            removed_elements = random.sample(current_elements.tolist(), k)

            old_in_set = current_in_set.copy()
            old_ap_count = current_ap_count.copy()

            for x in removed_elements:
                remove_element(x, current_in_set, current_ap_count)

            try_add_all(current_in_set, current_ap_count)
            new_size = get_size(current_in_set)

            if new_size >= current_size:
                current_size = new_size
                no_improve_count = 0
            else:
                current_in_set = old_in_set
                current_ap_count = old_ap_count

        elif op < 0.7 and len(current_elements) >= 1 and len(non_elements) >= 1:
            # 1-for-1 swap: remove one element, try to add another
            x_out = int(random.choice(current_elements))
            x_in = int(random.choice(non_elements))

            old_in_set = current_in_set.copy()
            old_ap_count = current_ap_count.copy()

            remove_element(x_out, current_in_set, current_ap_count)

            if add_element(x_in, current_in_set, current_ap_count):
                try_add_all(current_in_set, current_ap_count)
                new_size = get_size(current_in_set)
                if new_size > current_size:
                    current_size = new_size
                    no_improve_count = 0
                else:
                    current_in_set = old_in_set
                    current_ap_count = old_ap_count
            else:
                current_in_set = old_in_set
                current_ap_count = old_ap_count

        elif op < 0.9 and len(current_elements) >= 2:
            # Pair removal: remove 2 elements, greedily add back
            x1, x2 = random.sample(current_elements.tolist(), 2)

            old_in_set = current_in_set.copy()
            old_ap_count = current_ap_count.copy()

            remove_element(x1, current_in_set, current_ap_count)
            remove_element(x2, current_in_set, current_ap_count)

            try_add_all(current_in_set, current_ap_count)
            new_size = get_size(current_in_set)

            if new_size > current_size:
                current_size = new_size
                no_improve_count = 0
            else:
                current_in_set = old_in_set
                current_ap_count = old_ap_count

        else:
            # Larger kick: remove 4-5 elements
            k = random.choice([4, 5])
            if len(current_elements) < k:
                k = max(1, len(current_elements) - 1)
            removed_elements = random.sample(current_elements.tolist(), k)

            old_in_set = current_in_set.copy()
            old_ap_count = current_ap_count.copy()

            for x in removed_elements:
                remove_element(x, current_in_set, current_ap_count)

            try_add_all(current_in_set, current_ap_count)
            new_size = get_size(current_in_set)

            if new_size >= current_size:
                current_size = new_size
                no_improve_count = 0
            else:
                current_in_set = old_in_set
                current_ap_count = old_ap_count

        # Early exit if we've found a very large set
        if current_size > 260:
            break

    # 3. Polish phase: systematic local search with tighter budget
    polish_deadline = start_time + 85  # 5s before hard deadline

    while time.time() < polish_deadline:
        # Try all possible 1-for-1 swaps
        current_elements = np.where(current_in_set)[0]
        non_elements = np.where(~current_in_set)[0]

        if len(current_elements) == 0 or len(non_elements) == 0:
            break

        found_improvement = False
        for x_out in current_elements:
            if time.time() > polish_deadline:
                break
            for x_in in non_elements:
                if time.time() > polish_deadline:
                    break

                old_in_set = current_in_set.copy()
                old_ap_count = current_ap_count.copy()

                remove_element(int(x_out), current_in_set, current_ap_count)

                if add_element(int(x_in), current_in_set, current_ap_count):
                    try_add_all(current_in_set, current_ap_count)
                    new_size = get_size(current_in_set)
                    if new_size > current_size:
                        current_size = new_size
                        found_improvement = True
                        break
                    else:
                        current_in_set = old_in_set
                        current_ap_count = old_ap_count
                else:
                    current_in_set = old_in_set
                    current_ap_count = old_ap_count

            if found_improvement:
                break

        if not found_improvement:
            # Try pair removals
            found_pair_improvement = False
            for i in range(len(current_elements)):
                if time.time() > polish_deadline:
                    break
                for j in range(i + 1, len(current_elements)):
                    if time.time() > polish_deadline:
                        break

                    x1 = int(current_elements[i])
                    x2 = int(current_elements[j])

                    old_in_set = current_in_set.copy()
                    old_ap_count = current_ap_count.copy()

                    remove_element(x1, current_in_set, current_ap_count)
                    remove_element(x2, current_in_set, current_ap_count)

                    try_add_all(current_in_set, current_ap_count)
                    new_size = get_size(current_in_set)

                    if new_size > current_size:
                        current_size = new_size
                        found_pair_improvement = True
                        break
                    else:
                        current_in_set = old_in_set
                        current_ap_count = old_ap_count

                if found_pair_improvement:
                    break

            if not found_pair_improvement:
                break

    # Final polish: try to add any remaining elements
    try_add_all(current_in_set, current_ap_count)
    current_size = get_size(current_in_set)

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if current_in_set[i]]
    return sorted(result)
