"""Combined program: distance-2 independent set init + greedy repair + simulated annealing polish."""
# EVOLVE-BLOCK-START
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    total = q ** n

    # Precompute neighbors (Hamming distance <= radius)
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]

    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        temp = idx
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)

    # Build Distance-2 Graph for Independent Set Initialization
    dist2_adj = [set() for _ in range(total)]
    for idx in range(total):
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
                            dist2_adj[new_idx].add(idx)

    rng = random.Random(42)

    # Greedy Independent Set on distance-2 graph
    candidates = list(range(total))
    rng.shuffle(candidates)

    code_set = set()
    in_set = [False] * total

    for v in candidates:
        if in_set[v]:
            continue
        is_conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                is_conflict = True
                break
        if not is_conflict:
            code_set.add(v)
            in_set[v] = True

    code = list(code_set)

    # Repair to Cover
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1

    uncovered = [i for i in range(total) if counts[i] == 0]

    while uncovered and time.time() < deadline - 5.0:
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c

        if best_c == -1:
            break

        code_set.add(best_c)
        code.append(best_c)
        for u in neigh[best_c]:
            counts[u] += 1

        uncovered = [i for i in range(total) if counts[i] == 0]

    if uncovered:
        for u in uncovered:
            if u not in code_set:
                code_set.add(u)
                code.append(u)
                for v in neigh[u]:
                    counts[v] += 1

    best_code = code[:]
    best_size = len(code)

    # Simulated Annealing
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01

    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c):
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp

        move_type = rng.random()

        if move_type < 0.5:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    code.pop(idx)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

        elif move_type < 0.9:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)

                if new_c == old_c or new_c in code_set:
                    continue

                if is_valid_swap(old_c, new_c):
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    # Accept with probability if it doesn't break coverage
                    # Check if the swap would leave any uncovered
                    uncovered_after = 0
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in neigh_set[new_c]:
                            uncovered_after += 1
                    if uncovered_after == 0:
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]

        else:
            # Try Insert (if we're above some minimum)
            if len(code) < total and rng.random() < 0.3:
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1

        temp *= cool_rate

    # Final polish: try to remove redundant codewords
    for _ in range(100):
        removed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if is_removable(c):
                code.pop(i)
                code_set.discard(c)
                for u in neigh[c]:
                    counts[u] -= 1
                removed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
        if not removed:
            break

    # Convert to list of lists
    result = []
    for c in best_code:
        word = []
        temp_val = c
        for i in range(n):
            word.append(temp_val % q)
            temp_val //= q
        result.append(word)

    return result
# EVOLVE-BLOCK-END
