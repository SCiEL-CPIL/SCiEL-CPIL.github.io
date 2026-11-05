import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    N = 1000
    max_rep = 2
    MAX_SUM = 2 * N  # 2000

    def greedy_build(order, counts, in_set):
        """Greedy: add elements from order if valid."""
        S_list = []
        S_set = set()
        for x in order:
            if in_set[x]:
                continue
            # Check validity using numpy for speed
            if counts[2 * x] >= max_rep:
                continue
            if S_list:
                # Check x + y for all y in S_list
                sums = x + np.array(S_list, dtype=np.int32)
                if np.any(counts[sums] >= max_rep):
                    continue
            S_list.append(x)
            S_set.add(x)
            in_set[x] = True
            # Update counts
            if S_list[:-1]:
                counts[x + np.array(S_list[:-1], dtype=np.int32)] += 1
            counts[2 * x] += 1
        return S_list, S_set, counts, in_set

    def local_search(S_list, S_set, counts, in_set, time_budget):
        """Local search: try to add elements, then remove-one-add-two."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Phase 1: Try to add any candidate
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if not candidates:
                break
            random.shuffle(candidates)
            for x in candidates[:50]:
                if time.time() >= time_budget:
                    break
                if counts[2 * x] >= max_rep:
                    continue
                if S_list:
                    sums = x + np.array(S_list, dtype=np.int32)
                    if np.any(counts[sums] >= max_rep):
                        continue
                S_list.append(x)
                S_set.add(x)
                in_set[x] = True
                if S_list[:-1]:
                    counts[x + np.array(S_list[:-1], dtype=np.int32)] += 1
                counts[2 * x] += 1
                improved = True
                break

            if improved:
                continue

            # Phase 2: Remove-one-add-two
            if len(S_list) < 2:
                break

            # Try a few random removals
            for _ in range(10):
                if time.time() >= time_budget:
                    break
                u_idx = random.randint(0, len(S_list) - 1)
                u = S_list[u_idx]

                # Save state
                old_counts = counts.copy()
                old_in_set = in_set.copy()
                old_S_list = S_list.copy()
                old_S_set = S_set.copy()

                # Temporarily remove u
                in_set[u] = False
                S_list.pop(u_idx)
                S_set.discard(u)
                if S_list:
                    counts[u + np.array(S_list, dtype=np.int32)] -= 1
                counts[2 * u] -= 1

                # Now try to add two elements
                added = []
                candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
                random.shuffle(candidates)

                # Try to add first element
                for v1 in candidates[:60]:
                    if time.time() >= time_budget:
                        break
                    if counts[2 * v1] >= max_rep:
                        continue
                    if S_list:
                        sums = v1 + np.array(S_list, dtype=np.int32)
                        if np.any(counts[sums] >= max_rep):
                            continue
                    S_list.append(v1)
                    S_set.add(v1)
                    in_set[v1] = True
                    if S_list[:-1]:
                        counts[v1 + np.array(S_list[:-1], dtype=np.int32)] += 1
                    counts[2 * v1] += 1
                    added.append(v1)
                    break

                # Try to add second element
                if len(added) == 1:
                    candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                    random.shuffle(candidates2)
                    for v2 in candidates2[:60]:
                        if time.time() >= time_budget:
                            break
                        if counts[2 * v2] >= max_rep:
                            continue
                        if S_list:
                            sums = v2 + np.array(S_list, dtype=np.int32)
                            if np.any(counts[sums] >= max_rep):
                                continue
                        S_list.append(v2)
                        S_set.add(v2)
                        in_set[v2] = True
                        if S_list[:-1]:
                            counts[v2 + np.array(S_list[:-1], dtype=np.int32)] += 1
                        counts[2 * v2] += 1
                        added.append(v2)
                        break

                # If we added 2 elements, net gain is +1 (removed 1, added 2)
                if len(added) == 2:
                    improved = True
                    break
                else:
                    # Rollback: restore original state
                    counts[:] = old_counts
                    in_set[:] = old_in_set
                    S_list = old_S_list
                    S_set = old_S_set
                    break  # Skip remaining removals this iteration

        return S_list, S_set, counts, in_set

    best_size = 0
    best_set = []

    num_restarts = 150  # More restarts with faster numpy-based checks

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Reset state
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        in_set = np.zeros(N + 1, dtype=bool)

        if r == 0:
            order = list(range(1, N + 1))
        elif r == 1:
            order = list(range(N, 0, -1))
        elif r == 2:
            # Even then odd
            order = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        elif r == 3:
            # Odd then even
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        elif r == 4:
            # Random with bias toward smaller numbers
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: random.random() * x)
        elif r == 5:
            # Random with bias toward larger numbers
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: random.random() * (N - x + 1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        S_list, S_set, counts, in_set = greedy_build(order, counts, in_set)

        # Local search
        S_list, S_set, counts, in_set = local_search(
            S_list, S_set, counts, in_set, deadline
        )

        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)

    return best_set
