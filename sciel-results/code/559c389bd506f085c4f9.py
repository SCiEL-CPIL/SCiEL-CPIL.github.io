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

    best_size = 0
    best_set = []

    def try_add(x, S_list, counts, in_set):
        """Try to add x to the set. Returns True if successful."""
        if in_set[x]:
            return False
        if counts[2 * x] >= max_rep:
            return False
        for y in S_list:
            if counts[x + y] >= max_rep:
                return False
        # Add it
        in_set[x] = True
        S_list.append(x)
        for y in S_list[:-1]:
            counts[x + y] += 1
        counts[2 * x] += 1
        return True

    def try_remove(x, S_list, counts, in_set):
        """Remove x from the set."""
        idx = S_list.index(x)
        S_list.pop(idx)
        in_set[x] = False
        for y in S_list:
            counts[x + y] -= 1
        counts[2 * x] -= 1

    def greedy_with_score(order, counts, in_set, time_limit):
        """Greedy build with dynamic scoring: prefer elements that use less 'capacity'."""
        S_list = []
        S_set = set()
        
        for x in order:
            if time.time() > time_limit:
                break
            if in_set[x]:
                continue
            # Check validity
            valid = True
            if counts[2 * x] >= max_rep:
                valid = False
            else:
                for y in S_list:
                    if counts[x + y] >= max_rep:
                        valid = False
                        break
            if valid:
                S_list.append(x)
                S_set.add(x)
                in_set[x] = True
                for y in S_list[:-1]:
                    counts[x + y] += 1
                counts[2 * x] += 1
        
        return S_list, S_set, counts, in_set

    def local_search(S_list, S_set, counts, in_set, time_budget):
        """Local search with add and remove-one-add-two moves."""
        improved = True
        stall_count = 0
        while improved and time.time() < time_budget:
            improved = False
            stall_count += 1
            
            # Phase 1: Try to add any candidate
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if candidates:
                random.shuffle(candidates)
                for x in candidates[:50]:
                    if try_add(x, S_list, counts, in_set):
                        improved = True
                        stall_count = 0
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
                old_S_list = list(S_list)
                old_in_set = in_set.copy()

                # Temporarily remove u
                try_remove(u, S_list, counts, in_set)

                # Now try to add two elements
                added = []
                candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
                random.shuffle(candidates)

                # Try to add first element
                for v1 in candidates[:60]:
                    if try_add(v1, S_list, counts, in_set):
                        added.append(v1)
                        break

                # Try to add second element
                if len(added) == 1:
                    candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                    random.shuffle(candidates2)
                    for v2 in candidates2[:60]:
                        if try_add(v2, S_list, counts, in_set):
                            added.append(v2)
                            break

                # If we added 2 elements, net gain is +1
                if len(added) == 2:
                    improved = True
                    stall_count = 0
                    break
                else:
                    # Rollback
                    for v in added:
                        try_remove(v, S_list, counts, in_set)
                    # Restore u
                    counts[:] = old_counts
                    in_set[:] = old_in_set
                    S_list[:] = old_S_list
                    break

        return S_list, S_set, counts, in_set

    def hill_climb(S_list, S_set, counts, in_set, time_budget):
        """More aggressive local search with multiple move types."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            
            # Try all possible additions
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            random.shuffle(candidates)
            for x in candidates[:80]:
                if try_add(x, S_list, counts, in_set):
                    improved = True
                    break
            
            if improved:
                continue

            # Try remove-one-add-two
            if len(S_list) < 2:
                break

            for _ in range(15):
                if time.time() >= time_budget:
                    break
                u_idx = random.randint(0, len(S_list) - 1)
                u = S_list[u_idx]

                old_counts = counts.copy()
                old_S_list = list(S_list)
                old_in_set = in_set.copy()

                try_remove(u, S_list, counts, in_set)

                added = []
                candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
                random.shuffle(candidates)

                for v1 in candidates[:70]:
                    if try_add(v1, S_list, counts, in_set):
                        added.append(v1)
                        break

                if len(added) == 1:
                    candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                    random.shuffle(candidates2)
                    for v2 in candidates2[:70]:
                        if try_add(v2, S_list, counts, in_set):
                            added.append(v2)
                            break

                if len(added) == 2:
                    improved = True
                    break
                else:
                    for v in added:
                        try_remove(v, S_list, counts, in_set)
                    counts[:] = old_counts
                    in_set[:] = old_in_set
                    S_list[:] = old_S_list
                    break

        return S_list, S_set, counts, in_set

    num_restarts = 60

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
            order = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        elif r == 3:
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        elif r == 4:
            # Multiples of 3 first, then others
            order = list(range(3, N + 1, 3)) + [x for x in range(1, N + 1) if x % 3 != 0]
        elif r == 5:
            # Fibonacci-like spacing
            fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
            order = sorted(fibs) + [x for x in range(1, N + 1) if x not in fibs]
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        S_list, S_set, counts, in_set = greedy_with_score(order, counts, in_set, deadline)

        # Local search
        S_list, S_set, counts, in_set = local_search(
            S_list, S_set, counts, in_set, deadline
        )

        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)

    return best_set
