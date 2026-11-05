import itertools
import time
import sys
import random
import math

def construct(n):
    start_time = time.time()
    timeout = 110.0  # 40s margin under 150s wall-clock

    def elapsed():
        return time.time() - start_time

    # Classic recursive superpermutation
    def classic_superpermutation(n):
        if n == 1:
            return [1]
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    if n <= 5:
        cw = classic_superpermutation(n)
        return "".join(map(str, cw))

    # For n=6,7: optimization via permutation ordering
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)

    # Precompute overlap matrix efficiently
    # overlap[i][j] = max k in [1, n-1] such that suffix of perms[i] length k == prefix of perms[j] length k
    overlap = [[0] * N for _ in range(N)]

    # For each k from n-1 down to 1, map prefix of length k -> list of indices
    for k in range(n - 1, 0, -1):
        prefix_to_indices = {}
        for j in range(N):
            pref = perms[j][:k]
            if pref not in prefix_to_indices:
                prefix_to_indices[pref] = []
            prefix_to_indices[pref].append(j)
        for i in range(N):
            suff = perms[i][n - k:]
            if suff in prefix_to_indices:
                for j in prefix_to_indices[suff]:
                    if overlap[i][j] == 0:
                        overlap[i][j] = k

    # Energy = total length = n + sum over consecutive pairs of (n - overlap)
    # = N*n - sum of overlaps

    def compute_energy(order):
        total_ov = 0
        for i in range(N - 1):
            total_ov += overlap[order[i]][order[i + 1]]
        return N * n - total_ov

    # Build word from ordering
    def build_word(order):
        word = list(perms[order[0]])
        for i in range(N - 1):
            ov = overlap[order[i]][order[i + 1]]
            word.extend(perms[order[i + 1]][ov:])
        return word

    # Greedy construction (nearest neighbor)
    def greedy_order(start=None):
        if start is None:
            start = 0
        visited = [False] * N
        order = [start]
        visited[start] = True
        current = start
        for _ in range(N - 1):
            best_j = -1
            best_ov = 0
            for k in range(n - 1, 0, -1):
                suff = perms[current][n - k:]
                # Find unvisited j with prefix == suff of length k
                # We need a fast lookup: for each k, suffix -> list of j's
                # But we already have prefix_to_indices built above... let me just search
                for j in range(N):
                    if not visited[j] and perms[j][:k] == suff:
                        best_j = j
                        best_ov = k
                        break
                if best_j != -1:
                    break
            if best_j == -1:
                # Pick first unvisited
                for j in range(N):
                    if not visited[j]:
                        best_j = j
                        break
            visited[best_j] = True
            order.append(best_j)
            current = best_j
        return order

    # Faster greedy using precomputed maps
    # Build suffix lookup: for each k and each suffix value, list of i's with that suffix
    suffix_to_indices = [{} for _ in range(n)]
    for k in range(1, n):
        for i in range(N):
            suff = perms[i][n - k:]
            if suff not in suffix_to_indices[k]:
                suffix_to_indices[k][suff] = []
            suffix_to_indices[k][suff].append(i)

    def fast_greedy(start=0):
        visited = [False] * N
        order = [start]
        visited[start] = True
        current = start
        for _ in range(N - 1):
            best_j = -1
            best_ov = 0
            for k in range(n - 1, 0, -1):
                suff = perms[current][n - k:]
                candidates = suffix_to_indices[k].get(suff, [])
                for j in candidates:
                    if not visited[j]:
                        best_j = j
                        best_ov = k
                        break
                if best_j != -1:
                    break
            if best_j == -1:
                for j in range(N):
                    if not visited[j]:
                        best_j = j
                        break
            visited[best_j] = True
            order.append(best_j)
            current = best_j
        return order

    # Simulated Annealing with fast delta
    def run_sa(start_order, time_budget=30.0, T_start=10.0, T_end=0.01, alpha=0.9998):
        current_order = start_order[:]
        current_energy = compute_energy(current_order)
        best_order = current_order[:]
        best_energy = current_energy
        T = T_start

        # Precompute edge costs: cost[i][j] = n - overlap[i][j]
        # Actually we already have overlap, so cost = n - overlap
        # For delta computation we need to be careful

        def get_cost(a, b):
            return n - overlap[a][b]

        step_count = 0
        while elapsed() < start_time + time_budget:
            step_count += 1

            move_type = random.random()

            if move_type < 0.5:
                # Swap two non-adjacent positions
                i = random.randint(0, N - 2)
                j = random.randint(i + 2, N - 1)

                a = current_order[i]
                b = current_order[j]

                # Old edges affected: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
                delta = 0
                if i > 0:
                    delta += get_cost(current_order[i-1], a)
                if i < N - 1:
                    delta += get_cost(a, current_order[i+1])
                if j > 0:
                    delta += get_cost(current_order[j-1], b)
                if j < N - 1:
                    delta += get_cost(b, current_order[j+1])

                # After swap: position i has b, position j has a
                if i > 0:
                    delta -= get_cost(current_order[i-1], b)
                if i < N - 1:
                    delta -= get_cost(b, current_order[i+1])
                if j > 0:
                    delta -= get_cost(current_order[j-1], a)
                if j < N - 1:
                    delta -= get_cost(a, current_order[j+1])

                new_energy = current_energy + delta
                if new_energy < current_energy or (T > T_end and random.random() < math.exp(-delta / T)):
                    current_order[i], current_order[j] = b, a
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_order = current_order[:]
                else:
                    pass  # no change

            elif move_type < 0.8:
                # Relocate: move element at position i to position j
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if abs(i - j) <= 1:
                    continue

                elem = current_order[i]

                # Remove from i
                delta = 0
                if i > 0:
                    delta += get_cost(current_order[i-1], elem)
                if i < N - 1:
                    delta += get_cost(elem, current_order[i+1])
                if 0 < i < N - 1:
                    # New edge between i-1 and i+1
                    delta -= get_cost(current_order[i-1], current_order[i+1])

                # Adjust j for removal
                if j > i:
                    j -= 1

                # Insert at j
                if j > 0:
                    delta += get_cost(current_order[j-1], elem)
                if j < N - 1:
                    delta += get_cost(elem, current_order[j])
                if 0 < j < N - 1:
                    # Remove edge between j-1 and j
                    delta -= get_cost(current_order[j-1], current_order[j])

                new_energy = current_energy + delta
                if new_energy < current_energy or (T > T_end and random.random() < math.exp(-delta / T)):
                    # Perform move
                    current_order.pop(i)
                    current_order.insert(j, elem)
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_order = current_order[:]
                else:
                    pass

            else:
                # Reverse a segment
                i = random.randint(0, N - 3)
                j = random.randint(i + 2, N - 1)

                # Edges affected: all internal edges in [i,j] plus boundary edges
                delta = 0
                # Old boundary edges
                if i > 0:
                    delta += get_cost(current_order[i-1], current_order[i])
                if j < N - 1:
                    delta += get_cost(current_order[j], current_order[j+1])
                # Old internal edges
                for k in range(i, j):
                    delta += get_cost(current_order[k], current_order[k+1])
                # New boundary edges
                if i > 0:
                    delta -= get_cost(current_order[i-1], current_order[j])
                if j < N - 1:
                    delta -= get_cost(current_order[i], current_order[j+1])
                # New internal edges (reversed)
                for k in range(j, i, -1):
                    delta -= get_cost(current_order[k], current_order[k-1])

                new_energy = current_energy + delta
                if new_energy < current_energy or (T > T_end and random.random() < math.exp(-delta / T)):
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_order = current_order[:]
                else:
                    pass

            T *= alpha
            if T < T_end:
                T = T_end

        return best_order, best_energy

    # Get classic baseline
    classic_word = classic_superpermutation(n)
    classic_len = len(classic_word)

    best_len = classic_len
    best_result = "".join(map(str, classic_word))

    try:
        # Initial greedy ordering
        random.seed(42)
        g_order = fast_greedy(0)
        g_energy = compute_energy(g_order)

        if g_energy < classic_len:
            g_word = build_word(g_order)
            best_len = g_energy
            best_result = "".join(map(str, g_word))

        # Run SA with multiple restarts
        total_time_budget = min(90.0, timeout - 15.0)
        sa_time_per_run = total_time_budget / 3.0

        # Restart 1: from greedy
        sa_order1, sa_energy1 = run_sa(g_order, time_budget=sa_time_per_run)
        if sa_energy1 < best_len:
            best_len = sa_energy1
            best_result = "".join(map(str, build_word(sa_order1)))

        # Restart 2: from a different greedy start
        random.seed(123)
        g_order2 = fast_greedy(random.randint(0, N - 1))
        sa_order2, sa_energy2 = run_sa(g_order2, time_budget=sa_time_per_run)
        if sa_energy2 < best_len:
            best_len = sa_energy2
            best_result = "".join(map(str, build_word(sa_order2)))

        # Restart 3: from perturbed best
        if elapsed() < start_time + total_time_budget - sa_time_per_run:
            random.seed(777)
            perturbed = sa_order1[:]
            # Apply some random swaps
            for _ in range(20):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            sa_order3, sa_energy3 = run_sa(perturbed, time_budget=sa_time_per_run)
            if sa_energy3 < best_len:
                best_len = sa_energy3
                best_result = "".join(map(str, build_word(sa_order3)))

    except Exception:
        pass

    return best_result
