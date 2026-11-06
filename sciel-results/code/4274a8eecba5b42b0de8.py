import time
import random
import math
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 128
    numbers = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict pairs: for each x, store set of (a,b) pairs that form bad triple with x
    conflict_pairs = {i: set() for i in range(1, N + 1)}

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))

    # Convert to lists for faster iteration
    conflict_lists = {i: list(conflict_pairs[i]) for i in range(1, N + 1)}

    # Conflict degree: number of conflict pairs for each element
    conflict_degree = {i: len(conflict_lists[i]) for i in range(1, N + 1)}

    # Fast validity check: can we add x to set S (represented as boolean array)?
    def can_add(x, S_bool):
        for (a, b) in conflict_lists[x]:
            if S_bool[a] and S_bool[b]:
                return False
        return True

    # Greedy with different orderings
    def greedy_construct(ordering='random'):
        S = set()
        S_bool = [False] * (N + 1)

        if ordering == 'random':
            candidates = numbers[:]
            random.shuffle(candidates)
        elif ordering == 'ascending':
            candidates = numbers[:]
        elif ordering == 'descending':
            candidates = list(reversed(numbers))
        elif ordering == 'low_conflict':
            # Elements with fewer conflict pairs first
            candidates = sorted(numbers, key=lambda x: conflict_degree[x])
            random.shuffle([x for x in candidates if conflict_degree[x] == conflict_degree[candidates[0]]])
            # Actually just sort by degree, break ties randomly
            candidates = sorted(numbers, key=lambda x: (conflict_degree[x], random.random()))
        elif ordering == 'high_conflict':
            # Elements with more conflict pairs first (they're more constrained, place them early)
            candidates = sorted(numbers, key=lambda x: (-conflict_degree[x], random.random()))
        elif ordering == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in numbers if is_prime(x)]
            composites = [x for x in numbers if not is_prime(x)]
            random.shuffle(primes)
            random.shuffle(composites)
            candidates = primes + composites
        else:
            candidates = numbers[:]
            random.shuffle(candidates)

        for x in candidates:
            if can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True

        return S

    # Local search with systematic add + random swaps
    def local_search(S, max_iters=20000, time_budget=None):
        S = set(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True

        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        no_improve = 0
        local_start = time.time()

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if time_budget and (time.time() - local_start) > time_budget:
                break

            # Phase A: Try to add any element (systematic, not random)
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]

            for x in not_in_S:
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    added = True
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    break

            if not added:
                no_improve += 1

                # Phase B: Try random swaps
                if S and not_in_S:
                    improved = False
                    # Try up to 30 random single swaps
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        u = random.choice(list(S))
                        v = random.choice(not_in_S)

                        S_bool[u] = False
                        S.remove(u)

                        if can_add(v, S_bool):
                            S.add(v)
                            S_bool[v] = True
                            improved = True
                            # After a swap, try to add more
                            for x in not_in_S:
                                if x != v and can_add(x, S_bool):
                                    S.add(x)
                                    S_bool[x] = True
                                    current_size += 1
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_S = S.copy()
                                    break
                        else:
                            S.add(u)
                            S_bool[u] = True

                        if improved:
                            no_improve = 0
                            break

                    # Phase C: Try double swaps if stuck
                    if not improved and len(S) >= 2 and len(not_in_S) >= 2:
                        for _ in range(15):
                            if time.time() > deadline:
                                break
                            u1, u2 = random.sample(list(S), 2)
                            v1, v2 = random.sample(not_in_S, 2)

                            S_bool[u1] = False
                            S_bool[u2] = False
                            S.remove(u1)
                            S.remove(u2)

                            added_v1 = False
                            added_v2 = False
                            if can_add(v1, S_bool):
                                S.add(v1)
                                S_bool[v1] = True
                                added_v1 = True
                                if can_add(v2, S_bool):
                                    S.add(v2)
                                    S_bool[v2] = True
                                    added_v2 = True
                                    improved = True
                                    no_improve = 0
                                    # Try to add more after double swap
                                    for x in not_in_S:
                                        if x != v1 and x != v2 and can_add(x, S_bool):
                                            S.add(x)
                                            S_bool[x] = True
                                            current_size += 1
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_S = S.copy()
                                            break

                            # Revert if not improved
                            if not improved:
                                if added_v1:
                                    S.remove(v1)
                                    S_bool[v1] = False
                                S.add(u1)
                                S.add(u2)
                                S_bool[u1] = True
                                S_bool[u2] = True

                            if improved:
                                break

                        if not improved:
                            # Revert any partial changes
                            pass

                    if not improved and no_improve > 50:
                        break

        return best_S, best_size

    # Simulated Annealing
    def simulated_annealing(S_init, t_start=1.0, t_end=0.01, cooling=0.9995, max_iters=30000):
        S = set(S_init)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True

        current_size = len(S)
        best_S = S.copy()
        best_size = current_size

        T = t_start
        rng = random.Random()

        for it in range(max_iters):
            if time.time() > deadline:
                break

            if T < t_end:
                break

            move_type = rng.random()

            if move_type < 0.35:
                # Try to add a random element not in S
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                x = rng.choice(not_in_S)
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()

            elif move_type < 0.55:
                # Try to remove a random element from S
                if not S:
                    continue
                u = rng.choice(list(S))
                S.remove(u)
                S_bool[u] = False
                current_size -= 1

            else:
                # Swap: remove u from S, add v not in S
                if not S:
                    continue
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                u = rng.choice(list(S))
                v = rng.choice(not_in_S)

                S_bool[u] = False
                S.remove(u)

                if can_add(v, S_bool):
                    S.add(v)
                    S_bool[v] = True
                    # Try to add more after swap
                    for x in not_in_S:
                        if x != v and can_add(x, S_bool):
                            S.add(x)
                            S_bool[x] = True
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
                            break
                else:
                    S.add(u)
                    S_bool[u] = True

            T *= cooling

        return best_S, best_size

    best_S = set()
    best_size = 0

    # Phase 1: Multiple greedy starts with different orderings + local search
    orderings = ['low_conflict', 'high_conflict', 'random', 'ascending', 'descending', 'prime_first']
    for ordering in orderings:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S, max_iters=8000, time_budget=5.0)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 2: Simulated annealing from best solution
    if best_S:
        S, size = simulated_annealing(best_S, t_start=1.0, t_end=0.01, cooling=0.9995, max_iters=25000)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 3: More SA restarts from random greedy starts
    while time.time() < deadline - 10:
        S = greedy_construct('random')
        S, size = simulated_annealing(S, t_start=0.5, t_end=0.01, cooling=0.999, max_iters=15000)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 4: Final local search polish on best solution
    if best_S and time.time() < deadline - 2:
        S, size = local_search(best_S, max_iters=30000, time_budget=8.0)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Phase 5: Exhaustive single-swap polish (try every possible swap systematically)
    if best_S and time.time() < deadline - 0.5:
        S = set(best_S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        current_size = len(S)

        # Try every element not in S to see if it can be added
        for x in numbers:
            if not S_bool[x] and can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True
                current_size += 1

        # Try every single swap: for each u in S, v not in S
        for u in list(S):
            if time.time() > deadline:
                break
            S_bool[u] = False
            S.remove(u)
            for v in numbers:
                if v != u and not S_bool[v]:
                    if can_add(v, S_bool):
                        S.add(v)
                        S_bool[v] = True
                        # Try to add more
                        for x in numbers:
                            if not S_bool[x] and can_add(x, S_bool):
                                S.add(x)
                                S_bool[x] = True
                                current_size += 1
                        break
            # If no swap worked, revert
            if u not in S:
                S.add(u)
                S_bool[u] = True

        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    return list(best_S)

# EVOLVE-BLOCK-END
