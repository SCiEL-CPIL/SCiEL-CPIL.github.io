import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, remaining_time):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)

        # Build conflict graph: edge between i and j (i != j) if
        # values[i] + values[j] is in v_set, OR
        # 2*values[i] == values[j], OR
        # 2*values[j] == values[i]
        # This graph's Max Independent Set = Max Sum-Free Subset

        adj = [set() for _ in range(n)]

        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vi == vj:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vj == vi:
                    adj[i].add(j)
                    adj[j].add(i)

        # Convert to sorted adjacency lists for faster iteration
        adj_list = [sorted(adj[i]) for i in range(n)]
        degrees = [len(adj[i]) for i in range(n)]

        best_size = 0
        best_set = set()

        # Greedy: minimum degree first
        def greedy_mis_min():
            rem = set(range(n))
            iset = set()
            degs = degrees[:]
            while rem:
                if time.time() - start_time > DEADLINE - 15:
                    break
                # Find vertex with minimum degree in remaining
                min_v = min(rem, key=lambda v: degs[v])
                iset.add(min_v)
                rem.discard(min_v)
                for u in adj[min_v]:
                    if u in rem:
                        rem.discard(u)
                        for w in adj[u]:
                            if w in rem:
                                degs[w] -= 1
                        degs[u] = 0
            return iset

        # Greedy: random order
        def greedy_mis_random(rng):
            rem = set(range(n))
            iset = set()
            order = list(range(n))
            rng.shuffle(order)
            pos = {v: i for i, v in enumerate(order)}
            while rem:
                if time.time() - start_time > DEADLINE - 15:
                    break
                # Pick the earliest in random order among remaining
                candidates = [v for v in rem if pos[v] == min(pos[x] for x in rem)]
                v = candidates[0]
                iset.add(v)
                rem.discard(v)
                for u in adj[v]:
                    if u in rem:
                        rem.discard(u)
            return iset

        # Run greedy initializations
        g1 = greedy_mis_min()
        if len(g1) > best_size:
            best_size = len(g1)
            best_set = set(g1)

        rng = random.Random(42)
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 15:
                break
            g = greedy_mis_random(rng)
            if len(g) > best_size:
                best_size = len(g)
                best_set = set(g)

        # Local search (hill climbing with SA)
        # State: set of selected vertices
        # Moves: add a vertex, remove a vertex, swap (remove one, add another)
        # Objective: maximize size while maintaining independence

        S = set(best_set)

        def count_conflicts(S_set):
            """Count number of edges within S_set"""
            cnt = 0
            for v in S_set:
                for u in adj[v]:
                    if u in S_set and u > v:
                        cnt += 1
            return cnt

        current_conflicts = count_conflicts(S)
        # We want 0 conflicts and max size

        T = 5.0
        T_min = 0.01
        cooling = 0.999
        penalty = 1000

        current_score = len(S) - penalty * current_conflicts

        while T > T_min:
            if time.time() - start_time > DEADLINE - 10:
                break

            # Random move: add, remove, or swap
            move_type = rng.random()
            if move_type < 0.3:
                # Remove a random vertex from S
                if not S:
                    T *= cooling
                    continue
                v = rng.choice(list(S))
                new_S = S - {v}
                # Conflicts change: remove edges from v to S
                removed_conf = 0
                for u in adj[v]:
                    if u in S and u != v:
                        removed_conf += 1
                new_conf = current_conflicts - removed_conf // 2
                new_score = len(new_S) - penalty * new_conf
            elif move_type < 0.6:
                # Add a random vertex not in S
                not_in_S = [v for v in range(n) if v not in S]
                if not not_in_S:
                    T *= cooling
                    continue
                v = rng.choice(not_in_S)
                new_S = S | {v}
                added_conf = 0
                for u in adj[v]:
                    if u in S:
                        added_conf += 1
                new_conf = current_conflicts + added_conf // 2
                new_score = len(new_S) - penalty * new_conf
            else:
                # Swap: remove one from S, add one not in S
                if not S:
                    T *= cooling
                    continue
                not_in_S = [v for v in range(n) if v not in S]
                if not not_in_S:
                    T *= cooling
                    continue
                v_remove = rng.choice(list(S))
                v_add = rng.choice(not_in_S)
                if v_remove == v_add:
                    T *= cooling
                    continue
                new_S = S - {v_remove} | {v_add}
                # Compute new conflicts
                new_conf = count_conflicts(new_S)
                new_score = len(new_S) - penalty * new_conf

            delta = new_score - current_score
            if delta >= 0 or rng.random() < math.exp(-delta / T):
                S = new_S
                current_conflicts = new_conf
                current_score = new_score

                if current_conflicts == 0 and len(S) > best_size:
                    best_size = len(S)
                    best_set = set(S)

            T *= cooling

        # Multiple SA restarts from different starting points
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Start from a random valid solution
            if restart == 0:
                S = set(best_set)
            else:
                S = greedy_mis_random(rng)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = set(S)

            current_conflicts = count_conflicts(S)
            T = 3.0
            T_min = 0.01
            cooling = 0.999
            penalty = 1000
            current_score = len(S) - penalty * current_conflicts

            while T > T_min:
                if time.time() - start_time > DEADLINE - 8:
                    break

                move_type = rng.random()
                if move_type < 0.3:
                    if not S:
                        T *= cooling
                        continue
                    v = rng.choice(list(S))
                    new_S = S - {v}
                    removed_conf = 0
                    for u in adj[v]:
                        if u in S and u != v:
                            removed_conf += 1
                    new_conf = current_conflicts - removed_conf // 2
                    new_score = len(new_S) - penalty * new_conf
                elif move_type < 0.6:
                    not_in_S = [v for v in range(n) if v not in S]
                    if not not_in_S:
                        T *= cooling
                        continue
                    v = rng.choice(not_in_S)
                    new_S = S | {v}
                    added_conf = 0
                    for u in adj[v]:
                        if u in S:
                            added_conf += 1
                    new_conf = current_conflicts + added_conf // 2
                    new_score = len(new_S) - penalty * new_conf
                else:
                    if not S:
                        T *= cooling
                        continue
                    not_in_S = [v for v in range(n) if v not in S]
                    if not not_in_S:
                        T *= cooling
                        continue
                    v_remove = rng.choice(list(S))
                    v_add = rng.choice(not_in_S)
                    if v_remove == v_add:
                        T *= cooling
                        continue
                    new_S = S - {v_remove} | {v_add}
                    new_conf = count_conflicts(new_S)
                    new_score = len(new_S) - penalty * new_conf

                delta = new_score - current_score
                if delta >= 0 or rng.random() < math.exp(-delta / T):
                    S = new_S
                    current_conflicts = new_conf
                    current_score = new_score

                    if current_conflicts == 0 and len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)

                T *= cooling

        best_values = [values[i] for i in best_set]
        return best_values

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            if not is_sum_free(res):
                res = [x for x in vals if x != 0]
            results.append(res)
            continue

        vals = instance(size, seed)
        budget = min(remaining - 5, 15 + 70 * (size / 240))
        res = solve_instance(vals, budget)

        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]
            if not is_sum_free(res):
                res = [x for x in vals if x != 0]

        results.append(res)

    return results
