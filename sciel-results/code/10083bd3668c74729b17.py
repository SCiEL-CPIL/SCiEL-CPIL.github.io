import time
import random
import math


def construct():
    start_time = time.time()
    timeout = 55.0  # safety margin before the 60 s evaluator limit

    n = 256
    adj = [set() for _ in range(n + 1)]

    # Build conflict graph: edge (a,b) when (a*b) % (a+b) == 0
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)

    def greedy_mis(adj, n, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        remaining = set(range(1, n + 1))
        result = []

        while remaining:
            min_v = None
            min_deg = float('inf')
            best_cands = []

            for v in remaining:
                deg = sum(1 for u in adj[v] if u in remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_cands = [v]
                elif deg == min_deg:
                    best_cands.append(v)

            if random_seed is not None and best_cands:
                min_v = random.choice(best_cands)
            else:
                min_v = best_cands[0]

            result.append(min_v)

            to_remove = {min_v}
            for u in adj[min_v]:
                if u in remaining:
                    to_remove.add(u)
            remaining -= to_remove

            if time.time() - start_time > timeout - 5:
                break

        return set(result)

    def try_improve(S, adj, n):
        # Local search: remove a vertex, add an independent subset of its neighbours
        S = set(S)
        verts = list(S)
        random.shuffle(verts)

        for v in verts:
            if time.time() - start_time > timeout:
                return S

            S_new = S - {v}
            candidates = [u for u in adj[v] if u not in S_new]

            valid_cands = []
            for u in candidates:
                ok = True
                for w in adj[u]:
                    if w in S_new:
                        ok = False
                        break
                if ok:
                    valid_cands.append(u)

            if valid_cands:
                rem_c = set(valid_cands)
                add_set = set()
                while rem_c:
                    min_v_c = None
                    min_deg_c = float('inf')
                    for vc in rem_c:
                        deg = sum(1 for uc in adj[vc] if uc in rem_c)
                        if deg < min_deg_c:
                            min_deg_c = deg
                            min_v_c = vc
                    add_set.add(min_v_c)
                    rem_c -= {min_v_c}
                    for uc in adj[min_v_c]:
                        if uc in rem_c:
                            rem_c.discard(uc)

                if len(S_new) + len(add_set) > len(S):
                    S = S_new | add_set

        return S

    def anneal(S, adj, n, time_budget):
        # Simulated annealing over independent sets.
        # Uses set intersection for O(1)-amortised can_add checks, a high swap
        # probability to explore same-size configurations, and slow cooling so
        # the chain stays mobile. math.exp is used for the Boltzmann factor.
        S = set(S)
        T0 = 1.2
        T_end = 0.015
        cooling = 0.9995   # slow cooling -> more iterations at high T
        swap_p = 0.5       # high swap probability -> more exploration
        T = T0

        while time.time() - start_time < time_budget:
            if T < T_end:
                break

            # choose a random vertex in S to try removing
            v = random.choice(tuple(S))
            S_new = S - {v}

            # neighbours of v not in S_new
            cands = [u for u in adj[v] if u not in S_new]

            # valid if no neighbour of u is in S_new (O(1) via set intersection)
            valid = [u for u in cands if not (adj[u] & S_new)]

            # greedy independent subset of valid
            rem = set(valid)
            add = set()
            while rem:
                vc = random.choice(tuple(rem))
                add.add(vc)
                rem -= adj[vc]
                rem.discard(vc)

            S_cand = S_new | add
            d = len(S_cand) - len(S)

            if d >= 0 or random.random() < math.exp(d / T):
                S = S_cand

            T *= cooling

        return S

    # Initial solution
    best_sol = greedy_mis(adj, n)
    best_size = len(best_sol)

    # Local-search phase
    while time.time() - start_time < timeout - 2:
        sol = try_improve(best_sol, adj, n)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol

        # Simulated annealing refinement (strongest inspiration idea)
        anneal_budget = min(12.0, max(2.0, timeout - (time.time() - start_time) - 2))
        if time.time() - start_time + anneal_budget <= timeout - 1:
            sol2 = anneal(best_sol, adj, n, timeout - (time.time() - start_time) - 2)
            if len(sol2) > best_size:
                best_size = len(sol2)
                best_sol = sol2
                sol2 = try_improve(best_sol, adj, n)
                if len(sol2) > best_size:
                    best_size = len(sol2)
                    best_sol = sol2

        # Random restart to escape plateaus
        if random.random() < 0.25:
            new_sol = greedy_mis(adj, n, random_seed=int(time.time() * 100000) % (10**9))
            new_sol = try_improve(new_sol, adj, n)
            if len(new_sol) > best_size:
                best_size = len(new_sol)
                best_sol = new_sol

    return list(best_sol)
