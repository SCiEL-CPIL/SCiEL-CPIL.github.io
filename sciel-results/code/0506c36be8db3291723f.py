import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def is_sum_free(S):
        s = set(S)
        L = list(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(len(L)):
                b = L[j]
                if (a + b) in s:
                    return False
        return True

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V_set = set(elements)
        n = len(V_set)
        V_list = list(V_set)
        rng = random.Random(seed_off)

        # Build adjacency: for each v, list of (u, w) such that {v,u,w} is a hyperedge
        # u + w = v or v + u = w or v + w = u
        adj = {v: [] for v in V_set}
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            for j in range(i + 1, n):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))

        # Degree: number of hyperedge incidences
        deg = {v: len(adj[v]) for v in V_set}

        def check_conflict(v, S):
            """Return True if v can be added to S without conflict."""
            for (u, w) in adj[v]:
                if u == v:
                    if w in S:
                        return False
                else:
                    if u in S and w in S:
                        return False
            return True

        def greedy_fill(order, S=None):
            if S is None:
                S = set()
            for v in order:
                if v not in S and check_conflict(v, S):
                    S.add(v)
            return S

        def try_add_remaining(S, deadline):
            added = True
            while added and time.time() < deadline:
                added = False
                remaining = [x for x in V_set if x not in S]
                remaining.sort(key=lambda x: (deg[x], abs(x), x))
                for x in remaining:
                    if time.time() >= deadline:
                        break
                    if x not in S and check_conflict(x, S):
                        S.add(x)
                        added = True
            return S

        def local_search_1opt(S, budget, rng):
            """Try removing one element and re-greedy filling."""
            best = set(S)
            best_size = len(best)
            t_start_ls = time.time()
            S_list = list(best)
            rng.shuffle(S_list)
            for a in S_list:
                if time.time() - t_start_ls > budget:
                    break
                trial = set(best)
                trial.discard(a)
                remaining = [x for x in V_set if x not in trial]
                remaining.sort(key=lambda x: (deg[x], abs(x), x))
                for x in remaining:
                    if x not in trial and check_conflict(x, trial):
                        trial.add(x)
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
            return best

        def local_search_2opt(S, budget, rng):
            """Try removing two elements and re-greedy filling."""
            best = set(S)
            best_size = len(best)
            t_start_ls = time.time()
            S_list = list(best)
            rng.shuffle(S_list)
            n_ls = len(S_list)
            for i in range(n_ls):
                if time.time() - t_start_ls > budget:
                    break
                a = S_list[i]
                for j in range(i + 1, n_ls):
                    if time.time() - t_start_ls > budget:
                        break
                    b = S_list[j]
                    trial = set(best)
                    trial.discard(a)
                    trial.discard(b)
                    remaining = [x for x in V_set if x not in trial]
                    remaining.sort(key=lambda x: (deg[x], abs(x), x))
                    for x in remaining:
                        if x not in trial and check_conflict(x, trial):
                            trial.add(x)
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        break
            return best

        # --- Multiple greedy initializations ---
        candidates = []

        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        candidates.append(S_odds)

        # 2. Greedy by degree (ascending)
        order_deg = sorted(V_set, key=lambda x: (deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg))

        # 3. Greedy by degree (descending)
        order_deg_desc = sorted(V_set, key=lambda x: (-deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg_desc))

        # 4. Greedy by absolute value (ascending)
        order_abs = sorted(V_set, key=lambda x: abs(x))
        candidates.append(greedy_fill(order_abs))

        # 5. Greedy by absolute value (descending)
        order_abs_desc = sorted(V_set, key=lambda x: -abs(x))
        candidates.append(greedy_fill(order_abs_desc))

        # 6. By value descending
        candidates.append(greedy_fill(sorted(V_set, reverse=True)))

        # 7. Positives by value desc, then negatives
        pos = sorted([x for x in V_set if x > 0], reverse=True)
        neg = sorted([x for x in V_set if x < 0])
        candidates.append(greedy_fill(pos + neg))

        # 8-12. Random orderings
        for _ in range(5):
            order = list(V_set)
            rng.shuffle(order)
            candidates.append(greedy_fill(order))

        # Pick best candidate
        best_S = max(candidates, key=len)
        best_size = len(best_S)

        # Try adding remaining
        best_S = try_add_remaining(best_S, t_start + budget)
        best_size = len(best_S)

        # 1-opt local search
        if time.time() - t_start < budget * 0.7:
            ls_budget = min(budget * 0.2, 8)
            best_S = local_search_1opt(best_S, ls_budget, rng)
            best_size = len(best_S)
            best_S = try_add_remaining(best_S, t_start + budget)

        # 2-opt local search
        if time.time() - t_start < budget * 0.85:
            ls_budget = min(budget * 0.15, 6)
            best_S = local_search_2opt(best_S, ls_budget, rng)
            best_size = len(best_S)
            best_S = try_add_remaining(best_S, t_start + budget)

        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 70]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
