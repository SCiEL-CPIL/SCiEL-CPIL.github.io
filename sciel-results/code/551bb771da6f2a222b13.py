"""Combined: B&B exact solver + remove-and-retry + simulated-annealing refinement."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 146.0   # wall-clock limit (150s hard kill) with margin
    MARGIN = 3.0       # safety margin seconds

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    # ---------- conflict helpers (vectorised-ish but clear) ----------
    def can_add(x, s):
        """s is a set. Can x be added without a+b=c?"""
        if 2 * x in s:
            return False
        for y in s:
            if x + y in s:
                return False
            if x - y in s:
                return False
        return True

    def get_conflicts(x, s):
        c = set()
        if 2 * x in s:
            c.add(2 * x)
        for y in s:
            if x + y in s:
                c.add(x + y)
            if x - y in s:
                c.add(x - y)
        return c

    # ---------- greedy initialisers (multiple diverse orderings) ----------
    def greedy_from_order(S, order, val_to_idx):
        chosen_set = set()
        for idx in order:
            x = S[idx]
            if can_add(x, chosen_set):
                chosen_set.add(x)
        return chosen_set

    def greedy_multi_start(S, val_to_idx, deadline):
        n = len(S)
        rng = random.Random(12345)
        pos = [i for i in range(n) if S[i] > 0]
        neg = [i for i in range(n) if S[i] < 0]
        orders = []
        orders.append(list(range(n)))                                  # sorted ascending
        orders.append(list(range(n - 1, -1, -1)))                      # descending
        orders.append(sorted(range(n), key=lambda i: -abs(S[i])))      # by |x|
        orders.append(sorted(range(n), key=lambda i: abs(S[i])))       # by |x| asc
        orders.append(pos + neg)                                       # pos then neg
        orders.append(neg + pos)                                       # neg then pos
        # random shuffles
        for _ in range(6):
            o = list(range(n))
            rng.shuffle(o)
            orders.append(o)

        best = set()
        for o in orders:
            if time.time() > deadline:
                break
            c = greedy_from_order(S, o, val_to_idx)
            if len(c) > len(best):
                best = c
        return best

    # ---------- remove-and-retry ----------
    def remove_and_retry(S, initial, val_to_idx, deadline, max_rounds=120):
        s = set(initial)
        n = len(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() > deadline:
                break
            improved = False
            rounds += 1
            best_gain = 0
            best_new = None
            for x in list(s):
                s_no_x = set(s)
                s_no_x.discard(x)
                # candidates: all elements not in s_no_x, ordered by |.|
                cand = [S[i] for i in range(n) if S[i] not in s_no_x]
                cand.sort(key=lambda z: -abs(z))
                sc = set(s_no_x)
                for y in cand:
                    if can_add(y, sc):
                        sc.add(y)
                gain = len(sc) - len(s)
                if gain > best_gain:
                    best_gain = gain
                    best_new = sc
            if best_gain > 0 and best_new is not None:
                s = best_new
                improved = True
        return s

    # ---------- simulated annealing ----------
    def simulated_annealing(S, initial, val_to_idx, deadline):
        rng = random.Random(7)
        s = set(initial)
        n = len(S)
        best_s = set(s)
        best_size = len(s)
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        while time.time() < deadline:
            if T < T_min:
                T = 1.0  # restart temperature occasionally
            new_s = set(s)
            move = rng.random()
            if move < 0.35:
                # add a random candidate
                cand = [S[i] for i in range(n) if S[i] not in s]
                if cand:
                    x = cand[rng.randrange(len(cand))]
                    if can_add(x, new_s):
                        new_s.add(x)
                    else:
                        conf = get_conflicts(x, new_s)
                        for c in list(conf):
                            if rng.random() < 0.5:
                                new_s.discard(c)
                        if can_add(x, new_s):
                            new_s.add(x)
            elif move < 0.55:
                if s:
                    x = list(s)[rng.randrange(len(s))]
                    new_s.discard(x)
            else:
                # swap
                if s:
                    x = list(s)[rng.randrange(len(s))]
                    new_s.discard(x)
                    cand = [S[i] for i in range(n) if S[i] not in new_s]
                    for _ in range(min(3, len(cand))):
                        y = cand[rng.randrange(len(cand))]
                        if can_add(y, new_s):
                            new_s.add(y)
                            break
            delta = len(new_s) - len(s)
            if delta > 0 or (delta == 0 and rng.random() < 0.3):
                s = new_s
                if len(s) > best_size:
                    best_size = len(s)
                    best_s = set(s)
            elif delta < 0:
                if rng.random() < (T / T_min):
                    s = new_s
            T *= cooling
        return best_s

    # ---------- B&B exact solver (Max Independent Set on conflict graph) ----------
    def solve_mwis(S, deadline):
        n = len(S)
        val_to_idx = {v: i for i, v in enumerate(S)}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if S[i] + S[j] in val_to_idx:
                    adj[i].add(j)
                    adj[j].add(i)

        def greedy_independent_set(adj, n):
            remaining = set(range(n))
            chosen = []
            while remaining:
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                chosen.append(v)
                remaining.discard(v)
                remaining -= adj[v]
            return chosen

        init_set = greedy_independent_set(adj, n)
        best_size = len(init_set)
        best_set = set(init_set)

        import sys
        sys.setrecursionlimit(20000)

        def bound(P):
            if not P:
                return 0
            colors = {}
            current_color = 0
            P_list = list(P)
            P_list.sort(key=lambda x: len(adj[x] & P), reverse=True)
            for v in P_list:
                used = {colors[u] for u in adj[v] if u in colors}
                c = 0
                while c in used:
                    c += 1
                colors[v] = c
                if c > current_color:
                    current_color = c
            return current_color + 1

        def bb(P, I):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            if not P:
                if len(I) > best_size:
                    best_size = len(I)
                    best_set = set(I)
                return
            if len(I) + bound(P) <= best_size:
                return
            v = max(P, key=lambda x: len(adj[x] & P))
            I_new = I | {v}
            P_new = P - (adj[v] | {v})
            bb(P_new, I_new)
            P_new2 = P - {v}
            bb(P_new2, I)

        bb(set(range(n)), set())
        return [S[i] for i in best_set]

    def solve_instance(S, time_budget):
        deadline = start_time + time_budget
        n = len(S)
        val_to_idx = {v: i for i, v in enumerate(S)}

        # Stage 1: multi-start greedy
        g = greedy_multi_start(S, val_to_idx, deadline)
        # Stage 2: remove-and-retry
        r = remove_and_retry(S, g, val_to_idx, deadline)
        # Stage 3: simulated annealing
        a = simulated_annealing(S, r, val_to_idx, deadline)
        best = a
        # Stage 4: B&B (exact) if feasible within remaining time
        remaining = deadline - time.time()
        if remaining > 1.0:
            # B&B is only reasonable for smallish n; cap by n to avoid blow-up
            if n <= 130:
                bb_res = solve_mwis(S, deadline)
                if is_sum_free(bb_res) and len(bb_res) > len(best):
                    best = set(bb_res)
        return list(best)

    # ---------- drive all instances, distributing time ----------
    instances = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        instances.append(S)

    # Time allocation: give the small one a bit, medium more, large the rest.
    total_budget = DEADLINE - MARGIN - (time.time() - start_time)
    # proportional-ish: 60->20%, 120->35%, 240->45%
    fracs = [0.18, 0.32, 0.50]
    budgets = [total_budget * f for f in fracs]

    results = []
    for i, S in enumerate(instances):
        if time.time() - start_time > DEADLINE - MARGIN:
            results.append([x for x in S if x % 2 != 0])
            continue
        tb = min(budgets[i], DEADLINE - MARGIN - (time.time() - start_time))
        if tb < 1.0:
            tb = 1.0
        try:
            res = solve_instance(S, tb)
        except Exception:
            res = [x for x in S if x % 2 != 0]
        if not is_sum_free(res):
            res = [x for x in S if x % 2 != 0]
        results.append(res)

    return results
