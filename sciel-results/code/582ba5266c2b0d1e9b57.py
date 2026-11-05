import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 138  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def can_add(S, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S:
            if (a + x) in S or (x - a) in S:
                return False
        if (2 * x) in S:
            return False
        return True

    def conflict_count(S, x):
        """Count how many elements in S would conflict with x."""
        cnt = 0
        for a in S:
            if (a + x) in S or (x - a) in S:
                cnt += 1
        if (2 * x) in S:
            cnt += 1
        return cnt

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

    def greedy_from_order(order, V_set, S0=None):
        S = set(S0) if S0 is not None else set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def greedy_min_conflict(V_set, S0=None, max_passes=5):
        """Greedy that repeatedly picks the element with fewest conflicts."""
        S = set(S0) if S0 is not None else set()
        candidates = [x for x in V_set if x not in S]
        for _ in range(max_passes):
            if not candidates:
                break
            candidates.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            added_any = False
            new_candidates = []
            for x in candidates:
                if x not in S and can_add(S, x):
                    S.add(x)
                    added_any = True
                elif x not in S:
                    new_candidates.append(x)
            candidates = new_candidates
            if not added_any:
                break
        return S

    def try_add_remaining(S, V_set, deadline):
        added = True
        while added and time.time() < deadline:
            added = False
            remaining = [x for x in V_set if x not in S]
            remaining.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            for x in remaining:
                if time.time() >= deadline:
                    break
                if x not in S and can_add(S, x):
                    S.add(x)
                    added = True
        return S

    def hypergraph_greedy(V_set, seed_off, budget):
        """
        Hypergraph-inspired greedy: model sum-free as max independent set on 3-uniform hypergraph.
        Each hyperedge is a triple (a, b, c) with a+b=c. We want max subset with no hyperedge fully contained.
        Strategy: iteratively remove the vertex (element) whose removal destroys the most hyperedges
        while preserving the most potential for other vertices.
        """
        t_start = time.time()
        rng = random.Random(seed_off)
        V_list = list(V_set)
        V_s = set(V_list)
        
        # Build all sum triples
        triples = []
        V_sorted = sorted(V_list)
        V_idx = {v: i for i, v in enumerate(V_sorted)}
        for i, a in enumerate(V_sorted):
            for j, b in enumerate(V_sorted):
                c = a + b
                if c in V_idx:
                    triples.append((a, b, c))
        
        # Count how many triples each element participates in
        elem_triple_count = {v: 0 for v in V_list}
        for (a, b, c) in triples:
            elem_triple_count[a] += 1
            elem_triple_count[b] += 1
            elem_triple_count[c] += 1
        
        # Greedy: start with all elements, iteratively remove the element
        # whose removal breaks the most "active" triples (triples where all 3 are still present)
        # while minimizing loss of "potential" (elements that are in few triples)
        
        active = set(V_list)
        # Priority: remove elements that are in many active triples
        # But we want to keep as many elements as possible, so we remove the one
        # that breaks the most triples per element removed
        
        # Alternative: greedy construction - add elements one by one, preferring
        # elements that are in fewer triples (less constrained)
        
        # Try multiple strategies
        best = set()
        best_size = 0
        
        # Strategy 1: Add elements sorted by triple count (ascending)
        order1 = sorted(V_list, key=lambda x: (elem_triple_count[x], abs(x), x))
        S1 = greedy_from_order(order1, V_set)
        if len(S1) > best_size:
            best_size = len(S1)
            best = set(S1)
        
        # Strategy 2: Add elements sorted by triple count (descending)
        order2 = sorted(V_list, key=lambda x: (-elem_triple_count[x], abs(x), x))
        S2 = greedy_from_order(order2, V_set)
        if len(S2) > best_size:
            best_size = len(S2)
            best = set(S2)
        
        # Strategy 3: Random restarts with min-conflict greedy
        for _ in range(3):
            if time.time() - t_start > budget:
                break
            S3 = greedy_min_conflict(V_set, max_passes=5)
            if len(S3) > best_size:
                best_size = len(S3)
                best = set(S3)
        
        return best

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        best = set()
        best_size = 0
        rng = random.Random(seed_off)

        # Precompute potential conflict degree for each element
        # For element x, its "potential conflicts" are elements a in V where
        # a+x in V or x-a in V or 2x in V (these would conflict if both were in S)
        V_s = set(V)
        potential_degree = {}
        for x in V:
            deg = 0
            for a in V:
                if a == x:
                    continue
                if (a + x) in V_s or (x - a) in V_s:
                    deg += 1
            if (2 * x) in V_s:
                deg += 1
            potential_degree[x] = deg

        # Candidate structural orders
        orders = []

        # A. By value descending (largest first) — upper interval
        orders.append(sorted(V, key=lambda x: x, reverse=True))
        # B. By value ascending
        orders.append(sorted(V, key=lambda x: x))
        # C. By |x| descending
        orders.append(sorted(V, key=lambda x: abs(x), reverse=True))
        # D. Odds first, then evens by |x|
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        orders.append(sorted(odds, key=lambda x: abs(x), reverse=True) + sorted(evens, key=lambda x: abs(x), reverse=True))
        # E. Positives by value desc, then negatives by value asc (most neg)
        pos = sorted([x for x in V if x > 0], reverse=True)
        neg = sorted([x for x in V if x < 0])
        orders.append(pos + neg)
        # F. Negatives first (most negative), then positives
        neg2 = sorted([x for x in V if x < 0], reverse=True)
        pos2 = sorted([x for x in V if x > 0])
        orders.append(neg2 + pos2)
        # G. Upper value interval only: top 60% by value
        sv = sorted(V)
        k = int(n * 0.6)
        orders.append(sv[n-k:])
        # H. Top 60% by |x|
        sm = sorted(V, key=lambda x: abs(x), reverse=True)
        orders.append(sm[:k])
        # I. By potential degree ascending (least constrained first)
        orders.append(sorted(V, key=lambda x: (potential_degree[x], abs(x), x)))
        # J. By potential degree descending
        orders.append(sorted(V, key=lambda x: (-potential_degree[x], abs(x), x)))
        # K. Random orders
        for _ in range(8):
            rand_order = V[:]
            rng.shuffle(rand_order)
            orders.append(rand_order)

        # Run greedy on each order
        for order in orders:
            if time.time() - t_start > budget:
                break
            S = greedy_from_order(order, V_set)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Hypergraph-inspired greedy
        if time.time() - t_start < budget:
            hg_budget = min(budget * 0.3, 10)
            S_hg = hypergraph_greedy(V_set, seed_off + 100, hg_budget)
            if len(S_hg) > best_size:
                best_size = len(S_hg)
                best = set(S_hg)

        # Also try conflict-aware greedy from scratch
        if time.time() - t_start < budget:
            S = greedy_min_conflict(V_set, max_passes=5)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Local improvement: try adding any remaining element
        best = try_add_remaining(best, V_set, t_start + budget)
        best_size = len(best)

        # 1-opt swaps: remove one, try to add two
        improved = True
        while improved and time.time() - t_start < budget * 0.85:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for a in S_list:
                if time.time() - t_start > budget * 0.85:
                    break
                trial = set(best)
                trial.discard(a)
                # try to fill with conflict-aware ordering
                remaining = [x for x in V_set if x not in trial]
                remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                for x in remaining:
                    if x not in trial and can_add(trial, x):
                        trial.add(x)
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
                    improved = True
                    break
        best = try_add_remaining(best, V_set, t_start + budget)

        # 2-opt swaps: remove two, try to add three
        improved = True
        while improved and time.time() - t_start < budget:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for i in range(len(S_list)):
                if time.time() - t_start > budget:
                    break
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() - t_start > budget:
                        break
                    b = S_list[j]
                    trial = set(best)
                    trial.discard(a)
                    trial.discard(b)
                    # try to fill with conflict-aware ordering
                    remaining = [x for x in V_set if x not in trial]
                    remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                    for x in remaining:
                        if x not in trial and can_add(trial, x):
                            trial.add(x)
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
        best = try_add_remaining(best, V_set, t_start + budget)

        return list(best)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    # time budgets, leave margin
    budgets = [10, 28, 65]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            # fallback: odds
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        # verify
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
