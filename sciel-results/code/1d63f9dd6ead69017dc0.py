def construct():
    import math
    import random
    import time

    N = 256
    # Squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    # First, remove vertices where 2a is a perfect square
    valid_vertices = []
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_vertices.append(a)

    # Build conflict graph: edge between a,b (a<b) if a+b is a square
    adj = {a: set() for a in valid_vertices}
    for i in range(len(valid_vertices)):
        for j in range(i + 1, len(valid_vertices)):
            a = valid_vertices[i]
            b = valid_vertices[j]
            if a + b in squares:
                adj[a].add(b)
                adj[b].add(a)

    def greedy_independent_set(order):
        chosen = set()
        for v in order:
            if v not in chosen:
                # Check if v conflicts with any in chosen
                conflict = False
                for u in chosen:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    chosen.add(v)
        return chosen

    def local_search(initial_set):
        chosen = set(initial_set)
        improved = True
        max_iter = 50

        while improved and max_iter > 0:
            improved = False
            max_iter -= 1

            # 1. Add any non-conflicting vertex
            for v in valid_vertices:
                if v not in chosen:
                    ok = True
                    for u in chosen:
                        if u in adj[v]:
                            ok = False
                            break
                    if ok:
                        chosen.add(v)
                        improved = True

            # 2. Try Remove 1, Add 2
            found_improvement = False
            chosen_list = list(chosen)
            random.shuffle(chosen_list)

            for v in chosen_list:
                remaining = chosen - {v}
                candidates = []
                for w in valid_vertices:
                    if w in chosen:
                        continue
                    ok = True
                    for u in remaining:
                        if u in adj[w]:
                            ok = False
                            break
                    if ok:
                        candidates.append(w)

                if len(candidates) >= 2:
                    for i in range(len(candidates)):
                        w1 = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            w2 = candidates[j]
                            if w1 not in adj[w2]:
                                chosen = remaining | {w1, w2}
                                improved = True
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                if found_improvement:
                    break

            # 3. Try Remove 2, Add 3
            if not improved:
                found_improvement = False
                chosen_list = list(chosen)
                random.shuffle(chosen_list)
                max_pairs = 20
                pairs_checked = 0

                for i in range(len(chosen_list)):
                    if pairs_checked >= max_pairs:
                        break
                    v1 = chosen_list[i]
                    for j in range(i + 1, len(chosen_list)):
                        if pairs_checked >= max_pairs:
                            break
                        pairs_checked += 1
                        v2 = chosen_list[j]

                        remaining = chosen - {v1, v2}
                        candidates = []
                        for w in valid_vertices:
                            if w in chosen:
                                continue
                            ok = True
                            for u in remaining:
                                if u in adj[w]:
                                    ok = False
                                    break
                            if ok:
                                candidates.append(w)

                        if len(candidates) >= 3:
                            for k in range(len(candidates)):
                                w1 = candidates[k]
                                for l in range(k + 1, len(candidates)):
                                    w2 = candidates[l]
                                    if w1 in adj[w2]:
                                        continue
                                    for m in range(l + 1, len(candidates)):
                                        w3 = candidates[m]
                                        if w2 in adj[w3] or w1 in adj[w3]:
                                            continue
                                        chosen = remaining | {w1, w2, w3}
                                        improved = True
                                        found_improvement = True
                                        break
                                    if found_improvement:
                                        break
                                if found_improvement:
                                    break
                            if found_improvement:
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break

        return chosen

    # Strategy: Multiple restarts with random greedy + local search
    best_set = set()

    # Initial deterministic tries
    by_degree_asc = sorted(valid_vertices, key=lambda x: len(adj[x]))
    s = local_search(greedy_independent_set(by_degree_asc))
    if len(s) > len(best_set):
        best_set = s

    by_degree_desc = sorted(valid_vertices, key=lambda x: -len(adj[x]))
    s = local_search(greedy_independent_set(by_degree_desc))
    if len(s) > len(best_set):
        best_set = s

    # Random restarts
    start_time = time.time()
    limit = 55.0

    random.seed(42)
    for i in range(1000):
        if time.time() - start_time > limit:
            break

        random_order = valid_vertices[:]
        random.shuffle(random_order)
        s = greedy_independent_set(random_order)
        s = local_search(s)

        if len(s) > len(best_set):
            best_set = s

    return list(best_set)
