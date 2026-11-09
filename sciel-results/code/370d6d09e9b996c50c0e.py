import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 142.0   # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Graph adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Conflict sets: two edges conflict iff they share a vertex OR are joined by an edge
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        # ---- DSATUR backtracking (finds valid coloring, prunes by current best) ----
        colors = [-1] * m
        color_members = [set() for _ in range(2 * d + 20)]  # enough for any reasonable bound
        uncolored = set(range(m))
        static_deg = [len(conflict_sets[i]) for i in range(m)]
        best = [None]
        best_count = [10 ** 9]

        def expand_color_sets():
            need = len(color_members)
            for c in range(len(color_members)):
                if color_members[c]:
                    need = c + 1
            while need >= len(color_members):
                color_members.append(set())

        def backtrack(node_count, max_nodes):
            if time.time() > time_limit:
                return
            if not uncolored:
                mx = -1
                for c in range(len(color_members)):
                    if color_members[c]:
                        mx = c
                cnt = mx + 1
                if cnt < best_count[0]:
                    best_count[0] = cnt
                    best[0] = colors[:]
                return
            if node_count >= max_nodes:
                return
            # choose edge: max saturation, then max uncolored degree, then static degree
            best_i, best_key = None, (-1, -1, -1)
            for i in uncolored:
                sat = 0
                seen = set()
                for j in conflict_sets[i]:
                    cj = colors[j]
                    if cj != -1 and cj not in seen:
                        seen.add(cj)
                sat = len(seen)
                deg = 0
                for j in conflict_sets[i]:
                    if j in uncolored:
                        deg += 1
                key = (sat, deg, static_deg[i])
                if key > best_key:
                    best_key = key
                    best_i = i
            i = best_i
            # try colors in ascending order
            c = 0
            while True:
                ok = True
                for j in conflict_sets[i]:
                    if colors[j] == c:
                        ok = False
                        break
                if ok:
                    colors[i] = c
                    color_members[c].add(i)
                    uncolored.remove(i)
                    backtrack(node_count + 1, max_nodes)
                    uncolored.add(i)
                    color_members[c].discard(i)
                    colors[i] = -1
                    if best_count[0] <= c + 1:
                        # no point trying higher colors (can't improve)
                        break
                c += 1
                if c >= len(color_members):
                    expand_color_sets()
                if c >= best_count[0]:
                    break

        # ---- Simulated annealing with correct incremental conflict accounting ----
        def refine(colors_in, time_budget):
            rng = random.Random(seed + 7)
            cur = colors_in[:]
            max_c = max(cur) if cur else -1
            # per-color member sets
            members = defaultdict(set)
            for idx, c in enumerate(cur):
                members[c].add(idx)
            # total induced-matching conflicts
            total_conf = 0
            for c, ms in members.items():
                ms = list(ms)
                for a in range(len(ms)):
                    ea = ms[a]
                    for b in range(a + 1, len(ms)):
                        if ms[b] in conflict_sets[ea]:
                            total_conf += 1
            best_conf = total_conf
            best_colors = cur[:]
            best_max = max_c
            T = 8.0
            cooling = 0.99995
            steps = 0
            end = time.time() + time_budget
            while time.time() < end:
                i = rng.randrange(m)
                old_c = cur[i]
                new_c = rng.randint(0, max_c + 1)
                if new_c == old_c:
                    continue
                # conflicts gained/lost: pairs (i, j) where j in old/new color and conflicts with i
                rem = 0
                for j in members[old_c]:
                    if j != i and j in conflict_sets[i]:
                        rem += 1
                add = 0
                for j in members[new_c]:
                    if j != i and j in conflict_sets[i]:
                        add += 1
                delta = add - rem
                # apply
                members[old_c].discard(i)
                members[new_c].add(i)
                cur[i] = new_c
                total_conf += delta
                new_max = max_c
                if members[old_c] and old_c == max_c:
                    new_max = max_c
                elif not members[old_c] and old_c == max_c:
                    nm = -1
                    for cc, ms in members.items():
                        if ms and cc > nm:
                            nm = cc
                    new_max = nm
                if new_c > new_max:
                    new_max = new_c
                # energy: conflicts dominate, then number of colors
                old_e = best_conf * 1000 + best_max + 1
                new_e = total_conf * 1000 + new_max + 1
                if new_e <= old_e or rng.random() < (2.718281828 ** (-(new_e - old_e) / T)):
                    max_c = new_max
                    if total_conf < best_conf or (total_conf == best_conf and new_max < best_max):
                        best_conf = total_conf
                        best_colors = cur[:]
                        best_max = new_max
                else:
                    # revert
                    members[new_c].discard(i)
                    members[old_c].add(i)
                    cur[i] = old_c
                    total_conf -= delta
                T *= cooling
                steps += 1
                if steps % 5000 == 0 and best_conf == 0:
                    break
            return best_colors, best_conf, best_max

        # ---- Time-sliced loop: refine, then backtrack with improving bound ----
        # initial: greedy DSATUR (fast, no SA)
        colors0 = [-1] * m
        uncolored0 = set(range(m))
        for _ in range(m):
            best_i, best_key = None, (-1, -1)
            for i in uncolored0:
                seen = set()
                for j in conflict_sets[i]:
                    if colors0[j] != -1:
                        seen.add(colors0[j])
                deg = sum(1 for j in conflict_sets[i] if j in uncolored0)
                key = (len(seen), deg)
                if key > best_key:
                    best_key = key
                    best_i = i
            i = best_i
            taken = set()
            for j in conflict_sets[i]:
                if colors0[j] != -1:
                    taken.add(colors0[j])
            c = 0
            while c in taken:
                c += 1
            colors0[i] = c
            uncolored0.discard(i)

        cur_best = colors0[:]
        cur_conf = None
        # count conflicts of initial
        mm = defaultdict(set)
        for idx, c in enumerate(cur_best):
            mm[c].add(idx)
        for c, ms in mm.items():
            lst = list(ms)
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    if lst[b] in conflict_sets[lst[a]]:
                        cur_conf = (cur_conf or 0) + 1

        # reset backtracking state for use
        colors[:] = [-1] * m
        for s in color_members:
            s.clear()
        uncolored.clear()
        uncolored.update(range(m))
        best[0] = cur_best[:]
        best_count[0] = max(cur_best) + 1

        while time.time() < time_limit:
            # refine with a slice
            slice_t = min(2.0, max(0.3, time_limit - time.time()))
            cur_best, cur_conf, _ = refine(cur_best, slice_t)
            # update backtracking bound if we have a valid (conf==0) solution
            if cur_conf == 0:
                cnt = max(cur_best) + 1
                if cnt < best_count[0]:
                    best_count[0] = cnt
                    best[0] = cur_best[:]
            # backtrack to try to beat the current best
            colors[:] = [-1] * m
            for s in color_members:
                s.clear()
                for _ in range(len(s)):
                    pass
            for s in color_members:
                s.clear()
            uncolored.clear()
            uncolored.update(range(m))
            # reinit best in case
            if best[0] is None:
                best[0] = cur_best[:]
                best_count[0] = max(cur_best) + 1
            node_budget = max(20000, int((time_limit - time.time()) * 20000))
            backtrack(0, node_budget)
            if best[0] is not None:
                cur_best = best[0][:]
                # recompute conf of best (should be 0)
                cur_conf = 0
            if cur_conf == 0 and best_count[0] <= int(5 * d * d / 4):
                # reached conjecture bound; keep refining a bit but we're likely optimal
                if time.time() > time_limit - 1.0:
                    break

        # final: ensure we return a valid coloring (prefer best from backtracking if valid)
        if best[0] is not None:
            # verify best is valid
            ok = True
            mm = defaultdict(set)
            for idx, c in enumerate(best[0]):
                mm[c].add(idx)
            for c, ms in mm.items():
                lst = list(ms)
                for a in range(len(lst)):
                    for b in range(a + 1, len(lst)):
                        if lst[b] in conflict_sets[lst[a]]:
                            ok = False
            if ok:
                return best[0]
        # fallback: return cur_best if valid, else a simple greedy
        mm = defaultdict(set)
        for idx, c in enumerate(cur_best):
            mm[c].add(idx)
        ok = True
        for c, ms in mm.items():
            lst = list(ms)
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    if lst[b] in conflict_sets[lst[a]]:
                        ok = False
        if ok:
            return cur_best
        # last resort greedy
        cols = [-1] * m
        uncol = set(range(m))
        for _ in range(m):
            bi, bk = None, (-1, -1)
            for i in uncol:
                seen = set()
                for j in conflict_sets[i]:
                    if cols[j] != -1:
                        seen.add(cols[j])
                deg = sum(1 for j in conflict_sets[i] if j in uncol)
                k = (len(seen), deg)
                if k > bk:
                    bk = k
                    bi = i
            taken = set()
            for j in conflict_sets[bi]:
                if cols[j] != -1:
                    taken.add(cols[j])
            c = 0
            while c in taken:
                c += 1
            cols[bi] = c
            uncol.discard(bi)
        return cols

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0.5:
            # emergency greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cols = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near |= adj[x]
                taken = set()
                for f, c in cols.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                cols[e] = c
            results.append([cols[e] for e in edges])
            continue
        tl = max(1.0, remaining - 1.0)
        results.append(solve_instance(n, d, seed, tl))
    return results
