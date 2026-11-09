"""Rewritten program: clique partition of split graphs via C-partition + I-packing search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed, deadline):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        # shadow of each I-vertex: set of C-neighbors
        shadow = []
        for i in range(k, n):
            s = frozenset(c for c in range(k) if (c, i) in edge_set)
            shadow.append(s)
        sizes = [len(s) for s in shadow]

        # A partition of C into groups. For each group G, the group-clique = G + {i : S_i ⊆ G}.
        # C-C edges within same group: covered by group-clique.
        # C-C edges across groups: each becomes a 2-clique.
        # Cross edges (c,i): covered by group-clique if i packed into a group containing S_i;
        #   else each (c,i) is a 2-clique.
        # Cost = (#groups) + (#cross-group C-C pairs) + sum_{unpacked i} |S_i|
        # Note: a group of size 1 with no packed I-vertex and no cross-group C-C edges would
        # produce a size-1 "clique" -> invalid. But if it has cross-group C-C edges, those 2-cliques
        # cover its C-C edges, and the lone C-vertex itself doesn't need to be in a clique unless
        # it has edges. It has C-C edges (to other groups) -> covered by 2-cliques. So a singleton
        # C-group is OK as long as k>=2 (it has C-C edges). If k==1 edge case not in specs.
        # However the group-clique itself (G + packed I) must have size >=2. A singleton C-group
        # with no packed I-vertex yields a size-1 group-clique -> we must NOT emit it; instead the
        # lone C-vertex's edges are all cross-group C-C 2-cliques, so we skip emitting the group-clique.
        # We'll handle emission carefully.

        def cost_of_partition(groups):
            m = len(groups)
            # cross-group C-C pairs
            cross_cc = 0
            for a in range(m):
                ga = groups[a]
                for b in range(a + 1, m):
                    cross_cc += len(ga) * len(groups[b])
            # unpacked I-vertices
            gsets = [set(g) for g in groups]
            unpacked_sum = 0
            for s in shadow:
                if len(s) == 0:
                    continue
                packed = any(s <= gs for gs in gsets)
                if not packed:
                    unpacked_sum += len(s)
            return m + cross_cc + unpacked_sum

        rng = random.Random(98765)

        def random_partition(k, max_groups):
            g = rng.randint(1, max_groups)
            assign = [rng.randint(0, g - 1) for _ in range(k)]
            groups = [set() for _ in range(g)]
            for c, a in enumerate(assign):
                groups[a].add(c)
            groups = [frozenset(x) for x in groups if x]
            return groups if groups else [frozenset(range(k))]

        def improve(groups, deadline):
            cur = list(groups)
            cur_cost = cost_of_partition(cur)
            best = list(cur)
            best_c = cur_cost
            n_iter = 0
            while time.time() < deadline:
                n_iter += 1
                if len(cur) == 0:
                    break
                a = rng.randrange(len(cur))
                verts = list(cur[a])
                if not verts:
                    continue
                v = verts[rng.randrange(len(verts))]
                b = rng.randrange(len(cur) + 1)
                new_cur = list(cur)
                new_a = set(new_cur[a]); new_a.discard(v)
                new_cur[a] = frozenset(new_a)
                if b < len(cur):
                    new_b = set(new_cur[b]); new_b.add(v)
                    new_cur[b] = frozenset(new_b)
                else:
                    new_cur.append(frozenset([v]))
                new_cur = [g for g in new_cur if g]
                if not new_cur:
                    continue
                nc = cost_of_partition(new_cur)
                if nc <= cur_cost:
                    cur = new_cur
                    cur_cost = nc
                    if nc < best_c:
                        best_c = nc
                        best = list(cur)
                if n_iter % 300 == 0 and rng.random() < 0.12:
                    cur = random_partition(k, max(1, len(cur) + rng.randint(-2, 3)))
                    cur_cost = cost_of_partition(cur)
                    if cur_cost < best_c:
                        best_c = cur_cost
                        best = list(cur)
            return best, best_c

        deadline_inner = deadline - 1.0
        maxg = max(1, k)
        best_groups = [frozenset(range(k))]
        best_cost = cost_of_partition(best_groups)
        for restart in range(60):
            if time.time() > deadline_inner:
                break
            start = random_partition(k, min(maxg, max(1, restart + 2)))
            g, c = improve(start, deadline_inner)
            if c < best_cost:
                best_cost = c
                best_groups = g

        # Build cliques
        cliques = []
        gsets = [set(g) for g in best_groups]
        # assign each I-vertex to a group containing S_i (if any)
        assign_i = {}
        for idx, s in enumerate(shadow):
            i = k + idx
            if len(s) == 0:
                continue
            for gi, gs in enumerate(gsets):
                if s <= gs:
                    assign_i[i] = gi
                    break
        # cross-group C-C 2-cliques
        for a in range(len(best_groups)):
            ga = best_groups[a]
            for b in range(a + 1, len(best_groups)):
                for c in ga:
                    for c2 in best_groups[b]:
                        cliques.append([c, c2])
        # unpacked I-vertex cross edges as 2-cliques
        for idx, s in enumerate(shadow):
            i = k + idx
            if len(s) == 0:
                continue
            if i not in assign_i:
                for c in s:
                    cliques.append([c, i])
        # group cliques (only if size >= 2)
        for gi, g in enumerate(best_groups):
            members = set(g)
            for i, a in assign_i.items():
                if a == gi:
                    members.add(i)
            members = sorted(members)
            if len(members) >= 2:
                cliques.append(members)
        # Convert to plain ints / lists
        cliques = [[int(v) for v in cl] for cl in cliques]
        return cliques

    deadline = time.time() + 118.0
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 1.0:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = solve(*spec, deadline - 1.0)
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
