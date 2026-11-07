import time
import random

def construct():
    n = 15
    start_time = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - start_time

    def is_valid(adj, deg):
        # Group vertices by degree
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
        
        # Only check pairs of vertices with same degree
        for d, verts in deg_groups.items():
            if len(verts) < 2:
                continue
            for idx1 in range(len(verts)):
                for idx2 in range(idx1 + 1, len(verts)):
                    s = verts[idx1]
                    t = verts[idx2]
                    # Check if there's a simple path s-u-v-t
                    found = False
                    for u in adj[s]:
                        for v in adj[u]:
                            if v == s:
                                continue
                            if t in adj[v]:
                                # Check s-u-v-t is simple: s!=u, s!=v, s!=t, u!=v, u!=t, v!=t
                                if u != t and v != s and v != t:
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        return False
        return True

    def try_search(initial_edges, time_limit):
        edge_set = set()
        for u, v in initial_edges:
            key = (min(u, v), max(u, v))
            edge_set.add(key)

        edges = list(initial_edges)
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1

        # Repair if invalid
        while edges and not is_valid(adj, deg):
            e = edges.pop()
            u, v = e
            edge_set.discard((min(u,v), max(u,v)))
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1

        best_edges = edges[:]
        best_count = len(edges)

        improved = True
        while improved and elapsed() < time_limit:
            improved = False

            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_pairs)

            # Add edges
            for (u, v) in all_pairs:
                if elapsed() > time_limit:
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    continue

                nd = deg[:]
                nd[u] += 1
                nd[v] += 1
                na = [lst[:] for lst in adj]
                na[u].append(v)
                na[v].append(u)

                if is_valid(na, nd):
                    adj = na
                    deg = nd
                    edge_set.add(key)
                    edges.append([u, v])
                    improved = True
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = edges[:]

            # Swaps
            if not improved and elapsed() < time_limit:
                cur_list = list(edge_set)
                random.shuffle(cur_list)
                for (ru, rv) in cur_list:
                    if elapsed() > time_limit:
                        break

                    t_adj = [lst[:] for lst in adj]
                    t_deg = deg[:]
                    t_adj[ru].remove(rv)
                    t_adj[rv].remove(ru)
                    t_deg[ru] -= 1
                    t_deg[rv] -= 1
                    t_es = edge_set - {(ru, rv)}

                    if not is_valid(t_adj, t_deg):
                        continue

                    for (au, av) in all_pairs:
                        if elapsed() > time_limit:
                            break
                        akey = (min(au, av), max(au, av))
                        if akey in t_es:
                            continue

                        nd2 = t_deg[:]
                        nd2[au] += 1
                        nd2[av] += 1
                        na2 = [lst[:] for lst in t_adj]
                        na2[au].append(av)
                        na2[av].append(au)

                        if is_valid(na2, nd2):
                            adj = na2
                            deg = nd2
                            edge_set = t_es
                            edge_set.add(akey)
                            edges = [[u, v] for u, v in edge_set]
                            improved = True
                            if len(edges) > best_count:
                                best_count = len(edges)
                                best_edges = edges[:]
                            break
                    if improved:
                        break

        return best_edges, best_count

    seeds = []
    s1 = [[u, v] for u in range(7) for v in range(7, 15)]
    seeds.append(s1)
    for trial in range(3):
        s = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.6:
                    s.append([i, j])
        seeds.append(s)

    best_overall = []
    best_score = 0
    remaining = DEADLINE - elapsed()
    per_seed = remaining / len(seeds)

    for seed in seeds:
        if elapsed() > DEADLINE - 3:
            break
        tl = min(elapsed() + per_seed, DEADLINE - 3)
        res_edges, res_count = try_search(seed, tl)
        if res_count > best_score:
            best_score = res_count
            best_overall = res_edges

    return [[int(u), int(v)] for u, v in best_overall]
