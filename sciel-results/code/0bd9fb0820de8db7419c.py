import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 142.0

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    adj_mat = np.zeros((n, n), dtype=np.uint8)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
                edges.append((i, j))

    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)

    def count_c4_edge(u, v):
        """Count C4s containing edge (u,v) using matrix dot product."""
        nu = adj_mat[u].copy()
        nu[v] = 0
        nv = adj_mat[v].copy()
        nv[u] = 0
        return int(np.dot(nu, nv))

    def compute_total_c4():
        total = 0
        for (u, v) in edges:
            total += count_c4_edge(u, v)
        return total // 4

    # ---- Adjust to exactly 91 edges ----
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = float('inf')
            for (u, v) in list(edges):
                c = count_c4_edge(u, v)
                if c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            edge_set.discard(tuple(sorted(best)))
            adj_mat[best[0], best[1]] = 0
            adj_mat[best[1], best[0]] = 0
    elif m < 91:
        to_add = 91 - m
        for _ in range(to_add):
            best = None
            best_c = float('inf')
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    c = count_c4_edge(u, v)
                    if c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(best)
            adj_mat[best[0], best[1]] = 1
            adj_mat[best[1], best[0]] = 1

    total_c4 = compute_total_c4()
    best_edges = list(edges)
    best_total = total_c4

    def delta_swap(ru, rv, au, av):
        destroyed = count_c4_edge(ru, rv)
        adj_mat[ru, rv] = 0
        adj_mat[rv, ru] = 0
        created = count_c4_edge(au, av)
        adj_mat[ru, rv] = 1
        adj_mat[rv, ru] = 1
        return created - destroyed

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    nonedges = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]

    no_improve_count = 0
    max_no_improve = 80

    for it in range(1000000):
        if time.time() > deadline - 3.0:
            break

        # Score removal candidates
        rem_scores = []
        for (u, v) in edges:
            rem_scores.append((count_c4_edge(u, v), (u, v)))
        rem_scores.sort(key=lambda x: x[0])

        # Score addition candidates
        add_scores = []
        for (u, v) in nonedges:
            add_scores.append((count_c4_edge(u, v), (u, v)))
        add_scores.sort(key=lambda x: x[0])

        found = False
        top_k = 12
        for rc, re in rem_scores[:top_k]:
            if found:
                break
            ru, rv = re
            for ac, ae in add_scores[:top_k]:
                au, av = ae
                d = delta_swap(ru, rv, au, av)
                if d < 0:
                    adj_mat[ru, rv] = 0
                    adj_mat[rv, ru] = 0
                    adj_mat[au, av] = 1
                    adj_mat[av, au] = 1
                    edge_set.discard(tuple(sorted((ru, rv))))
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    nonedges = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]
                    total_c4 += d
                    found = True
                    break

        if found:
            no_improve_count = 0
            if total_c4 < best_total:
                best_total = total_c4
                best_edges = list(edges)
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                for _ in range(4):
                    if not nonedges:
                        break
                    re = edges[np.random.randint(len(edges))]
                    ae = nonedges[np.random.randint(len(nonedges))]
                    ru, rv = re
                    au, av = ae
                    d = delta_swap(ru, rv, au, av)
                    adj_mat[ru, rv] = 0
                    adj_mat[rv, ru] = 0
                    adj_mat[au, av] = 1
                    adj_mat[av, au] = 1
                    edge_set.discard(tuple(sorted((ru, rv))))
                    edge_set.add(tuple(sorted((au, av))))
                    edges.remove(re)
                    edges.append((au, av))
                    nonedges = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]
                    total_c4 += d
                no_improve_count = 0

    # Final polish
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                rem_scores = []
                for (u, v) in edges:
                    rem_scores.append((count_c4_edge(u, v), (u, v)))
                rem_scores.sort(key=lambda x: x[0])
                add_scores = []
                for (u, v) in nonedges:
                    add_scores.append((count_c4_edge(u, v), (u, v)))
                add_scores.sort(key=lambda x: x[0])

                for rc, re in rem_scores[:20]:
                    ru, rv = re
                    for ac, ae in add_scores[:20]:
                        au, av = ae
                        d = delta_swap(ru, rv, au, av)
                        if d < 0:
                            adj_mat[ru, rv] = 0
                            adj_mat[rv, ru] = 0
                            adj_mat[au, av] = 1
                            adj_mat[av, au] = 1
                            edge_set.discard(tuple(sorted((ru, rv))))
                            edge_set.add(tuple(sorted((au, av))))
                            edges.remove(re)
                            edges.append((au, av))
                            nonedges = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]
                            total_c4 += d
                            improved = True
                            break
                    if improved:
                        break

                if total_c4 < best_total:
                    best_total = total_c4
                    best_edges = list(edges)

    final_total = compute_total_c4()
    if final_total < best_total:
        best_total = final_total
        best_edges = list(edges)

    return [[e[0], e[1]] for e in best_edges]
