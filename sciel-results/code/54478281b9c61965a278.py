import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115

    def get_elapsed():
        return time.time() - start_time

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)

        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        def greedy_partition(uncovered, vertex_order):
            uncovered = set(uncovered)
            partition = []

            while uncovered and get_elapsed() < DEADLINE - 5:
                best_clique = None
                best_count = 0

                for start_v in vertex_order:
                    neighbors = set()
                    for e in uncovered:
                        if start_v in e:
                            other = e[0] if e[1] == start_v else e[1]
                            neighbors.add(other)

                    if not neighbors:
                        continue

                    clique = [start_v]
                    candidates = set(neighbors)

                    sorted_cands = sorted(candidates, key=lambda c: len(candidates & adj[c]), reverse=True)

                    for cand in sorted_cands:
                        if cand not in candidates:
                            continue
                        can_add = all(cand in adj[c] for c in clique)
                        if can_add:
                            clique.append(cand)
                            candidates.discard(cand)
                            candidates &= adj[cand]

                    if len(clique) >= 2:
                        clique_set = set(clique)
                        covered = sum(1 for e in uncovered if e[0] in clique_set and e[1] in clique_set)
                        if covered > best_count:
                            best_count = covered
                            best_clique = list(clique)

                if best_clique is None or best_count == 0:
                    e = next(iter(uncovered))
                    best_clique = list(e)

                partition.append(best_clique)

                clique_set = set(best_clique)
                to_remove = {e for e in uncovered if e[0] in clique_set and e[1] in clique_set}
                uncovered -= to_remove

            for e in uncovered:
                partition.append(list(e))

            return partition

        best_partition = None
        best_size = float('inf')

        for order in [list(range(n)), list(range(n - 1, -1, -1))]:
            if get_elapsed() > DEADLINE - 10:
                break
            part = greedy_partition(edge_set, order)
            if len(part) < best_size:
                best_size = len(part)
                best_partition = part

        rng = random.Random(42)
        for _ in range(5):
            if get_elapsed() > DEADLINE - 10:
                break
            order = list(range(n))
            rng.shuffle(order)
            part = greedy_partition(edge_set, order)
            if len(part) < best_size:
                best_size = len(part)
                best_partition = part

        return best_partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)

    return [result1, result2]
