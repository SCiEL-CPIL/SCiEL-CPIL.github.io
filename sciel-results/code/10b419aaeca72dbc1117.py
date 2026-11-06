import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, time_budget):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))

        cliques = []
        remaining_edges = set(edge_set)

        def get_adj(rem_edges):
            adj = [set() for _ in range(n)]
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_best_clique(rem_edges, adj, time_limit):
            """Find a clique using the minimum common neighborhood heuristic."""
            if time.time() > time_limit:
                return None

            if not rem_edges:
                return None

            # Find the edge with the minimum number of common neighbors
            min_common = float('inf')
            target_edge = None
            for u, v in rem_edges:
                common = adj[u] & adj[v]
                count = len(common)
                if count < min_common:
                    min_common = count
                    target_edge = (u, v)
                    if count == 0:
                        break

            if target_edge is None:
                return None

            u, v = target_edge
            
            R = {u, v}
            P = adj[u] & adj[v]
            
            while P:
                best_v = max(P, key=lambda x: len(adj[x] & P))
                R.add(best_v)
                P = P & adj[best_v]
                
                if time.time() > time_limit:
                    break

            if len(R) < 2:
                return list(target_edge)
            
            return list(R)

        # Main loop: iteratively find best clique and remove its edges
        while remaining_edges and time.time() < deadline:
            adj = get_adj(remaining_edges)
            best = find_best_clique(remaining_edges, adj, deadline)

            if best is None or len(best) < 2:
                e = next(iter(remaining_edges))
                best = list(e)

            # Remove edges in best clique
            clique_set = set(best)
            for u in best:
                for v in best:
                    if u < v:
                        e = (u, v)
                        remaining_edges.discard(e)
            
            cliques.append(best)

        # Post-processing: try to merge adjacent cliques or find better partitions
        if time.time() < deadline - 10:
            new_cliques = []
            used = [False] * len(cliques)
            for i in range(len(cliques)):
                if used[i]:
                    continue
                merged = set(cliques[i])
                for j in range(i + 1, len(cliques)):
                    if used[j]:
                        continue
                    is_clique = True
                    for v in cliques[j]:
                        for u in merged:
                            if v == u:
                                continue
                            e = (u, v) if u < v else (v, u)
                            if e not in edge_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        merged.update(cliques[j])
                        used[j] = True
                new_cliques.append(list(merged))
                used[i] = True

            covered = set()
            valid = True
            for clique in new_cliques:
                if len(clique) < 2:
                    valid = False
                    break
                for u, v in combinations(clique, 2):
                    e = (u, v) if u < v else (v, u)
                    if e not in edge_set:
                        valid = False
                        break
                    if e in covered:
                        valid = False
                        break
                    covered.add(e)
                if not valid:
                    break

            if valid and covered == edge_set:
                cliques = new_cliques

        return cliques

    c1 = solve(random_graph(32, 0.6, 601), deadline)
    c2 = solve(random_graph(64, 0.55, 602), deadline)

    return [c1, c2]
