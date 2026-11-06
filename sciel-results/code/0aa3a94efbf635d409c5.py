import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        # Use bitmasks for adjacency to speed up operations
        # adj_mask[u] is a bitmask where bit v is 1 if (u, v) is an edge
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        remaining_edges = set(edges)
        cliques = []

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add remaining edges as individual cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Build the subgraph of remaining edges
            # We want to find the largest clique in this subgraph.
            # Greedy strategy:
            # 1. Compute degrees in the subgraph.
            # 2. Pick the vertex with the highest degree.
            # 3. Intersect its neighbors with the current candidate set.
            # 4. Repeat.

            # Compute degrees in the remaining subgraph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            # Candidates are all vertices with degree > 0
            candidates = set(u for u in range(n) if deg[u] > 0)
            if not candidates:
                break

            # Greedily build a clique
            clique = []
            # Start with the vertex of highest degree
            start_v = max(candidates, key=lambda x: deg[x])
            clique.append(start_v)
            candidates.discard(start_v)

            # The clique must be a subset of the neighbors of start_v
            # So we restrict candidates to neighbors of start_v in the remaining graph
            candidates &= set(adj_mask[start_v] & ((1 << n) - 1) and [i for i in range(n) if (adj_mask[start_v] >> i) & 1 and i in candidates])
            # Wait, the above line is inefficient. Let's do it properly.
            candidates = set(i for i in candidates if (adj_mask[start_v] >> i) & 1)

            current_clique_mask = (1 << start_v)
            
            while candidates:
                if time.time() > deadline:
                    break
                
                # Find the vertex in candidates with the highest degree
                # within the subgraph induced by candidates
                # AND connected to all vertices in the current clique
                # Actually, since we maintain candidates as neighbors of all clique members,
                # we just need the highest degree in the induced subgraph of candidates.
                
                best_v = -1
                best_deg = -1
                for v in candidates:
                    # Count neighbors of v in candidates
                    # v's neighbors in the remaining graph are given by adj_mask[v]
                    # We only care about neighbors that are in candidates
                    # candidates is a set, so we can iterate
                    # But for speed, let's use bitmask if possible, or just simple loop
                    # n is small (32, 64), so simple loop is fine
                    deg_v = 0
                    for u in candidates:
                        if u == v: continue
                        if (adj_mask[v] >> u) & 1:
                            deg_v += 1
                    if deg_v > best_deg:
                        best_deg = deg_v
                        best_v = v
                
                if best_v == -1:
                    break

                clique.append(best_v)
                candidates.discard(best_v)
                current_clique_mask |= (1 << best_v)
                
                # Update candidates: they must be neighbors of best_v
                candidates = set(u for u in candidates if (adj_mask[best_v] >> u) & 1)

            # Add the found clique to the result
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges in this clique from remaining_edges
                for u, v in combinations(clique, 2):
                    e = (u, v) if u < v else (v, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            elif len(clique) == 1:
                # If we only found a single vertex, it means no edges left for it?
                # No, if candidates was empty, it means no edges connected to it in the subgraph?
                # Actually, if clique size is 1, it means no other vertex was connected to it in the remaining graph.
                # But we picked it because it had degree > 0.
                # This case shouldn't happen if we pick correctly, but if it does,
                # we should just leave the edges for the next iteration.
                # However, if we can't form a clique of size >= 2, we must pick an edge.
                # Let's handle the case where the greedy clique is too small.
                pass

        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 140

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        remaining_edges = set(edges)
        cliques = []

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Compute degrees in the remaining subgraph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            candidates = set(u for u in range(n) if deg[u] > 0)
            if not candidates:
                break

            # Greedily build a clique
            clique = []
            start_v = max(candidates, key=lambda x: deg[x])
            clique.append(start_v)
            candidates.discard(start_v)
            
            # Restrict candidates to neighbors of start_v
            candidates = set(i for i in candidates if (adj_mask[start_v] >> i) & 1)

            while candidates:
                if time.time() > deadline:
                    break
                
                # Find vertex in candidates with highest degree in induced subgraph
                best_v = -1
                best_deg = -1
                for v in candidates:
                    deg_v = 0
                    for u in candidates:
                        if u == v: continue
                        if (adj_mask[v] >> u) & 1:
                            deg_v += 1
                    if deg_v > best_deg:
                        best_deg = deg_v
                        best_v = v
                
                if best_v == -1:
                    break

                clique.append(best_v)
                candidates.discard(best_v)
                candidates = set(u for u in candidates if (adj_mask[best_v] >> u) & 1)

            if len(clique) >= 2:
                cliques.append(clique)
                for u, v in combinations(clique, 2):
                    e = (u, v) if u < v else (v, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            else:
                # Fallback: pick an arbitrary edge
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.remove(e)

        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]
# EVOLVE-BLOCK-END
