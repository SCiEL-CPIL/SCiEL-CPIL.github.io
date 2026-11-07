import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        cliques = []
        covered_edges_in_C = set()
        covered_edge_count = defaultdict(int)  # track how many times each C-edge is covered

        # Sort I vertices by ascending degree in C
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors[v] & set(range(k))))

        for v in I_vertices:
            if time.time() > deadline:
                break
            N_v = neighbors[v] & set(range(k))
            if not N_v:
                continue

            # Find subset S of N_v maximizing size while minimizing conflicts
            # A conflict for edge (u,w) is when it's already covered
            S = set(N_v)
            
            # Greedy removal of conflicts - use conflict count as proxy
            changed = True
            while changed:
                changed = False
                # Calculate conflict count for each u in S
                conflict_count = {u: 0 for u in S}
                for u in S:
                    for w in S:
                        if u < w and (u, w) in covered_edges_in_C:
                            conflict_count[u] += 1
                            conflict_count[w] += 1
                
                if any(c > 0 for c in conflict_count.values()):
                    # Remove vertex with max conflicts
                    worst = max(conflict_count, key=lambda x: conflict_count[x])
                    S.discard(worst)
                    changed = True

            if len(S) >= 2:
                cliques.append([v] + sorted(S))
                for u1, u2 in combinations(sorted(S), 2):
                    covered_edges_in_C.add((u1, u2))
                    covered_edge_count[(u1, u2)] += 1
            elif len(S) == 1:
                u = next(iter(S))
                cliques.append([v, u])
            
            # Remaining neighbors of v must be covered by singletons
            for u in N_v:
                if u not in S:
                    cliques.append([v, u])

        # Handle remaining edges in C
        # Build graph of remaining edges - prioritize edges with higher coverage count
        rem_edges = {}
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    rem_edges[(u, w)] = 0  # uncovered edges have count 0
        
        # Process edges in order of decreasing "urgency" (higher coverage count first)
        # This helps find larger cliques for edges that are harder to cover
        edge_order = sorted(rem_edges.keys(), key=lambda e: -covered_edge_count.get(e, 0))
        
        for edge in edge_order:
            if time.time() > deadline:
                break
            
            if edge not in rem_edges:
                continue  # already covered by a previous clique in this phase
            
            u, w = edge
            clique_vertices = {u, w}
            
            # Expand clique - find vertices connected to all in clique_vertices
            candidates = []
            for x in range(k):
                if x in clique_vertices:
                    continue
                is_connected = True
                for y in clique_vertices:
                    if (min(x, y), max(x, y)) not in rem_edges:
                        is_connected = False
                        break
                if is_connected:
                    candidates.append(x)
            
            # Add all candidates to make a maximal clique
            clique_vertices.update(candidates)
            
            # Remove edges covered by this clique
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_edges.pop((u1, u2), None)
            
            cliques.append(sorted(clique_vertices))

        # Safety net - verify and fix any uncovered edges
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
        
        covered_all = set()
        for cl in cliques:
            for u1, u2 in combinations(cl, 2):
                covered_all.add((min(u1, u2), max(u1, u2)))
        
        uncovered = all_edges - covered_all
        unc_set = set(uncovered)
        while unc_set:
            u, w = next(iter(unc_set))
            clique_vertices = {u, w}
            candidates = []
            for x in range(n):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in unc_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)
            clique_vertices.update(candidates)
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                unc_set.discard((u1, u2))
            if len(clique_vertices) >= 2:
                cliques.append(sorted(clique_vertices))

        results.append([[int(x) for x in cl] for cl in cliques])

    return results
