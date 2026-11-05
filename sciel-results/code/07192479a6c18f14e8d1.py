# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []

        # Precompute: for each vertex, which cliques it hits
        # cliques is a list of frozensets
        vertex_cliques = defaultdict(set)
        for idx, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(idx)
        
        # Convert to lists for faster iteration if needed, but sets are fine for intersection
        # Greedy Set Cover Initialization
        # We want to pick vertices that cover the most "uncovered" cliques.
        # Weight of a vertex v = number of cliques it belongs to.
        
        cover = set()
        uncovered = set(range(len(cliques)))
        
        # Precompute clique counts for vertices
        # This is static for the instance
        # v_count[v] = len(vertex_cliques[v])
        
        # To speed up greedy, we can maintain a list of (count, vertex) and update,
        # but for n<=40 and few cliques, a simple loop is often fast enough.
        # However, we can optimize by sorting vertices by initial degree in the "clique hypergraph".
        
        # Let's use a slightly smarter greedy:
        # 1. Calculate initial coverage size for each vertex.
        # 2. Pick the one with max coverage.
        # 3. Remove covered cliques.
        # 4. Repeat.
        
        # Optimization: Pre-compute vertex_cliques as lists for faster access?
        # Sets are fine.
        
        # Initial sort by number of cliques covered
        # We don't need to re-sort every time if we just scan all vertices each step (n is small).
        
        while uncovered and time.time() < time_budget - 10:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in cover:
                    continue
                # Count how many uncovered cliques this vertex hits
                # Intersection of vertex_cliques[v] and uncovered
                cnt = len(vertex_cliques[v] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                # Should not happen if uncovered is not empty and graph is valid
                break
                
            cover.add(best_v)
            # Update uncovered
            # uncovered -= vertex_cliques[best_v]
            uncovered.difference_update(vertex_cliques[best_v])
            
            if not uncovered:
                break

        # Local Search: 2-opt swaps
        # Try to replace one vertex in cover with one not in cover, if it reduces size or maintains size but allows further reduction?
        # Actually, standard local search for hitting set:
        # 1. Try to remove a vertex if it doesn't break validity (redundancy removal).
        # 2. Try to swap: remove v1, add v2, if it allows removal of v3? (3-opt)
        
        # Phase 1: Redundancy removal
        changed = True
        while changed and time.time() < time_budget - 5:
            changed = False
            for v in list(cover):
                test_cover = set(cover)
                test_cover.discard(v)
                # Check if test_cover is valid
                valid = True
                for c in cliques:
                    if not (c & test_cover):
                        valid = False
                        break
                if valid:
                    cover = test_cover
                    changed = True
                    break # Restart loop as indices/sets changed

        # Phase 2: 1-for-1 Swaps (might not reduce size immediately, but can enable further reductions)
        # Actually, for minimum hitting set, a swap that keeps size same is only useful if it enables a subsequent removal.
        # So we look for: remove v1, add v2, such that now some v3 can be removed.
        # This is a 3-opt move.
        
        if time.time() < time_budget - 2:
            improved = True
            while improved and time.time() < time_budget - 2:
                improved = False
                cover_list = list(cover)
                # Try all pairs (v1, v2) where v1 in cover, v2 not in cover
                for v1 in cover_list:
                    if time.time() > time_budget - 2:
                        break
                    for v2 in range(n):
                        if v2 in cover:
                            continue
                        # Candidate new cover: cover - {v1} + {v2}
                        new_cover = set(cover)
                        new_cover.discard(v1)
                        new_cover.add(v2)
                        
                        # Check if this new cover is valid
                        valid = True
                        for c in cliques:
                            if not (c & new_cover):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Now check if we can remove any vertex from new_cover to make it smaller
                        # If yes, accept the swap (size decreases by 1 effectively via the removal)
                        # Note: The swap itself keeps size constant. The removal makes it size-1.
                        # So we are looking for a path: Cover -> (swap) -> Cover' -> (remove) -> Cover''
                        # If Cover'' is valid and smaller, we take it.
                        
                        for v3 in list(new_cover):
                            test_final = set(new_cover)
                            test_final.discard(v3)
                            # Check validity of test_final
                            final_valid = True
                            for c in cliques:
                                if not (c & test_final):
                                    final_valid = False
                                    break
                            if final_valid:
                                # We found a better solution
                                cover = test_final
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        return sorted(cover)

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 30:
            # Fallback: return all vertices (valid but large)
            results.append(list(range(n)))
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 15 - elapsed) / max(1, remaining_instances)
        per_instance = max(2.0, min(per_instance, deadline - 15 - elapsed))

        sol = solve_instance(n, p, seed, start_time + elapsed + per_instance)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
