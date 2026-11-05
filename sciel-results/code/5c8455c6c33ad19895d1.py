"""Strategy: Exploit clique structure (edges + triangles dominant) via MVC-based initialization with triangle repair and diverse-restart local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(n, adj):
        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            # Pivot: choose vertex in P ∪ X with max |P ∩ N(u)|
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
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

            P_set = set(P)
            X_set = set(X)

            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                bk(new_R, new_P, new_X)
                P_set.discard(v)
                X_set.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_instance(n, adj, cliques, time_budget):
        """
        Solve minimum hitting set for cliques (all size >= 2).
        Strategy:
          1. Greedy MVC on edges (size-2 cliques) with multiple orderings
          2. Repair: add vertices to cover triangles not yet hit
          3. Local search: remove redundancies, try swaps
          4. Multiple restarts with different randomizations
        """
        if not cliques:
            return set()

        # Separate cliques by size
        edge_cliques = [c for c in cliques if len(c) == 2]
        tri_cliques = [c for c in cliques if len(c) == 3]
        larger_cliques = [c for c in cliques if len(c) >= 4]

        # Precompute coverage maps
        # For each vertex, which edge-cliques it covers
        vertex_edge_cliques = defaultdict(list)
        for i, c in enumerate(edge_cliques):
            for v in c:
                vertex_edge_cliques[v].append(i)

        # For each vertex, which tri-cliques it covers
        vertex_tri_cliques = defaultdict(list)
        for i, c in enumerate(tri_cliques):
            for v in c:
                vertex_tri_cliques[v].append(i)

        # For each vertex, which larger cliques it covers
        vertex_large_cliques = defaultdict(list)
        for i, c in enumerate(larger_cliques):
            for v in c:
                vertex_large_cliques[v].append(i)

        def is_valid(selected):
            for c in cliques:
                hit = False
                for v in c:
                    if v in selected:
                        hit = True
                        break
                if not hit:
                    return False
            return True

        def greedy_mvc(edges_list, n, rng):
            """Greedy minimum vertex cover: repeatedly pick vertex covering most uncovered edges."""
            if not edges_list:
                return set()
            uncovered = set(range(len(edges_list)))
            selected = set()
            # Shuffle vertex order for diversity
            order = list(range(n))
            rng.shuffle(order)
            # Use degree-weighted random selection
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in order:
                    if v in selected:
                        continue
                    gain = 0
                    for ei in vertex_edge_cliques.get(v, []):
                        if ei in uncovered:
                            gain += 1
                    # Small random perturbation for diversity
                    gain += rng.random() * 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain <= 0.01:
                    break
                selected.add(best_v)
                for ei in vertex_edge_cliques.get(best_v, []):
                    if ei in uncovered:
                        uncovered.discard(ei)
            return selected

        def repair_triangles(selected, rng):
            """Add vertices to cover uncovered triangles."""
            selected = set(selected)
            changed = True
            while changed:
                changed = False
                for tc in tri_cliques:
                    hit = False
                    for v in tc:
                        if v in selected:
                            hit = True
                            break
                    if not hit:
                        # Pick the vertex in this triangle that covers the most other uncovered triangles
                        best_v = None
                        best_score = -1
                        for v in tc:
                            if v in selected:
                                continue
                            score = len(vertex_tri_cliques.get(v, [])) + rng.random() * 0.5
                            if score > best_score:
                                best_score = score
                                best_v = v
                        if best_v is not None:
                            selected.add(best_v)
                            changed = True
            return selected

        def remove_redundant(selected):
            """Remove any vertex whose removal keeps all cliques hit."""
            selected = set(selected)
            changed = True
            while changed:
                changed = False
                for v in list(selected):
                    test = set(selected)
                    test.discard(v)
                    # Check validity quickly
                    valid = True
                    for c in cliques:
                        hit = False
                        for u in c:
                            if u in test:
                                hit = True
                                break
                        if not hit:
                            valid = False
                            break
                    if valid:
                        selected = test
                        changed = True
            return selected

        def local_search(selected, rng, budget):
            """Aggressive local search: try removing non-critical vertices, swapping to reduce size."""
            selected = set(selected)
            best = set(selected)
            best_size = len(best)
            t = time.time()

            while time.time() - t < budget and best_size > 0:
                improved = False

                # Phase 1: Remove any redundant vertex
                for v in list(selected):
                    test = set(selected)
                    test.discard(v)
                    valid = True
                    for c in cliques:
                        hit = False
                        for u in c:
                            if u in test:
                                hit = True
                                break
                        if not hit:
                            valid = False
                            break
                    if valid:
                        selected = test
                        best = set(selected)
                        best_size = len(best)
                        improved = True
                        break

                if improved:
                    continue

                # Phase 2: Try swaps: remove v1, add v2, if valid and same or smaller size
                if selected:
                    # Find removable candidates (vertices not critical for any clique)
                    removable = []
                    for v in selected:
                        critical = False
                        for c in cliques:
                            if v in c:
                                others = [u for u in c if u != v]
                                if not any(u in selected for u in others):
                                    critical = True
                                    break
                        if not critical:
                            removable.append(v)

                    if removable:
                        v_rem = rng.choice(removable)
                        test = set(selected)
                        test.discard(v_rem)

                        # Find best vertex to add that covers the most newly-uncovered cliques
                        # Determine which cliques are now uncovered
                        uncovered_c = []
                        for i, c in enumerate(cliques):
                            if not any(u in test for u in c):
                                uncovered_c.append(c)

                        if not uncovered_c:
                            # Just removing works
                            selected = test
                            best = set(selected)
                            best_size = len(best)
                            improved = True
                        else:
                            # Score each candidate vertex by how many uncovered cliques it hits
                            cand_scores = {}
                            for v in range(n):
                                if v in test:
                                    continue
                                score = 0
                                for c in uncovered_c:
                                    if v in c:
                                        score += 1
                                if score > 0:
                                    cand_scores[v] = score

                            if cand_scores:
                                # Pick best (or random among top few)
                                max_score = max(cand_scores.values())
                                top = [v for v, s in cand_scores.items() if s >= max_score - 1]
                                v_add = rng.choice(top)
                                test2 = set(test)
                                test2.add(v_add)
                                if len(test2) <= len(selected):
                                    # Verify validity
                                    valid = True
                                    for c in cliques:
                                        if not any(u in test2 for u in c):
                                            valid = False
                                            break
                                    if valid:
                                        selected = test2
                                        if len(selected) < best_size:
                                            best = set(selected)
                                            best_size = len(best)
                                        improved = True

                if not improved:
                    # Try a random kick: remove a random vertex, re-repair
                    if len(selected) > 1:
                        v_kick = rng.choice(list(selected))
                        test = set(selected)
                        test.discard(v_kick)
                        # Quick repair: add back vertices to cover uncovered cliques
                        for c in cliques:
                            if not any(u in test for u in c):
                                # Add the vertex in c with highest degree in the clique set
                                best_v = max(c, key=lambda v: len(vertex_edge_cliques.get(v, [])) + len(vertex_tri_cliques.get(v, [])))
                                test.add(best_v)
                        if len(test) < best_size:
                            selected = test
                            best = set(selected)
                            best_size = len(best)
                            improved = True

                if time.time() - t > budget:
                    break

            return best

        # Main: multiple restarts
        best_solution = None
        best_size = n + 1
        t = time.time()
        restart_budget = max(0.5, time_budget - 2.0)
        num_restarts = max(5, int(restart_budget / 2.0))

        for restart in range(num_restarts):
            if time.time() > start_time + time_budget - 1.0:
                break
            rng = random.Random(1000 + restart * 7919)

            # Phase 1: Greedy MVC on edges
            if edge_cliques:
                selected = greedy_mvc(edge_cliques, n, rng)
            else:
                selected = set()

            # Phase 2: Repair triangles
            if tri_cliques:
                selected = repair_triangles(selected, rng)

            # Phase 3: Repair larger cliques
            if larger_cliques:
                for c in larger_cliques:
                    if not any(v in selected for v in c):
                        # Add the vertex in c with most total coverage
                        best_v = max(c, key=lambda v: len(vertex_edge_cliques.get(v, [])) + len(vertex_tri_cliques.get(v, [])) + len(vertex_large_cliques.get(v, [])))
                        selected.add(best_v)

            # Phase 4: Remove redundancies
            selected = remove_redundant(selected)

            # Phase 5: Local search
            remaining = max(0.5, time_budget - (time.time() - start_time) - 1.0)
            ls_budget = min(remaining, 3.0)
            if ls_budget > 0.1:
                selected = local_search(selected, rng, ls_budget)

            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)

            # Early stop if we hit a very good solution
            if best_size <= max(1, int(0.15 * n)):
                break

        if best_solution is None:
            # Fallback: all vertices
            best_solution = set(range(n))

        return best_solution

    results = []

    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        if time.time() > deadline - 15:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        per_instance = remaining / max(1, 3 - idx)
        per_instance = max(2.0, min(per_instance, remaining))

        sol = solve_instance(n, adj, cliques, per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
