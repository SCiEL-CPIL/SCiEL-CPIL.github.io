"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivoting."""
        cliques = []
        R = set()
        P = set(range(n))
        X = set()

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Pivot selection: choose u in P ∪ X with max |R ∩ N(u)|
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda u: len(R & adj[u]))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)

        bk(R, P, X)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy set cover: pick vertex that hits most uncovered cliques."""
        if not cliques:
            return set()

        # Map vertex -> list of clique indices it belongs to
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].append(i)

        uncovered = set(range(len(cliques)))
        selected = set()

        while uncovered:
            if time.time() > DEADLINE:
                break
            # Pick vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            # Remove all cliques containing best_v from uncovered
            for ci in vertex_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)

        return selected

    def local_search(selected, cliques, n, time_limit):
        """Improve hitting set via removal and 2-exchange."""
        selected = set(selected)

        def is_valid(sel):
            for clique in cliques:
                if not (clique & sel):
                    return False
            return True

        # Phase 1: Try removing each vertex
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                if time.time() > time_limit:
                    break
                trial = selected - {v}
                if is_valid(trial):
                    selected = trial
                    improved = True

        # Phase 2: 2-exchange - try swapping one vertex for another
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                if time.time() > time_limit:
                    break
                # For each v in selected, try removing v and adding some w not in selected
                # to see if we can maintain validity with same or smaller size
                # Actually, 2-exchange: remove v, add w (w not in selected)
                # This doesn't reduce size, so skip unless we can also remove another
                # Better: try to find w that allows removal of v
                for w in range(n):
                    if w in selected:
                        continue
                    trial = (selected - {v}) | {w}
                    if len(trial) < len(selected):
                        if is_valid(trial):
                            selected = trial
                            improved = True
                            break
        return selected

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]

    for n, p, seed in instances:
        if time.time() > DEADLINE:
            # Return what we have with best effort
            results.append(list(range(n)))
            continue

        adj, edges = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)

        # Filter cliques with at least 2 vertices (already done in BK)
        if not cliques:
            results.append([])
            continue

        # Greedy hitting set
        selected = greedy_hitting_set(cliques, n)

        # Local search with time budget
        remaining = DEADLINE - time.time()
        ls_time = min(remaining * 0.5, 10.0)  # up to 10s for local search
        selected = local_search(selected, cliques, n, time.time() + ls_time)

        results.append(sorted(selected))

    return results
# EVOLVE-BLOCK-END
