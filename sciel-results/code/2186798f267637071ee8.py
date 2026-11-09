import time
import random
import math

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build initial adjacency and edge set
    init_adj = [set() for _ in range(n)]
    init_edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        init_edge_set.add((u, v))
        init_adj[u].add(v)
        init_adj[v].add(u)

    def make_edge(u, v):
        return (u, v) if u < v else (v, u)

    def score_biclique(L, R):
        # Geometric mean of side sizes: rewards balanced bicliques over lopsided stars
        return math.sqrt(len(L) * len(R))

    def is_biclique(L, R, adj):
        for x in L:
            for y in R:
                if x == y or y not in adj[x]:
                    return False
        return True

    def expand_biclique(L, R, adj):
        """Expand L and R to a maximal biclique in the given adjacency."""
        L = set(L)
        R = set(R)
        if not L or not R:
            return L, R
        changed = True
        while changed:
            changed = False
            new_L = set()
            for x in L:
                if R.issubset(adj[x]):
                    new_L.add(x)
            if len(new_L) != len(L):
                L = new_L
                changed = True
            new_R = set()
            for y in R:
                if L.issubset(adj[y]):
                    new_R.add(y)
            if len(new_R) != len(R):
                R = new_R
                changed = True
        return L, R

    def get_max_biclique(u, v, adj):
        # Orientation 1: u in L, v in R
        L1 = set(adj[v])
        R1 = set(adj[u])
        if u not in L1 or v not in R1:
            size1 = 0
            L1, R1 = set(), set()
        else:
            L1, R1 = expand_biclique(L1, R1, adj)
            size1 = score_biclique(L1, R1)

        # Orientation 2: u in R, v in L
        L2 = set(adj[u])
        R2 = set(adj[v])
        if v not in L2 or u not in R2:
            size2 = 0
            L2, R2 = set(), set()
        else:
            L2, R2 = expand_biclique(L2, R2, adj)
            size2 = score_biclique(L2, R2)

        if size1 >= size2:
            return size1, list(L1), list(R1)
        else:
            return size2, list(L2), list(R2)

    def pivot_bicliques(adj, remaining, rng):
        """Build candidate bicliques by pivoting on each vertex (inspiration's common_neighbors)."""
        candidates = []
        for u in range(n):
            neighbors = set(v for v in adj[u] if make_edge(u, v) in remaining)
            if not neighbors:
                continue
            # Try u as singleton left with all neighbors as right, then expand
            L = {u}
            R = set(neighbors)
            L, R = expand_biclique(L, R, adj)
            if L and R and is_biclique(L, R, adj):
                covered = set()
                for x in L:
                    for y in R:
                        e = make_edge(x, y)
                        if e in remaining:
                            covered.add(e)
                if covered:
                    candidates.append((score_biclique(L, R), L, R, covered))
            # Try u as singleton right with all neighbors as left, then expand
            L2 = set(neighbors)
            R2 = {u}
            L2, R2 = expand_biclique(L2, R2, adj)
            if L2 and R2 and is_biclique(L2, R2, adj):
                covered = set()
                for x in L2:
                    for y in R2:
                        e = make_edge(x, y)
                        if e in remaining:
                            covered.add(e)
                if covered:
                    candidates.append((score_biclique(L2, R2), L2, R2, covered))
        return candidates

    def greedy_partition(adj, remaining, seed_rng=None):
        """Greedy set-cover: at each step pick the biclique covering most uncovered edges,
        using geometric-mean scoring to prefer balanced bicliques."""
        if seed_rng is not None:
            rng = seed_rng
        else:
            rng = random.Random()

        bicliques = []
        remaining = set(remaining)
        adj = [set(s) for s in adj]

        while remaining:
            best_score = -1
            best_L = None
            best_R = None
            best_edges = None

            # Candidate 1: edge-seed bicliques
            edges_list = list(remaining)
            rng.shuffle(edges_list)
            for u, v in edges_list:
                size, L, R = get_max_biclique(u, v, adj)
                if size > best_score:
                    best_score = size
                    best_L = L
                    best_R = R
                    covered = set()
                    for x in L:
                        for y in R:
                            e = make_edge(x, y)
                            if e in remaining:
                                covered.add(e)
                    best_edges = covered

            # Candidate 2: pivot bicliques (balanced, from inspiration)
            for size, L, R, covered in pivot_bicliques(adj, remaining, rng):
                if size > best_score:
                    best_score = size
                    best_L = list(L)
                    best_R = list(R)
                    best_edges = covered

            if best_L is None or best_score <= 0:
                # Fallback: single edge
                u, v = next(iter(remaining))
                bicliques.append([[u], [v]])
                remaining.discard(make_edge(u, v))
                continue

            bicliques.append([list(best_L), list(best_R)])

            # Remove covered edges
            for e in best_edges:
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)

        return bicliques

    def try_merge(bicliques, adj):
        """Try to merge any pair of bicliques if their union is still a valid biclique.
        Full pairwise scan (not just adjacent) to find all mergeable pairs."""
        changed = True
        while changed:
            changed = False
            new_bicliques = list(bicliques)
            i = 0
            while i < len(new_bicliques):
                merged = False
                for j in range(i + 1, len(new_bicliques)):
                    L1, R1 = new_bicliques[i]
                    L2, R2 = new_bicliques[j]
                    merged_L = set(L1) | set(L2)
                    merged_R = set(R1) | set(R2)
                    if is_biclique(merged_L, merged_R, adj):
                        new_bicliques[i] = [list(merged_L), list(merged_R)]
                        new_bicliques.pop(j)
                        changed = True
                        merged = True
                        break
                i += 1
            bicliques = new_bicliques
        return bicliques

    def local_search(bicliques, adj, rng):
        """Shuffle biclique order and re-merge to escape local merge orderings."""
        shuffled = list(bicliques)
        rng.shuffle(shuffled)
        return try_merge(shuffled, adj)

    def validate(bicliques, edge_set):
        """Check that bicliques form a valid partition."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            for x in L:
                for y in R:
                    if x == y:
                        return False
                    e = make_edge(x, y)
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set

    # Run multiple restarts with different random seeds
    best_sol = None
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    num_restarts = 300

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 13)
        sol = greedy_partition(init_adj, init_edge_set, seed_rng=rng)

        # Try to merge
        sol = try_merge(sol, init_adj)

        # Additional local search: shuffle and re-merge
        if time.time() < deadline - 2:
            sol = local_search(sol, init_adj, rng)

        if validate(sol, init_edge_set):
            if len(sol) < best_count:
                best_count = len(sol)
                best_sol = sol
                # Early exit if we found a very good solution
                if best_count <= 4:
                    break

    if best_sol is None:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = list(init_adj[u])
            if neighbors:
                result.append([[u], neighbors])
        return result

    # Convert to proper Python types
    final_result = []
    for L, R in best_sol:
        final_result.append([[int(x) for x in L], [int(x) for x in R]])

    return final_result
