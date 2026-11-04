import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 100  # leave buffer for overhead

    # Generate support-size-2 candidates (220 vectors, all mutually compatible)
    candidates2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates2.append(v)
    candidates2 = np.array(candidates2, dtype=np.int64)
    n2 = len(candidates2)  # 220

    # Generate support-size-3 candidates (1320 vectors)
    candidates3 = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates3.append(v)
    candidates3 = np.array(candidates3, dtype=np.int64)
    n3 = len(candidates3)  # 1320

    # Combine all candidates
    all_candidates = np.vstack([candidates2, candidates3])
    n_total = len(all_candidates)

    # Compute norms squared
    norms_sq = np.sum(all_candidates ** 2, axis=1)

    # Build conflict matrix: conflict if dist^2 < max_norm^2
    # dist^2 = norm_i^2 + norm_j^2 - 2*dot(i,j)
    # Need dist^2 >= max_norm^2 for all pairs
    # For a set S, max_norm^2 = max(norm_sq[i] for i in S)
    # This is complex. Instead, for support-2 (norm^2=2) and support-3 (norm^2=3),
    # if we only use support-3 vectors, max_norm^2=3, need dist^2>=3, i.e., dot <= (3+3-3)/2 = 1.5, so dot <= 1
    # If mixing, it's more complex. Let's just use support-3 vectors.

    # For support-3 vectors: conflict if dot > 1
    dots3 = candidates3 @ candidates3.T
    conflict3 = (dots3 > 1)
    np.fill_diagonal(conflict3, False)

    # Build complement graph (compatibility graph) for maximum clique
    # compat[i,j] = True if not conflict3[i,j]
    compat = ~conflict3
    np.fill_diagonal(compat, False)

    # Tomita's maximum clique algorithm with pivot-based coloring
    # We'll work with the compatibility graph

    # Convert to adjacency sets
    adj = [set() for _ in range(n3)]
    for i in range(n3):
        for j in range(i + 1, n3):
            if compat[i, j]:
                adj[i].add(j)
                adj[j].add(i)

    best_size = 0
    best_clique = []

    # Tomita's algorithm: expand(R, P) where R is current clique, P is candidates
    # Uses greedy coloring to get upper bound

    def greedy_color(P_list, adj_sets):
        """Greedy coloring of subgraph induced by P_list. Returns list of colors."""
        colors = []
        colored = [False] * len(P_list)
        color_idx = 0
        uncolored = list(range(len(P_list)))
        while uncolored:
            color_idx += 1
            # Greedily assign color_idx to independent set
            current_color_set = []
            new_uncolored = []
            for idx in uncolored:
                v = P_list[idx]
                # Check if v conflicts with any in current_color_set
                can_color = True
                for w_idx in current_color_set:
                    w = P_list[w_idx]
                    if w in adj_sets[v]:  # They are adjacent in compat graph, so NOT independent
                        can_color = False
                        break
                if can_color:
                    current_color_set.append(idx)
                    colored[idx] = True
                else:
                    new_uncolored.append(idx)
            colors.extend([color_idx] * len(current_color_set))
            uncolored = new_uncolored
        return colors

    def expand(R, P_list):
        nonlocal best_size, best_clique
        if time.time() - start_time > time_limit:
            return

        if not P_list:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return

        # Greedy coloring to get upper bound
        colors = greedy_color(P_list, adj)
        max_color = max(colors) if colors else 0

        if len(R) + max_color <= best_size:
            return

        # Pivot-based selection: choose vertex with max degree in P
        # Build list of (vertex, color) sorted by color descending
        v_color_list = [(P_list[i], colors[i]) for i in range(len(P_list))]
        v_color_list.sort(key=lambda x: -x[1])

        # Process vertices in order of decreasing color
        for v, c in v_color_list:
            if time.time() - start_time > time_limit:
                return

            if len(R) + c <= best_size:
                break

            # Include v in clique
            R.append(v)
            # New P: vertices in P_list that are compatible with v
            P_new = [u for u in P_list if u != v and u in adj[v]]
            expand(R, P_new)
            R.pop()

            # Remove v from P_list for subsequent iterations
            P_list.remove(v)

    # Start with empty clique, all vertices as candidates
    # Order vertices by degree (descending) for better pruning
    degrees = [len(adj[i]) for i in range(n3)]
    order = sorted(range(n3), key=lambda x: -degrees[x])
    initial_P = order

    expand([], initial_P)

    # Also try with support-2 as baseline
    baseline = candidates2

    if best_clique and best_size > n2:
        result = candidates3[best_clique]
        return result
    else:
        return baseline
