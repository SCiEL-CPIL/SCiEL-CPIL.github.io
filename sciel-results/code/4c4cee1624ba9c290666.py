import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    m = len(edges)

    # Find all maximal bicliques
    def find_maximal_bicliques():
        bicliques = []
        # Try all pairs as seeds
        for u, v in edges:
            if u >= v:
                continue
            # Expand from seed (u, v)
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Expand L
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                for u_prime in range(n):
                    if u_prime not in L:
                        if (adj[u_prime] & mask_R) == mask_R:
                            L.add(u_prime)
                            changed = True
                # Expand R
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                for v_prime in range(n):
                    if v_prime not in R:
                        valid = True
                        for lu in L:
                            if not (adj[lu] & (1 << v_prime)):
                                valid = False
                                break
                        if valid:
                            R.add(v_prime)
                            changed = True

            # Check if this is a new maximal biclique
            L_frozenset = frozenset(L)
            R_frozenset = frozenset(R)
            is_new = True
            for existing_L, existing_R in bicliques:
                if L_frozenset <= existing_L and R_frozenset <= existing_R:
                    is_new = False
                    break
            if is_new:
                bicliques.append((L_frozenset, R_frozenset))

        # Remove non-maximal ones
        maximal = []
        for i, (L_i, R_i) in enumerate(bicliques):
            is_maximal = True
            for j, (L_j, R_j) in enumerate(bicliques):
                if i != j and L_i <= L_j and R_i <= R_j and (L_i != L_j or R_i != R_j):
                    is_maximal = False
                    break
            if is_maximal:
                maximal.append((L_i, R_i))

        return maximal

    # Precompute edge coverage for each biclique
    def get_biclique_edges(L, R):
        edge_set = set()
        for u in L:
            for v in R:
                if (u, v) in edge_index:
                    edge_set.add(edge_index[(u, v)])
                elif (v, u) in edge_index:
                    edge_set.add(edge_index[(v, u)])
        return frozenset(edge_set)

    # Find candidate bicliques (maximal + some smaller ones for flexibility)
    candidates = []
    max_bicliques = find_maximal_bicliques()
    for L, R in max_bicliques:
        edge_set = get_biclique_edges(L, R)
        if len(edge_set) > 0:
            candidates.append((L, R, edge_set))

    # Add single-edge bicliques as fallback
    for i, (u, v) in enumerate(edges):
        candidates.append(({u}, {v}, frozenset([i])))

    # Sort by size descending for better pruning
    candidates.sort(key=lambda x: len(x[2]), reverse=True)

    # Branch and bound for minimum set cover
    all_edges_mask = (1 << m) - 1
    best_solution = None
    best_count = m  # Upper bound: one biclique per edge

    # Greedy initial solution
    def greedy_solution():
        covered = 0
        solution = []
        remaining_candidates = list(range(len(candidates)))
        while covered != all_edges_mask:
            best_idx = -1
            best_new = 0
            for idx in remaining_candidates:
                new_edges = candidates[idx][2] & ~covered
                new_count = bin(new_edges).count('1')
                if new_count > best_new:
                    best_new = new_count
                    best_idx = idx
            if best_idx == -1 or best_new == 0:
                break
            solution.append(best_idx)
            covered |= candidates[best_idx][2]
            remaining_candidates.remove(best_idx)
        return solution, len(solution)

    greedy_sol, greedy_count = greedy_solution()
    if greedy_count < best_count:
        best_count = greedy_count
        best_solution = greedy_sol[:]

    # Lower bound: each biclique covers at most max_edge_count edges
    max_cover = max(len(c[2]) for c in candidates) if candidates else 1

    def lower_bound(remaining_mask):
        if remaining_mask == 0:
            return 0
        # Simple bound: ceil(remaining_edges / max_cover)
        remaining_count = bin(remaining_mask).count('1')
        return (remaining_count + max_cover - 1) // max_cover

    # Branch and bound
    def branch_and_bound(remaining_mask, current_count, current_solution):
        nonlocal best_solution, best_count

        if time.time() > deadline:
            return

        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_solution = current_solution[:]
            return

        # Prune if we can't beat best
        lb = lower_bound(remaining_mask)
        if current_count + lb >= best_count:
            return

        # Find an uncovered edge to branch on
        # Pick the edge with fewest covering candidates
        edge_indices = []
        mask = remaining_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            edge_indices.append(idx)
            mask ^= lsb

        # Choose edge with minimum candidate count
        best_edge = edge_indices[0]
        best_edge_candidates = None
        for e in edge_indices:
            candidates_for_e = []
            for ci, (L, R, edges) in enumerate(candidates):
                if edges & (1 << e):
                    candidates_for_e.append(ci)
            if best_edge_candidates is None or len(candidates_for_e) < len(best_edge_candidates):
                best_edge = e
                best_edge_candidates = candidates_for_e
                if len(candidates_for_e) <= 1:
                    break

        if not best_edge_candidates:
            return

        # Sort candidates by size descending
        best_edge_candidates.sort(key=lambda ci: len(candidates[ci][2]), reverse=True)

        for ci in best_edge_candidates:
            if time.time() > deadline:
                return

            L, R, edge_set = candidates[ci]
            new_remaining = remaining_mask & ~edge_set
            current_solution.append(ci)

            branch_and_bound(new_remaining, current_count + 1, current_solution)

            current_solution.pop()

            # If we found a solution with count 1, stop
            if best_count == 1:
                return

    # Start branch and bound
    branch_and_bound(all_edges_mask, 0, [])

    # Convert solution to bicliques
    if best_solution is None:
        # Fallback to greedy
        best_solution = greedy_sol

    result = []
    for ci in best_solution:
        L, R, _ = candidates[ci]
        left = [int(x) for x in sorted(L)]
        right = [int(x) for x in sorted(R)]
        result.append([left, right])

    return result
