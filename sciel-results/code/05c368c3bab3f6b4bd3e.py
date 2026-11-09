"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

DEADLINE = 78  # safety margin before 90s kill

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)

    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i

    def get_cycle_score(cycle):
        """Return the one-based max edge index in the cycle."""
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            idx = edge_pos[(u, v)]
            if idx + 1 > max_idx:
                max_idx = idx + 1
        return max_idx

    def build_adjacency(threshold):
        """Build adjacency list for edges with position < threshold, sorted by position."""
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        return adj

    def two_opt_single_pass(cycle, threshold):
        """Single forward pass of 2-opt to reduce max edge index."""
        cycle = list(cycle)
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[j + 1]
                if a == c or a == d or b == c or b == d:
                    continue
                old_max = max(edge_pos[(a, b)], edge_pos[(c, d)])
                new_max = max(edge_pos[(a, c)], edge_pos[(b, d)])
                if new_max < old_max and new_max < threshold:
                    cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
        return cycle

    def find_cycle_warnsdorff(threshold, start_vertex=0):
        """
        Find Hamiltonian cycle using backtracking with Warnsdorff heuristic.
        At each step, prefer the neighbor with fewest unvisited neighbors (dead-end first).
        Also uses forced-move detection.
        """
        adj = build_adjacency(threshold)
        # Build adjacency sets for O(1) lookup
        adj_set = [set(adj[u]) for u in range(n)]

        def backtrack(path, visited):
            if time.time() - start_time > DEADLINE:
                return None
            if len(path) == n:
                last = path[-1]
                if start_vertex in adj_set[last]:
                    return path[:]
                return None

            last = path[-1]
            # Get unvisited neighbors
            candidates = [v for v in adj[last] if not visited[v]]
            if not candidates:
                return None

            # Check for forced moves: if a candidate has no other unvisited neighbors
            # (and isn't the start), it's forced
            forced = None
            for v in candidates:
                if v == start_vertex and len(path) < n - 1:
                    continue
                # Count unvisited neighbors of v (excluding last which is visited)
                unvisited_count = 0
                for w in adj_set[v]:
                    if not visited[w] and w != last:
                        unvisited_count += 1
                    elif w == last:
                        pass  # last is visited, doesn't count
                    elif w == start_vertex and len(path) == n - 1:
                        # start can be the final connection
                        unvisited_count += 1
                if v != start_vertex and unvisited_count == 0 and len(path) < n - 1:
                    forced = v
                    break

            if forced is not None:
                path.append(forced)
                visited[forced] = True
                result = backtrack(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited[forced] = False
                return None

            # Warnsdorff: sort by number of unvisited neighbors (fewest first)
            def warnsdorff_key(v):
                if v == start_vertex and len(path) < n - 1:
                    return float('inf')  # can't go to start yet
                count = 0
                for w in adj_set[v]:
                    if not visited[w]:
                        count += 1
                return count

            candidates.sort(key=warnsdorff_key)

            for v in candidates:
                if v == start_vertex and len(path) < n - 1:
                    continue
                path.append(v)
                visited[v] = True
                result = backtrack(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited[v] = False
            return None

        visited = [False] * n
        visited[start_vertex] = True
        path = [start_vertex]
        return backtrack(path, visited)

    def assignment_based_cycle(threshold):
        """
        Use minimum-cost assignment to find a 2-factor, then merge cycles.
        """
        INF_COST = 1e9
        cost = np.full((n, n), INF_COST, dtype=float)
        for i in range(threshold):
            u, v = edge_order[i]
            cost[u][v] = i
            cost[v][u] = i
        np.fill_diagonal(cost, INF_COST)

        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r][c] >= INF_COST:
                return None

        next_v = {}
        for r, c in zip(row_ind, col_ind):
            next_v[r] = c

        # Find cycles
        visited = [False] * n
        cycles = []
        for s in range(n):
            if not visited[s]:
                cyc = []
                cur = s
                while not visited[cur]:
                    visited[cur] = True
                    cyc.append(cur)
                    cur = next_v[cur]
                cycles.append(cyc)

        if len(cycles) == 1:
            return cycles[0]

        # Merge cycles by edge swaps
        adj_set = [set() for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj_set[u].add(v)
            adj_set[v].add(u)

        merged = True
        while merged:
            if time.time() - start_time > DEADLINE:
                break
            merged = False
            for i in range(len(cycles)):
                for j in range(i + 1, len(cycles)):
                    c1 = cycles[i]
                    c2 = cycles[j]
                    found = False
                    for ai in range(len(c1)):
                        a = c1[ai]
                        b = c1[(ai + 1) % len(c1)]
                        for ci in range(len(c2)):
                            c = c2[ci]
                            d = c2[(ci + 1) % len(c2)]
                            if c in adj_set[a] and d in adj_set[b]:
                                # Remove a->b and c->d, add a->c and b->d
                                seg1 = c1[ai+1:] + c1[:ai+1]
                                seg2 = c2[ci:] + c2[:ci]
                                new_cycle = seg1 + seg2
                                if len(new_cycle) == len(c1) + len(c2):
                                    cycles[i] = new_cycle
                                    del cycles[j]
                                    merged = True
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        if len(cycles) == 1:
            return cycles[0]
        return None

    def try_find_cycle(threshold):
        """Try multiple methods to find a Hamiltonian cycle at given threshold."""
        # Method 1: Assignment-based
        cyc = assignment_based_cycle(threshold)
        if cyc:
            return cyc
        # Method 2: Warnsdorff backtracking from vertex 0
        cyc = find_cycle_warnsdorff(threshold, 0)
        if cyc:
            return cyc
        # Method 3: Try a few other starting vertices
        for s in [1, 2, 12, 13, 23]:
            if time.time() - start_time > DEADLINE:
                break
            cyc = find_cycle_warnsdorff(threshold, s)
            if cyc:
                return cyc
        return None

    # Binary search for minimum threshold
    best_cycle = None
    best_threshold = num_edges

    # First, find any feasible cycle (use full graph)
    cyc = try_find_cycle(num_edges)
    if cyc:
        best_cycle = cyc
        best_threshold = get_cycle_score(cyc)
    else:
        return list(range(n))

    # Binary search
    lo = n
    hi = best_threshold
    while lo < hi:
        if time.time() - start_time > DEADLINE - 5:
            break
        mid = (lo + hi) // 2
        cyc = try_find_cycle(mid)
        if cyc:
            best_cycle = cyc
            best_threshold = mid
            hi = mid
        else:
            lo = mid + 1

    # Polish with 2-opt (a few passes)
    if best_cycle:
        current_score = get_cycle_score(best_cycle)
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 2:
                break
            improved = two_opt_single_pass(best_cycle, current_score)
            new_score = get_cycle_score(improved)
            if new_score < current_score:
                best_cycle = improved
                current_score = new_score
            else:
                break

        # Try to reduce threshold further
        for t in range(current_score - 1, max(n - 1, current_score - 15), -1):
            if time.time() - start_time > DEADLINE - 2:
                break
            cyc = try_find_cycle(t)
            if cyc:
                improved = two_opt_single_pass(cyc, t)
                if get_cycle_score(improved) < current_score:
                    best_cycle = improved
                    current_score = get_cycle_score(improved)
                break

    return best_cycle if best_cycle else list(range(n))

# EVOLVE-BLOCK-END
