import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)

    m = len(edges)
    # Normalize edge list
    norm_edges = []
    for u, v in edges:
        norm_edges.append((min(u, v), max(u, v)))
    # Map normalized edge -> index
    edge_to_idx = {}
    for i, (u, v) in enumerate(norm_edges):
        edge_to_idx[(u, v)] = i

    def edge_idx(u, v):
        return edge_to_idx.get((min(u, v), max(u, v)), -1)

    def time_ok():
        return time.time() < deadline

    def find_ham_cycle_fast(used_edges, adj, n, deadline, seed=None):
        """Fast Hamiltonian cycle finder with good heuristics."""
        if not time_ok():
            return None
        if seed is None:
            rng = random.Random()
        else:
            rng = random.Random(seed)

        # Precompute available neighbors
        avail = {}
        for v in range(n):
            avail[v] = [u for u in adj[v] if (min(u, v), max(u, v)) not in used_edges]
            if len(avail[v]) < 2:
                return None

        # Try multiple random starts
        for attempt in range(20):
            if not time_ok():
                return None

            # Choose start with minimum available degree
            start_candidates = sorted(range(n), key=lambda v: len(avail[v]))
            if attempt < 5:
                start = start_candidates[attempt % len(start_candidates)]
            else:
                start = rng.choice(start_candidates)

            remaining = set(range(n)) - {start}
            path = [start]

            def backtrack(path, remaining):
                if not time_ok():
                    return None
                if len(path) == n:
                    last = path[-1]
                    edge = (min(last, start), max(last, start))
                    if edge not in used_edges and last in adj[start]:
                        return path[:]
                    return None

                last = path[-1]
                candidates = []
                for v in remaining:
                    if v in adj[last]:
                        edge = (min(last, v), max(last, v))
                        if edge not in used_edges:
                            # Count future options
                            future_opts = 0
                            for w in avail[v]:
                                if w != last and (w in remaining or (w == start and len(path) == n - 1)):
                                    future_opts += 1
                            if future_opts >= 1:
                                candidates.append((future_opts, v))

                if not candidates:
                    return None

                # Sort by fewest options first (MRV heuristic)
                candidates.sort(key=lambda x: x[0])

                # Randomize among top candidates
                if len(candidates) > 1 and rng.random() < 0.3:
                    top_k = min(3, len(candidates))
                    candidates = candidates[:top_k]
                    rng.shuffle(candidates)

                for _, v in candidates:
                    if not time_ok():
                        return None
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        path.append(v)
                        remaining.remove(v)
                        result = backtrack(path, remaining)
                        if result is not None:
                            return result
                        path.pop()
                        remaining.add(v)
                return None

            result = backtrack(path, remaining)
            if result is not None:
                return result

        return None

    def try_ilp_residual(remaining_edges, max_cycles):
        """
        ILP on remaining edges to find edge-disjoint Hamiltonian cycles.
        remaining_edges: list of (u,v) tuples that are still available
        """
        import numpy as np
        from scipy.optimize import LinearConstraint, milp, Bounds

        if not time_ok():
            return None

        # Build edge list for remaining edges
        rem_edges = []
        rem_edge_set = set()
        for e in remaining_edges:
            ne = (min(e[0], e[1]), max(e[0], e[1]))
            if ne not in rem_edge_set:
                rem_edge_set.add(ne)
                rem_edges.append(ne)

        m_rem = len(rem_edges)
        if m_rem < n * max_cycles:
            return None

        # Build adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Check each vertex has degree >= 2*max_cycles
        for v in range(n):
            if len(rem_adj[v]) < 2 * max_cycles:
                return None

        K = max_cycles
        def var(e, c):
            return e * K + c

        nvars = m_rem * K

        # Objective: maximize number of edges used
        c_obj = np.full(nvars, -1.0)

        lb = np.zeros(nvars)
        ub = np.ones(nvars)
        bounds = Bounds(lb, ub)
        integrality = np.ones(nvars)

        A_rows = []
        lb_rows = []
        ub_rows = []

        # Degree constraints
        for v in range(n):
            incident = [e for e in range(m_rem) if rem_edges[e][0] == v or rem_edges[e][1] == v]
            for c in range(K):
                row = np.zeros(nvars)
                for e in incident:
                    row[var(e, c)] = 1.0
                A_rows.append(row)
                lb_rows.append(2.0)
                ub_rows.append(2.0)

        # Edge capacity
        for e in range(m_rem):
            row = np.zeros(nvars)
            for c in range(K):
                row[var(e, c)] = 1.0
            A_rows.append(row)
            lb_rows.append(0.0)
            ub_rows.append(1.0)

        if not A_rows:
            return None

        A = np.array(A_rows)
        lb_arr = np.array(lb_rows)
        ub_arr = np.array(ub_rows)

        # Solve with subtour elimination
        for iteration in range(15):
            if not time_ok():
                return None

            try:
                constraint = LinearConstraint(A, lb_arr, ub_arr)
                time_left = max(1, int(deadline - time.time()) - 5)
                res = milp(
                    c=c_obj,
                    constraints=constraint,
                    integrality=integrality,
                    bounds=bounds,
                    options={'time_limit': min(25, time_left), 'mip_rel_gap': 0.0}
                )
            except Exception:
                return None

            if not res.success or res.x is None:
                return None

            x = res.x
            cycles = []
            valid = True
            for c in range(K):
                cycle_edges = []
                for e in range(m_rem):
                    if abs(x[var(e, c)] - 1.0) < 1e-6:
                        cycle_edges.append(e)
                if len(cycle_edges) != n:
                    valid = False
                    break
                # Build cycle
                cycle_adj = defaultdict(list)
                for e in cycle_edges:
                    u, v = rem_edges[e]
                    cycle_adj[u].append(v)
                    cycle_adj[v].append(u)
                for v in range(n):
                    if len(cycle_adj[v]) != 2:
                        valid = False
                        break
                if not valid:
                    break
                # Trace
                start = min(cycle_adj.keys())
                path = [start]
                prev = None
                current = start
                for _ in range(n):
                    next_v = None
                    for nb in cycle_adj[current]:
                        if nb != prev:
                            next_v = nb
                            break
                    if next_v is None:
                        valid = False
                        break
                    path.append(next_v)
                    prev, current = current, next_v
                if path[-1] != path[0]:
                    valid = False
                    break
                path = path[:-1]
                if len(set(path)) != n:
                    valid = False
                    break
                cycles.append(path)

            if not valid:
                return None

            # Check all Hamiltonian
            all_ham = all(len(cycles[c]) == n and len(set(cycles[c])) == n for c in range(K))
            if all_ham:
                return cycles

            # Add subtour cuts
            added = False
            for c in range(K):
                S = set(cycles[c])
                if 2 <= len(S) <= n - 2:
                    row2 = np.zeros(nvars)
                    for e in range(m_rem):
                        u, v = rem_edges[e]
                        if (u in S) != (v in S):
                            row2[var(e, c)] = 1.0
                    A_rows.append(row2)
                    lb_rows.append(2.0)
                    ub_rows.append(np.inf)
                    added = True

            if not added:
                break

            A = np.array(A_rows)
            lb_arr = np.array(lb_rows)
            ub_arr = np.array(ub_rows)

        return None

    def try_backtracking(max_cycles):
        """Fallback: find edge-disjoint Hamiltonian cycles via backtracking."""
        if not time_ok():
            return None

        used_edges = set()
        cycles = []

        for c in range(max_cycles):
            if not time_ok():
                break
            cycle = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=c * 1000)
            if cycle is None:
                # Try more random restarts
                found = False
                for attempt in range(30):
                    if not time_ok():
                        break
                    cycle = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=c * 1000 + attempt + 100)
                    if cycle is not None:
                        found = True
                        break
                if not found:
                    return cycles if cycles else None
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append(cycle)

        return cycles

    # Strategy: First find one Hamiltonian cycle via fast heuristic,
    # then use ILP on the residual graph to find additional cycles.
    best_cycles = None

    # Step 1: Find first Hamiltonian cycle quickly
    used_edges = set()
    cycle1 = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=42)
    if cycle1 is None:
        for s in range(50):
            if not time_ok():
                break
            cycle1 = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=s)
            if cycle1 is not None:
                break

    if cycle1 is not None:
        for i in range(n):
            u = cycle1[i]
            v = cycle1[(i + 1) % n]
            used_edges.add((min(u, v), max(u, v)))
        best_cycles = [cycle1]

        # Step 2: Try ILP on residual for 2 more cycles
        if time_ok():
            remaining = [e for e in norm_edges if e not in used_edges]
            result = try_ilp_residual(remaining, 2)
            if result is not None and len(result) >= 2:
                best_cycles = best_cycles + result[:2]
            elif result is not None and len(result) >= 1:
                best_cycles = best_cycles + result[:1]

        # Step 3: If ILP didn't find enough, try backtracking for more
        if len(best_cycles) < 3 and time_ok():
            for c in range(len(best_cycles), 3):
                cycle = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=c * 500 + 7)
                if cycle is None:
                    for s in range(20):
                        if not time_ok():
                            break
                        cycle = find_ham_cycle_fast(used_edges, adj, n, deadline, seed=c * 500 + 7 + s)
                        if cycle is not None:
                            break
                if cycle is not None:
                    for i in range(n):
                        u = cycle[i]
                        v = cycle[(i + 1) % n]
                        used_edges.add((min(u, v), max(u, v)))
                    best_cycles.append(cycle)
                else:
                    break
    else:
        # Fallback: pure backtracking
        for max_c in [3, 2, 1]:
            if not time_ok():
                break
            result = try_backtracking(max_c)
            if result is not None and len(result) >= max_c:
                best_cycles = result[:max_c]
                break

    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Last resort fallback
        cycle = find_ham_cycle_fast(set(), adj, n, deadline, seed=0)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
