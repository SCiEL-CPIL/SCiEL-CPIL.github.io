"""Stronger algorithm for edge path decomposition."""
import time
import sys
from collections import defaultdict, deque
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def odd_count(degs):
        """Count vertices with odd degree."""
        return sum(1 for d in degs.values() if d % 2 == 1)

    def greedy_decompose(edges, adj, n, time_limit):
        """
        Greedy decomposition with smart path extension.
        Strategy: maintain a pool of open paths. Always extend the path whose
        current endpoint has the fewest remaining neighbors (most constrained).
        When choosing next vertex, prefer one that keeps odd-degree count low.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        degs = defaultdict(int)
        for u, v in remaining:
            degs[u] += 1
            degs[v] += 1

        paths = []  # list of paths
        # Open endpoints: (constraint_score, path_index, end_flag)
        # constraint_score = number of remaining neighbors of the endpoint
        open_ends = []  # heap of (score, path_idx, is_head)

        def endpoint_score(v):
            """Lower score = more constrained = should extend first."""
            return sum(1 for nb in adj[v] if tuple(sorted((v, nb))) in remaining)

        def try_extend(path_idx, is_head):
            """Try to extend the path at the given end. Returns True if extended."""
            path = paths[path_idx]
            v = path[0] if is_head else path[-1]

            # Find candidate neighbors with remaining edges
            candidates = []
            for nb in adj[v]:
                e = tuple(sorted((v, nb)))
                if e in remaining:
                    # Score: prefer neighbor that reduces odd count
                    # If we remove edge (v,nb): degs[v]-1, degs[nb]-1
                    # Odd count change depends on current parity
                    old_odd = (degs[v] % 2) + (degs[nb] % 2)
                    new_odd = ((degs[v]-1) % 2) + ((degs[nb]-1) % 2)
                    odd_delta = new_odd - old_odd  # -2, 0, or +2
                    # Prefer lower odd_delta (reduces odd count)
                    # Also prefer higher degree neighbor to keep graph connected
                    candidates.append((odd_delta, -degs[nb], nb))

            if not candidates:
                return False

            candidates.sort()
            # Try the best candidate
            _, _, next_v = candidates[0]
            e = tuple(sorted((v, next_v)))
            remaining.remove(e)
            degs[v] -= 1
            degs[next_v] -= 1

            if is_head:
                path.insert(0, next_v)
            else:
                path.append(next_v)

            return True

        # Main loop
        while remaining:
            # Find vertex with max degree to start a new path
            if not paths or len(remaining) > 0:
                # Start new path from highest degree vertex
                start = max(degs.keys(), key=lambda v: degs[v]) if degs else None
                if start is None:
                    break

            # If we have open paths, try to extend the most constrained one
            if paths:
                # Rebuild heap (simpler than maintaining it)
                heap = []
                for i, path in enumerate(paths):
                    head_v = path[0]
                    tail_v = path[-1]
                    hs = endpoint_score(head_v)
                    ts = endpoint_score(tail_v)
                    # Only add if there are remaining edges from that endpoint
                    if hs > 0:
                        heap.append((hs, i, True))
                    if ts > 0:
                        heap.append((ts, i, False))

                if heap:
                    heapq.heapify(heap)
                    extended = False
                    while heap:
                        score, idx, is_head = heapq.heappop(heap)
                        if try_extend(idx, is_head):
                            extended = True
                            break
                    if extended:
                        continue

            # No extension possible, start a new path
            if not degs:
                break
            start = max(degs.keys(), key=lambda v: degs[v])
            path = [start]
            paths.append(path)

            # Extend this path greedily
            while True:
                v = path[-1]
                candidates = []
                for nb in adj[v]:
                    e = tuple(sorted((v, nb)))
                    if e in remaining:
                        old_odd = (degs[v] % 2) + (degs[nb] % 2)
                        new_odd = ((degs[v]-1) % 2) + ((degs[nb]-1) % 2)
                        odd_delta = new_odd - old_odd
                        candidates.append((odd_delta, -degs[nb], nb))
                if not candidates:
                    break
                candidates.sort()
                _, _, next_v = candidates[0]
                e = tuple(sorted((v, next_v)))
                remaining.remove(e)
                degs[v] -= 1
                degs[next_v] -= 1
                path.append(next_v)

            # Also try to extend from the head
            while True:
                v = path[0]
                candidates = []
                for nb in adj[v]:
                    e = tuple(sorted((v, nb)))
                    if e in remaining:
                        old_odd = (degs[v] % 2) + (degs[nb] % 2)
                        new_odd = ((degs[v]-1) % 2) + ((degs[nb]-1) % 2)
                        odd_delta = new_odd - old_odd
                        candidates.append((odd_delta, -degs[nb], nb))
                if not candidates:
                    break
                candidates.sort()
                _, _, next_v = candidates[0]
                e = tuple(sorted((v, next_v)))
                remaining.remove(e)
                degs[v] -= 1
                degs[next_v] -= 1
                path.insert(0, next_v)

            if time.time() > time_limit:
                break

        return paths, remaining

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        if used_edges != all_edges:
            return False
        return True

    def merge_improve(paths, edges, adj, n, time_limit):
        """
        Try to reduce path count by merging paths at shared endpoints.
        Also try 2-opt style moves: split a path and reattach pieces.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]

        def get_endpoints(path):
            return path[0], path[-1]

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Build endpoint map
            endpoint_map = defaultdict(list)
            for i, p in enumerate(current):
                if len(p) >= 2:
                    endpoint_map[p[0]].append((i, 'head'))
                    endpoint_map[p[-1]].append((i, 'tail'))

            # Try to merge two paths at a shared endpoint
            merged_any = False
            for v in list(endpoint_map.keys()):
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue

                for i in range(len(entries)):
                    idx1, role1 = entries[i]
                    for j in range(i+1, len(entries)):
                        idx2, role2 = entries[j]
                        if idx1 == idx2:
                            continue

                        p1 = current[idx1]
                        p2 = current[idx2]

                        new_path = None
                        # Determine how to merge
                        if role1 == 'tail' and role2 == 'head':
                            # p1 ends at v, p2 starts at v
                            if not any(x in p1 for x in p2[1:]):
                                new_path = p1 + p2[1:]
                        elif role1 == 'head' and role2 == 'tail':
                            # p1 starts at v, p2 ends at v
                            if not any(x in p2 for x in p1[1:]):
                                new_path = p1[1:] + p2
                        elif role1 == 'tail' and role2 == 'tail':
                            # Both end at v
                            if not any(x in p1 for x in p2[:-1]):
                                new_path = p1 + p2[::-1][1:]
                        elif role1 == 'head' and role2 == 'head':
                            # Both start at v
                            if not any(x in p2 for x in p1[1:]):
                                new_path = p1[::-1][1:] + p2

                        if new_path and len(set(new_path)) == len(new_path):
                            # Check all edges valid
                            valid = True
                            for k in range(len(new_path)-1):
                                e = tuple(sorted((new_path[k], new_path[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                # Perform merge
                                new_current = []
                                for idx, p in enumerate(current):
                                    if idx != idx1 and idx != idx2:
                                        new_current.append(p)
                                new_current.append(new_path)

                                if verify_decomposition(new_current, edges, n):
                                    current = new_current
                                    improved = True
                                    merged_any = True
                                    break
                    if merged_any:
                        break
                if merged_any:
                    break

            # Try to extend a path by absorbing another path's endpoint
            # If path A ends at v and path B starts at w, and edge (v,w) exists
            # and is not used, we can merge: A + [w] + B[1:]
            if not improved and time.time() < time_limit:
                for i, p1 in enumerate(current):
                    if len(p1) < 2:
                        continue
                    for j, p2 in enumerate(current):
                        if i == j or len(p2) < 2:
                            continue
                        # Try: p1's tail connects to p2's head via an edge
                        v = p1[-1]
                        w = p2[0]
                        e = tuple(sorted((v, w)))
                        if e in edge_set:
                            # Check if this edge is already used
                            used = False
                            for path in current:
                                for k in range(len(path)-1):
                                    if tuple(sorted((path[k], path[k+1]))) == e:
                                        used = True
                                        break
                                if used:
                                    break
                            if not used:
                                # Check vertex disjointness (except the connection)
                                if not any(x in p1 for x in p2[1:]) and not any(x in p2 for x in p1[:-1]):
                                    new_path = p1 + p2[1:]
                                    if len(set(new_path)) == len(new_path):
                                        new_current = []
                                        for idx, p in enumerate(current):
                                            if idx != i and idx != j:
                                                new_current.append(p)
                                        new_current.append(new_path)
                                        if verify_decomposition(new_current, edges, n):
                                            current = new_current
                                            improved = True
                                            break
                        if improved:
                            break
                    if improved:
                        break

        return current

    def local_search(paths, edges, adj, n, time_limit):
        """
        Local search: try to move a contiguous segment from one path to another,
        or re-route a path to absorb edges from another path.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        rng = Random(42)

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try: for each pair of paths, see if we can split one and merge
            # parts with the other to reduce total count
            for i in range(len(current)):
                p1 = current[i]
                if len(p1) < 3:
                    continue
                for j in range(len(current)):
                    if i == j:
                        continue
                    p2 = current[j]
                    if len(p2) < 2:
                        continue

                    # Try moving a prefix of p1 to connect with p2
                    # p1 = [a, b, c, ...], p2 = [..., x, y]
                    # If edge (a, y) exists and is unused, and vertices disjoint:
                    # new_p1 = p1[1:], new_p2 = [y] + p2[1:] + ... hmm complex

                    # Simpler: try to find if p1's head connects to p2's tail
                    # or p1's tail connects to p2's head via unused edge
                    # This is similar to merge_improve but let's try different orientations

                    # Try: reverse p1 and connect
                    rev_p1 = p1[::-1]
                    v = rev_p1[-1]  # original p1[0]
                    w = p2[-1]
                    e = tuple(sorted((v, w)))
                    if e in edge_set:
                        used = False
                        for path in current:
                            for k in range(len(path)-1):
                                if tuple(sorted((path[k], path[k+1]))) == e:
                                    used = True
                                    break
                            if used:
                                break
                        if not used:
                            if not any(x in rev_p1 for x in p2[:-1]):
                                new_path = rev_p1 + p2[:-1]
                                if len(set(new_path)) == len(new_path):
                                    new_current = []
                                    for idx, p in enumerate(current):
                                        if idx != i and idx != j:
                                            new_current.append(p)
                                    new_current.append(new_path)
                                    if verify_decomposition(new_current, edges, n):
                                        current = new_current
                                        improved = True
                                        break
                    if improved:
                        break
                if improved:
                    break

        return current

    def solve_one(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)

        # Phase 1: Greedy decomposition
        paths, remaining = greedy_decompose(edges, adj, n, time_limit)

        # If remaining edges exist (shouldn't), add them as single-edge paths
        if remaining:
            for e in remaining:
                paths.append(list(e))

        # Phase 2: Merge improvement
        if time.time() < time_limit - 5:
            paths = merge_improve(paths, edges, adj, n, time_limit - 5)

        # Phase 3: Local search
        if time.time() < time_limit - 2:
            paths = local_search(paths, edges, adj, n, time_limit - 2)

        # Verify
        if not verify_decomposition(paths, edges, n):
            # Fallback: each edge is its own path
            paths = [list(e) for e in edges]

        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        n, p, seed = spec
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Not enough time, use fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        paths = solve_one(n, p, seed, time.time() + remaining_time - 5)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
