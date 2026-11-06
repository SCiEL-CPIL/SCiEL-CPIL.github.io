import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle_bfs(n, adj, remaining_edges):
        """
        Find a long cycle using BFS-based approach.
        We look for cycles by finding two paths from a start vertex that meet.
        """
        if not remaining_edges:
            return None

        # Get vertices with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return None

        best_cycle = None
        rng = random.Random(12345)

        for attempt in range(80):
            if time.time() > deadline - 10:
                break

            start = rng.choice(candidates)

            # BFS from start, tracking parent to reconstruct paths
            # We want to find a cycle: start -> ... -> start
            # Use BFS to find shortest path back to start (which is trivially length 0)
            # Instead, we do a modified BFS: find a path from a neighbor of start back to start
            # that doesn't use the first edge.

            # Better approach: for each neighbor of start, do BFS to find a path back to start
            # that doesn't use the edge (start, neighbor)
            found = False
            for neighbor in adj[start]:
                if (min(start, neighbor), max(start, neighbor)) not in remaining_edges:
                    continue

                # BFS from neighbor, avoiding the edge back to start initially
                # We want to find a path from neighbor to start that uses >= 2 edges
                # (i.e., neighbor -> ... -> start where the path has at least 2 edges)
                # This means the cycle is start -> neighbor -> ... -> start

                visited = {start, neighbor}
                parent = {neighbor: None}  # parent[neighbor] = None means it's the root of BFS
                queue = deque([neighbor])
                found_path = False

                while queue and not found_path:
                    curr = queue.popleft()
                    for nei in adj[curr]:
                        if nei == start:
                            if curr != neighbor:  # Path has at least 2 edges: neighbor->curr->start
                                # Reconstruct path
                                path = [start]
                                node = curr
                                while node is not None:
                                    path.append(node)
                                    node = parent[node]
                                path.append(start)  # Close the cycle
                                # path is [start, ..., curr, neighbor, start]? No.
                                # Let's trace: BFS tree rooted at neighbor.
                                # parent[neighbor] = None
                                # parent[curr] = some node on path from neighbor
                                # The path from neighbor to curr is: neighbor -> ... -> curr
                                # So cycle is: start -> neighbor -> ... -> curr -> start
                                cycle = [start]
                                node = curr
                                while node != neighbor:
                                    cycle.append(node)
                                    node = parent[node]
                                cycle.append(neighbor)
                                # cycle is [start, ..., neighbor] and we close to start
                                # Check distinctness (should be fine with BFS tree)
                                if len(cycle) >= 3:
                                    # Verify all edges exist
                                    valid = True
                                    for i in range(len(cycle)):
                                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                                        if (min(u, v), max(u, v)) not in remaining_edges:
                                            valid = False
                                            break
                                    if valid:
                                        if best_cycle is None or len(cycle) > len(best_cycle):
                                            best_cycle = cycle
                                        found_path = True
                                        break
                        elif nei not in visited:
                            visited.add(nei)
                            parent[nei] = curr
                            queue.append(nei)

                if found_path:
                    break

            if best_cycle and len(best_cycle) > n * 0.7:
                break

        return best_cycle

    def find_cycle_greedy(n, adj, remaining_edges):
        """
        Greedy cycle finding: start from a vertex, extend path greedily.
        """
        if not remaining_edges:
            return None

        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return None

        best_cycle = None
        rng = random.Random(67890)

        for attempt in range(60):
            if time.time() > deadline - 10:
                break

            start = rng.choice(candidates)
            visited = {start}
            path = [start]
            prev = -1

            # Greedy walk: at each step, pick an unvisited neighbor (or start if we can close)
            curr = start
            while True:
                # Check if we can close the cycle
                if curr != start and (min(curr, start), max(curr, start)) in remaining_edges and len(path) >= 3:
                    cycle = path[:]
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break

                # Find next vertex
                next_vertex = None
                neighbors = list(adj[curr])
                rng.shuffle(neighbors)
                for nei in neighbors:
                    if (min(curr, nei), max(curr, nei)) not in remaining_edges:
                        continue
                    if nei == start:
                        if len(path) >= 3:
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            break
                        else:
                            continue
                    if nei not in visited:
                        next_vertex = nei
                        break

                if next_vertex is None or (curr != start and next_vertex == start and len(path) < 3):
                    break

                if next_vertex == start:
                    cycle = path[:]
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break

                visited.add(next_vertex)
                path.append(next_vertex)
                curr = next_vertex

            if best_cycle and len(best_cycle) > n * 0.7:
                break

        return best_cycle

    def decompose(edges, n):
        remaining = set(edges)
        pieces = []

        # Build initial adjacency
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break

            # Try multiple cycle-finding strategies
            cycle = None

            # Strategy 1: BFS-based
            cycle = find_longest_cycle_bfs(n, adj, remaining)

            # Strategy 2: Greedy walk
            if not cycle:
                cycle = find_cycle_greedy(n, adj, remaining)

            # Strategy 3: Try simple 3-cycles
            if not cycle:
                for u in range(n):
                    if time.time() > deadline - 5:
                        break
                    for v in adj[u]:
                        if v <= u:
                            continue
                        if (min(u, v), max(u, v)) not in remaining:
                            continue
                        for w in adj[v]:
                            if w == u:
                                continue
                            if (min(v, w), max(v, w)) not in remaining:
                                continue
                            if (min(u, w), max(u, w)) in remaining:
                                cycle = [u, v, w]
                                break
                        if cycle:
                            break
                    if cycle:
                        break

            if not cycle:
                break

            # Add cycle to pieces
            pieces.append(cycle)

            # Remove cycle edges from remaining and update adjacency
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
                    # Update adjacency
                    adj[u].remove(v)
                    adj[v].remove(u)

        # Remaining edges are singletons
        for u, v in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Fallback: just use all edges as singletons
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
