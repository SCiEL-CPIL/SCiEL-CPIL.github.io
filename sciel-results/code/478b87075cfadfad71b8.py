"""
Cycle decomposition of sparse random graphs to minimize piece count.
Each piece is either a single edge or a cycle (>=3 distinct vertices).
We want to cover all edges with edge-disjoint cycles and as few leftover single edges as possible.
"""
import time
import random
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin under 150s

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n, time_budget=45.0):
        """
        Greedy + local search: repeatedly find cycles in the remaining edge set,
        remove them, and continue. Use randomized restarts with different strategies.
        """
        if not edges:
            return []

        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        def find_cycles_in_graph(adj, edge_set, num_cycles_to_find=None, random_seed=42):
            """
            Find cycles in the graph using randomized DFS/BFS approaches.
            Returns a list of cycles (each cycle is a list of vertices).
            """
            rng = random.Random(random_seed)
            cycles = []
            used_edges = set()

            # Strategy 1: Find cycles by looking for back-edges in DFS
            # Strategy 2: Random walk to find cycles
            # Strategy 3: For each vertex, try to find a cycle through it

            # We'll use a combination: repeatedly pick a random edge, try to extend
            # it into a cycle.

            remaining_edges = set(edge_set) - used_edges

            # Build adjacency for remaining edges
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

            # Try to find many cycles
            attempts = 0
            max_attempts = 500
            while remaining_edges and attempts < max_attempts:
                attempts += 1
                # Pick a random edge to start
                if not remaining_edges:
                    break
                e = rng.choice(list(remaining_edges))
                u, v = e

                # Try to find a cycle containing this edge
                # Do a BFS/DFS from v to u avoiding the direct edge
                # Actually, let's try to find a path from v to u in the graph
                # without using edge (u,v), then the path + (u,v) forms a cycle.

                # BFS from v to u
                parent = {v: None}
                queue = deque([v])
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    if cur == u:
                        found = True
                        break
                    for nb in rem_adj.get(cur, []):
                        if nb not in parent and not (nb == u and cur == v):  # avoid going back via the same edge
                            # Actually we need to be careful: we want a path from v to u
                            # that doesn't use edge (u,v) directly as the first step
                            # Let's just do standard BFS but skip the direct edge (u,v)
                            if (cur, nb) == (u, v) or (cur, nb) == (v, u):
                                # We can still traverse it if it's not the direct shortcut
                                # Actually for finding a cycle through edge (u,v),
                                # we need a path from u to v that doesn't use (u,v).
                                # So in BFS from v, we should not go directly to u via (v,u)
                                # unless we're doing something else. Let me reconsider.
                                pass
                            if nb not in parent:
                                parent[nb] = cur
                                queue.append(nb)

                if found:
                    # Reconstruct path from u to v
                    path = []
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        if cur == v:
                            break
                        cur = parent[cur]
                    path.reverse()  # now path is from v to u
                    # The cycle is path + closing edge (u,v)
                    # Vertices in cycle: path[0], path[1], ..., path[-1]
                    # Edges in cycle: (path[i], path[i+1]) for i in range(len(path)-1) plus (path[-1], path[0])
                    cycle_vertices = path
                    # Verify all edges exist in remaining_edges
                    valid = True
                    for i in range(len(cycle_vertices) - 1):
                        a, b = cycle_vertices[i], cycle_vertices[i + 1]
                        if (a, b) not in remaining_edges and (b, a) not in remaining_edges:
                            valid = False
                            break
                    if valid and len(cycle_vertices) >= 2:
                        # Check closing edge
                        a, b = cycle_vertices[-1], cycle_vertices[0]
                        if (a, b) in remaining_edges or (b, a) in remaining_edges:
                            # Remove all edges in the cycle
                            for i in range(len(cycle_vertices) - 1):
                                a, b = cycle_vertices[i], cycle_vertices[i + 1]
                                if (a, b) in remaining_edges:
                                    remaining_edges.remove((a, b))
                                elif (b, a) in remaining_edges:
                                    remaining_edges.remove((b, a))
                            a, b = cycle_vertices[-1], cycle_vertices[0]
                            if (a, b) in remaining_edges:
                                remaining_edges.remove((a, b))
                            elif (b, a) in remaining_edges:
                                remaining_edges.remove((b, a))
                            cycles.append(cycle_vertices)
                            if num_cycles_to_find is not None and len(cycles) >= num_cycles_to_find:
                                break
                        else:
                            valid = False
                    if not valid:
                        continue
                else:
                    continue

            return cycles

        def greedy_decompose(seed_offset=0):
            """
            Greedy approach: repeatedly find the longest cycle, remove it.
            """
            rng = random.Random(12345 + seed_offset)
            remaining = set(edge_set)
            pieces = []

            while remaining:
                # Build adjacency
                rem_adj = defaultdict(set)
                for u, v in remaining:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)

                # Try to find a long cycle
                # Use randomized approach: pick random start, do DFS to find cycle
                best_cycle = None
                for _ in range(20):
                    # Pick a random vertex with degree >= 2
                    candidates = [v for v in rem_adj if len(rem_adj[v]) >= 2]
                    if not candidates:
                        break
                    start = rng.choice(candidates)

                    # DFS to find a cycle
                    stack = [(start, [start], {start})]
                    found_cycle = None
                    while stack:
                        node, path, visited = stack.pop()
                        for nb in sorted(rem_adj.get(node, [])):
                            if nb == path[0] and len(path) >= 3:
                                # Found a cycle
                                if found_cycle is None or len(path) > len(found_cycle):
                                    found_cycle = path[:]
                                # Don't break; keep looking for longer
                            elif nb not in visited:
                                new_path = path + [nb]
                                new_visited = visited | {nb}
                                stack.append((nb, new_path, new_visited))
                        if found_cycle and len(found_cycle) > 10:
                            # Good enough
                            break

                    if found_cycle and (best_cycle is None or len(found_cycle) > len(best_cycle)):
                        best_cycle = found_cycle

                if best_cycle and len(best_cycle) >= 3:
                    # Remove edges in the cycle
                    for i in range(len(best_cycle) - 1):
                        a, b = best_cycle[i], best_cycle[i + 1]
                        if (a, b) in remaining:
                            remaining.remove((a, b))
                        elif (b, a) in remaining:
                            remaining.remove((b, a))
                    a, b = best_cycle[-1], best_cycle[0]
                    if (a, b) in remaining:
                        remaining.remove((a, b))
                    elif (b, a) in remaining:
                        remaining.remove((b, a))
                    pieces.append(best_cycle)
                else:
                    # No cycle found; take a single edge
                    e = rng.choice(list(remaining))
                    remaining.remove(e)
                    pieces.append([e[0], e[1]])

            return pieces

        # Try multiple strategies and keep the best
        best_pieces = None
        best_count = float('inf')

        strategies = [
            ("greedy", 0),
            ("greedy", 1),
            ("greedy", 2),
        ]

        for name, offset in strategies:
            if time.time() - start_time > time_budget:
                break
            pieces = greedy_decompose(offset)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces

        if best_pieces is None:
            # Fallback: all edges as single pieces
            best_pieces = [[u, v] for u, v in edge_set]

        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = TIME_LIMIT - elapsed
        if remaining_time < 5:
            # Not enough time; return trivial solution for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(n, p, seed)
        pieces = decompose(edges, n, time_budget=min(remaining_time - 2, 45.0))
        results.append(pieces)

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total pieces: {total}")
