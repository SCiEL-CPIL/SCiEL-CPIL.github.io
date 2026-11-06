import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        remaining_edges = set(edges)
        pieces = []

        def get_adjacency(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_shortest_cycle(adj, remaining_edges):
            """Find the shortest cycle in the graph. Returns list of vertices or None."""
            if not remaining_edges:
                return None

            best_cycle = None
            best_len = float('inf')

            # For each edge, check if removing it still connects its endpoints
            # If so, we have a cycle. Find the shortest such cycle.
            for u, v in list(remaining_edges):
                if time.time() > time_limit:
                    break

                # Temporarily remove edge (u, v)
                remaining_edges.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)

                # BFS from u to find shortest path to v
                if v in adj[u]:
                    # Should not happen since we removed it, but check
                    pass

                queue = deque([(u, [u])])
                visited = {u}
                found_path = None

                while queue:
                    if time.time() > time_limit:
                        break
                    node, path = queue.popleft()
                    if node == v:
                        found_path = path
                        break

                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [neighbor]
                            # Prune: if current path length already >= best, stop
                            if len(new_path) >= best_len:
                                continue
                            queue.append((neighbor, new_path))

                # Restore edge (u, v)
                remaining_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)

                if found_path:
                    # Cycle is found_path (vertices u, ..., v), closing edge (v, u)
                    cycle_len = len(found_path)
                    if cycle_len >= 3 and cycle_len < best_len:
                        best_len = cycle_len
                        best_cycle = list(found_path)
                        # If we found a triangle, it's optimal, stop early
                        if best_len == 3:
                            break

            return best_cycle

        while remaining_edges:
            if time.time() > time_limit:
                break

            adj = get_adjacency(remaining_edges)
            cycle = find_shortest_cycle(adj, remaining_edges)

            if cycle:
                pieces.append(cycle)
                # Remove edges of the cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    if (u, v) in remaining_edges:
                        remaining_edges.remove((u, v))
                    elif (v, u) in remaining_edges:
                        remaining_edges.remove((v, u))
                u, v = cycle[-1], cycle[0]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            else:
                # No more cycles, output all remaining edges as single edges
                for edge in remaining_edges:
                    pieces.append(list(edge))
                remaining_edges.clear()
                break

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time: give more time to larger/more complex graphs
        remaining_time = deadline - time.time()
        graphs_left = len(graphs) - i
        time_per_graph = max(1.0, remaining_time / graphs_left * 0.9)
        time_limit = time.time() + time_per_graph

        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

        # If time is critical, fallback to trivial for remaining
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
    print(f"Total pieces: {sum(len(r) for r in res)}")
