import time
import random
import sys
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

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        def find_longest_cycle_in_subgraph(sub_edges, max_depth=20):
            """Find a long cycle in the subgraph defined by sub_edges using iterative DFS."""
            if not sub_edges:
                return None

            # Build local adjacency
            local_adj = defaultdict(set)
            for u, v in sub_edges:
                local_adj[u].add(v)
                local_adj[v].add(u)

            candidates = [v for v in local_adj if len(local_adj[v]) >= 2]
            if not candidates:
                return None

            best_cycle = None
            best_len = 0

            # Sort by degree descending
            candidates.sort(key=lambda x: len(local_adj[x]), reverse=True)

            for start in candidates[:15]:
                # Iterative DFS to find longest cycle starting from 'start'
                stack = [(start, [start], {start})]
                while stack:
                    current, path, visited = stack.pop()
                    if len(path) >= 3:
                        # Check if we can close
                        if (current, start) in sub_edges or (start, current) in sub_edges:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                    # Extend
                    for neighbor in local_adj[current]:
                        if neighbor == start:
                            continue
                        if neighbor not in visited:
                            new_path = path + [neighbor]
                            new_visited = visited | {neighbor}
                            # Prune: if even completing can't beat best, skip
                            if len(new_path) + 1 <= best_len:
                                continue
                            if len(new_path) >= max_depth:
                                continue
                            stack.append((neighbor, new_path, new_visited))

            return best_cycle

        # Strategy: Color the line graph greedily to partition edges into color classes.
        # Then extract longest cycle from each color class.
        # This is a different global approach vs. local peeling.

        # Greedy edge coloring: assign each edge a color such that adjacent edges get different colors.
        edge_list = list(remaining_edges)
        edge_color = {}
        color_used = set()

        # Sort edges by endpoint degrees to improve coloring
        def edge_key(e):
            u, v = e
            return (len(adj[u]), len(adj[v]))
        edge_list.sort(key=edge_key, reverse=True)

        for e in edge_list:
            u, v = e
            # Find forbidden colors from adjacent edges
            forbidden = set()
            for neighbor in adj[u]:
                e2 = (u, neighbor) if u < neighbor else (neighbor, u)
                if e2 in edge_color:
                    forbidden.add(edge_color[e2])
            for neighbor in adj[v]:
                e2 = (v, neighbor) if v < neighbor else (neighbor, v)
                if e2 in edge_color:
                    forbidden.add(edge_color[e2])

            # Assign smallest available color
            c = 0
            while c in forbidden:
                c += 1
            edge_color[e] = c
            color_used.add(c)

        # Group edges by color
        color_classes = defaultdict(set)
        for e, c in edge_color.items():
            color_classes[c].add(e)

        # For each color class, extract longest cycle(s)
        # Process color classes in order of size (largest first)
        sorted_colors = sorted(color_classes.keys(), key=lambda c: len(color_classes[c]), reverse=True)

        for c in sorted_colors:
            sub_edges = color_classes[c]
            if not sub_edges:
                continue

            # Extract cycles from this color class
            while sub_edges:
                if time.time() > time_limit:
                    break

                cycle = find_longest_cycle_in_subgraph(sub_edges)
                if cycle:
                    pieces.append(cycle)
                    # Remove cycle edges from sub_edges and remaining_edges
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = (u, v) if u < v else (v, u)
                        sub_edges.discard(e)
                        remaining_edges.discard(e)
                    u, v = cycle[-1], cycle[0]
                    e = (u, v) if u < v else (v, u)
                    sub_edges.discard(e)
                    remaining_edges.discard(e)
                else:
                    # No cycle in this color class; add each edge as a piece
                    for e in sub_edges:
                        pieces.append(list(e))
                        remaining_edges.discard(e)
                    sub_edges.clear()

        # Any remaining edges (shouldn't be any if coloring covered all)
        if remaining_edges:
            for e in remaining_edges:
                pieces.append(list(e))

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)

        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

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
                    u, v = piece[k], piece[k+1]
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
