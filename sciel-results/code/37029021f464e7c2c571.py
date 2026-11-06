import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Greedy edge removal: prefer removing edges that are "bridges" or in low-degree areas
        # to keep the graph more connected for path extraction
        def greedy_remove():
            nonlocal remaining_edges
            if not remaining_edges:
                return []
            
            # Find all edges
            all_edges = list(remaining_edges)
            # Score each edge: prefer removing edges where both endpoints have high degree in remaining graph
            # This is a heuristic; let's try removing edges that are "central" first to break cycles?
            # Actually, for path cover, we want to keep the graph as path-like as possible.
            # A common heuristic is to remove edges to make the graph a collection of paths/cycles.
            
            # Let's try a simple greedy: remove edges one by one, preferring those that don't disconnect the graph?
            # That's expensive. 
            # Alternative: just remove edges in a way that maximizes the chance of finding long paths.
            # Let's stick with the beam search but improve the extraction.
            
            # For now, return an empty list to indicate no pre-processing
            return []

        remaining_edges = set(edges)
        paths = []

        def extract_path_beam(beam_width=10, max_depth=100):
            """Beam search to find a long simple path in remaining_edges."""
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            if not active_vertices:
                return None

            vertices_list = list(active_vertices)
            random.shuffle(vertices_list)

            best_path = None
            best_len = 0

            # Limit starts based on time
            max_starts = min(50, len(vertices_list))

            for start in vertices_list[:max_starts]:
                # Check if start has any remaining edge
                has_edge = False
                for n in adj[start]:
                    e = (min(start, n), max(start, n))
                    if e in remaining_edges:
                        has_edge = True
                        break
                if not has_edge:
                    continue

                # Beam search
                # Each beam entry: (path, visited_set, current_node)
                beam = [([start], {start}, start)]

                for depth in range(max_depth):
                    if time.time() > deadline - 2.0:
                        break
                    new_beam = []
                    for path, visited, node in beam:
                        # Find valid neighbors
                        valid_neighbors = []
                        for n in adj[node]:
                            if n in visited:
                                continue
                            e = (min(node, n), max(node, n))
                            if e in remaining_edges:
                                valid_neighbors.append(n)

                        if not valid_neighbors:
                            # Dead end; keep as candidate
                            if len(path) > best_len:
                                best_path = path[:]
                                best_len = len(path)
                            continue

                        # Score neighbors: prefer those with fewer options (Warnsdorff)
                        scored = []
                        for n in valid_neighbors:
                            count = 0
                            for m in adj[n]:
                                if m in visited:
                                    continue
                                e = (min(n, m), max(n, m))
                                if e in remaining_edges:
                                    count += 1
                            scored.append((count, n))
                        scored.sort()

                        # Take top beam_width neighbors for this beam entry
                        for _, n in scored[:beam_width]:
                            new_path = path + [n]
                            new_visited = visited | {n}
                            new_beam.append((new_path, new_visited, n))

                    if not new_beam:
                        break

                    # Keep only top beam_width entries by path length
                    new_beam.sort(key=lambda x: -len(x[0]))
                    beam = new_beam[:beam_width]

                    # Update best if any beam entry is longer
                    if beam:
                        longest = max(len(p) for p, v, n in beam)
                        if longest > best_len:
                            best_len = longest
                            best_path = beam[0][0][:]

                # After beam search, check final beam entries
                for path, visited, node in beam:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path[:]

            return best_path

        def extract_path_local(beam_width=10, max_depth=100):
            """Extract a path using local remaining edges context."""
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            if not active_vertices:
                return None

            vertices_list = list(active_vertices)
            random.shuffle(vertices_list)

            best_path = None
            best_len = 0
            max_starts = min(50, len(vertices_list))

            for start in vertices_list[:max_starts]:
                has_edge = False
                for n in adj[start]:
                    e = (min(start, n), max(start, n))
                    if e in remaining_edges:
                        has_edge = True
                        break
                if not has_edge:
                    continue

                beam = [([start], {start}, start)]

                for depth in range(max_depth):
                    if time.time() > deadline - 2.0:
                        break
                    new_beam = []
                    for path, visited, node in beam:
                        valid_neighbors = []
                        for n in adj[node]:
                            if n in visited:
                                continue
                            e = (min(node, n), max(node, n))
                            if e in remaining_edges:
                                valid_neighbors.append(n)

                        if not valid_neighbors:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_path = path[:]
                            continue

                        scored = []
                        for n in valid_neighbors:
                            count = 0
                            for m in adj[n]:
                                if m in visited:
                                    continue
                                e2 = (min(n, m), max(n, m))
                                if e2 in remaining_edges:
                                    count += 1
                            scored.append((count, n))
                        scored.sort()

                        for _, n in scored[:beam_width]:
                            new_path = path + [n]
                            new_visited = visited | {n}
                            new_beam.append((new_path, new_visited, n))

                    if not new_beam:
                        break

                    new_beam.sort(key=lambda x: -len(x[0]))
                    beam = new_beam[:beam_width]

                    if beam:
                        longest = max(len(p) for p, v, n in beam)
                        if longest > best_len:
                            best_len = longest
                            best_path = beam[0][0][:]

                for path, visited, node in beam:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path[:]

            return best_path

        # Phase 1: Greedy extraction with beam search
        while remaining_edges:
            if time.time() > deadline - 2.0:
                for e in remaining_edges:
                    paths.append(list(e))
                break

            best_path = extract_path_beam(beam_width=10, max_depth=100)

            if best_path and len(best_path) > 1:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i + 1]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)
            else:
                if remaining_edges:
                    e = next(iter(remaining_edges))
                    paths.append(list(e))
                    remaining_edges.discard(e)

        # Phase 2: Polish — try to replace short paths with longer ones
        # Sort paths by length, try to re-extract the shortest ones
        if time.time() < deadline - 5.0:
            # Identify shortest paths
            path_indices = list(range(len(paths)))
            path_indices.sort(key=lambda i: len(paths[i]))

            # Try to improve the shortest paths
            for idx in path_indices[:5]:  # Only try to improve up to 5 shortest
                if time.time() > deadline - 3.0:
                    break
                old_path = paths[idx]
                if len(old_path) <= 1:
                    continue

                # Remove old path's edges from a copy
                old_edges = set()
                for i in range(len(old_path) - 1):
                    u, v = old_path[i], old_path[i + 1]
                    e = (min(u, v), max(u, v))
                    old_edges.add(e)

                # Temporarily add these edges back to remaining
                for e in old_edges:
                    remaining_edges.add(e)

                # Try to extract a longer path
                new_path = extract_path_beam(beam_width=10, max_depth=100)

                if new_path and len(new_path) > len(old_path):
                    # Accept the new path
                    paths[idx] = new_path
                    for i in range(len(new_path) - 1):
                        u, v = new_path[i], new_path[i + 1]
                        e = (min(u, v), max(u, v))
                        remaining_edges.discard(e)
                else:
                    # Revert: keep old path
                    for e in old_edges:
                        remaining_edges.discard(e)

        # Phase 3: Local improvement — try to extend paths by finding unused edges
        if time.time() < deadline - 5.0:
            # For each path, try to extend it at either end using remaining edges
            for _iteration in range(3):  # Multiple passes
                if time.time() > deadline - 3.0:
                    break
                
                improved = False
                for idx in range(len(paths)):
                    if time.time() > deadline - 3.0:
                        break
                    
                    path = paths[idx]
                    if len(path) < 2:
                        continue
                    
                    # Try to extend at the start
                    start_node = path[0]
                    extended = False
                    for n in adj[start_node]:
                        e = (min(start_node, n), max(start_node, n))
                        if e in remaining_edges:
                            # Check if n is not already in the path
                            if n not in path:
                                paths[idx] = [n] + path
                                remaining_edges.discard(e)
                                extended = True
                                improved = True
                                break
                    
                    # Try to extend at the end
                    if not extended:
                        end_node = path[-1]
                        for n in adj[end_node]:
                            e = (min(end_node, n), max(end_node, n))
                            if e in remaining_edges:
                                # Check if n is not already in the path
                                if n not in path:
                                    paths[idx] = path + [n]
                                    remaining_edges.discard(e)
                                    extended = True
                                    improved = True
                                    break
                
                if not improved:
                    break

        results.append(paths)

    return results
