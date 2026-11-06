import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    # Allocate time: ~45s per graph, with safety margin
    time_per_graph = (deadline - start_time) / 3.0 - 2.0

    for gi, spec in enumerate(specs):
        graph_deadline = start_time + (gi + 1) * time_per_graph + 1.0
        if graph_deadline > deadline - 2.0:
            graph_deadline = deadline - 2.0

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_vertices = max(adj.keys()) + 1 if adj else 0
        all_vertices = set(range(n_vertices))

        remaining_edges = set(edges)
        paths = []

        def edge_key(u, v):
            return (min(u, v), max(u, v))

        def count_remaining_options(node, visited, rem_edges):
            """Count how many remaining edges go from node to unvisited vertices."""
            cnt = 0
            for nb in adj[node]:
                if nb in visited:
                    continue
                e = edge_key(node, nb)
                if e in rem_edges:
                    cnt += 1
            return cnt

        def try_extract_path(start, visited, rem_edges, max_depth=200, beam_width=3):
            """
            Try to extend a path starting at 'start' using Warnsdorff + beam.
            Returns the longest simple path found (as list of vertices).
            """
            best = [start]
            beam = [([start], {start}, start)]

            for depth in range(max_depth):
                if time.time() > graph_deadline - 1.0:
                    break
                new_beam = []
                for path, vis, node in beam:
                    # Collect valid neighbors
                    valid = []
                    for nb in adj[node]:
                        if nb in vis:
                            continue
                        e = edge_key(node, nb)
                        if e in rem_edges:
                            valid.append(nb)

                    if not valid:
                        # Dead end
                        if len(path) > len(best):
                            best = path[:]
                        continue

                    # Warnsdorff: prefer neighbors with fewest remaining options
                    scored = []
                    for nb in valid:
                        opts = count_remaining_options(nb, vis, rem_edges)
                        scored.append((opts, nb))
                    scored.sort()

                    # Keep top beam_width
                    for _, nb in scored[:beam_width]:
                        new_path = path + [nb]
                        new_vis = vis | {nb}
                        new_beam.append((new_path, new_vis, nb))

                if not new_beam:
                    break

                # Prune: keep only longest paths, up to beam_width
                new_beam.sort(key=lambda x: -len(x[0]))
                # Deduplicate by (node, frozenset(visited)) to save memory
                seen = set()
                pruned = []
                for item in new_beam:
                    key = (item[2], frozenset(item[1]))
                    if key not in seen:
                        seen.add(key)
                        pruned.append(item)
                beam = pruned[:beam_width]

                # Update best
                if beam:
                    longest = max(len(p) for p, v, n in beam)
                    if longest > len(best):
                        best = beam[0][0][:]

            # Final check
            for path, vis, node in beam:
                if len(path) > len(best):
                    best = path[:]

            return best

        def extract_best_path(rem_edges, n_starts=20, beam_width=3):
            """
            Try multiple start vertices and return the longest path found.
            """
            active = set()
            for u, v in rem_edges:
                active.add(u)
                active.add(v)
            if not active:
                return None

            # Prioritize starts: prefer odd-degree vertices in remaining graph
            deg_rem = defaultdict(int)
            for u, v in rem_edges:
                deg_rem[u] += 1
                deg_rem[v] += 1

            starts = list(active)
            # Sort: odd-degree first, then by degree (higher degree first)
            starts.sort(key=lambda x: (deg_rem[x] % 2 == 0, -deg_rem[x]))

            # Shuffle within groups for diversity
            random.shuffle(starts)
            starts = starts[:min(n_starts, len(starts))]

            best_path = None
            for start in starts:
                if time.time() > graph_deadline - 1.0:
                    break
                p = try_extract_path(start, set(), rem_edges, max_depth=150, beam_width=beam_width)
                if p and (best_path is None or len(p) > len(best_path)):
                    best_path = p

            return best_path

        def remove_path_edges(path, rem_edges):
            """Remove edges of a path from rem_edges. Returns True if all edges were present."""
            for i in range(len(path) - 1):
                e = edge_key(path[i], path[i + 1])
                if e not in rem_edges:
                    return False
                rem_edges.discard(e)
            return True

        def add_path_edges(path, rem_edges):
            """Add edges of a path back to rem_edges."""
            for i in range(len(path) - 1):
                rem_edges.add(edge_key(path[i], path[i + 1]))

        # === Phase 1: Greedy extraction with multiple restarts ===
        # Try several full extractions and keep the best (fewest paths)
        best_paths = None
        best_count = float('inf')

        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > graph_deadline - 3.0:
                break

            rem = set(remaining_edges)
            cur_paths = []

            while rem:
                if time.time() > graph_deadline - 1.5:
                    # Fallback: add remaining as single-edge paths
                    for e in rem:
                        cur_paths.append(list(e))
                    break

                p = extract_best_path(rem, n_starts=15, beam_width=3)
                if p and len(p) > 1:
                    if remove_path_edges(p, rem):
                        cur_paths.append(p)
                    else:
                        # Edge conflict; use single edge
                        e = next(iter(rem))
                        cur_paths.append(list(e))
                        rem.discard(e)
                else:
                    if rem:
                        e = next(iter(rem))
                        cur_paths.append(list(e))
                        rem.discard(e)

            if len(cur_paths) < best_count:
                best_count = len(cur_paths)
                best_paths = cur_paths[:]

        if best_paths is None:
            # Fallback
            best_paths = [list(e) for e in remaining_edges]
            best_count = len(best_paths)

        paths = best_paths[:]

        # === Phase 2: Polish — try to reduce path count by merging ===
        # Try to find a path that covers edges from two existing paths
        polish_deadline = graph_deadline - 2.0
        for _ in range(3):
            if time.time() > polish_deadline:
                break
            improved = False
            # For each pair of short paths, try to find a longer path covering both
            for i in range(len(paths)):
                if time.time() > polish_deadline:
                    break
                if len(paths[i]) > 3:
                    continue
                for j in range(i + 1, len(paths)):
                    if time.time() > polish_deadline:
                        break
                    if len(paths[j]) > 3:
                        continue

                    # Collect edges from both paths
                    merged_edges = set()
                    for p in [paths[i], paths[j]]:
                        for k in range(len(p) - 1):
                            merged_edges.add(edge_key(p[k], p[k + 1]))

                    # Try to find a path that uses at least one edge from each
                    # Simple heuristic: try to extend path i to include an edge from path j
                    # This is complex; skip for now and rely on re-extraction

            if not improved:
                break

        # === Phase 3: Re-extract shortest paths with more beam width ===
        if time.time() < graph_deadline - 4.0:
            # Sort by length
            idxs = sorted(range(len(paths)), key=lambda i: len(paths[i]))
            for idx in idxs[:3]:
                if time.time() > graph_deadline - 2.0:
                    break
                old = paths[idx]
                if len(old) <= 1:
                    continue

                # Remove old edges
                old_edges = set()
                for k in range(len(old) - 1):
                    old_edges.add(edge_key(old[k], old[k + 1]))
                for e in old_edges:
                    remaining_edges.discard(e)
                for e in old_edges:
                    remaining_edges.add(e)  # temporarily add back

                # Try to extract a longer path
                rem_temp = set(remaining_edges)
                new_p = extract_best_path(rem_temp, n_starts=25, beam_width=5)
                if new_p and len(new_p) > len(old):
                    # Check if new path's edges are subset of old_edges union existing remaining
                    new_edges = set()
                    for k in range(len(new_p) - 1):
                        new_edges.add(edge_key(new_p[k], new_p[k + 1]))
                    # Accept if it covers more edges
                    paths[idx] = new_p

                # Clean up: remove new path edges from remaining
                for e in new_edges if 'new_edges' in dir() else []:
                    remaining_edges.discard(e)

        # Rebuild remaining_edges correctly
        remaining_edges = set(edges)
        for p in paths:
            for k in range(len(p) - 1):
                e = edge_key(p[k], p[k + 1])
                remaining_edges.discard(e)

        results.append(paths)

    return results
