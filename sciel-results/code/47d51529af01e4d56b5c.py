"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        n = max(max(u, v) for u, v in edge_set) + 1 if edge_set else 0

        # Step 1: Find a path/cycle cover using greedy + local search
        # We'll work with a permutation-like structure: each vertex has at most 2 incident "used" edges
        # Build initial decomposition by finding edge-disjoint cycles and paths

        # First, get all odd-degree vertices
        deg = defaultdict(int)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        odd_verts = [v for v in deg if deg[v] % 2 == 1]

        # Find a minimum-weight matching on odd vertices to pair them up
        # This tells us where to "break" trails to form paths
        # For now, use a simple greedy matching
        matched = set()
        match_pairs = []
        for i, u in enumerate(odd_verts):
            if u in matched:
                continue
            # Find closest unmatched odd vertex
            best_v = None
            best_dist = float('inf')
            for v in odd_verts[i+1:]:
                if v in matched:
                    continue
                # Use BFS distance in the graph
                dist = _bfs_distance(u, v, edge_set, n)
                if dist < best_dist:
                    best_dist = dist
                    best_v = v
            if best_v is not None:
                matched.add(u)
                matched.add(best_v)
                match_pairs.append((u, best_v))

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        # Step 2: Find edge-disjoint cycles greedily, then improve
        remaining = set(edge_set)
        pieces = []

        def build_adj(rem):
            a = defaultdict(set)
            for u, v in rem:
                a[u].add(v)
                a[v].add(u)
            return a

        adj_rem = build_adj(remaining)

        # Find cycles using randomized DFS with restarts
        def find_cycle(start, max_len=80, attempts=50):
            if len(remaining) < 3:
                return None
            best = None
            for _ in range(attempts):
                path = [start]
                visited = {start}
                cur = start
                ok = False
                for _ in range(max_len):
                    nbrs = [nb for nb in adj_rem.get(cur, []) 
                            if (min(cur, nb), max(cur, nb)) in remaining]
                    if not nbrs:
                        break
                    unvis = [nb for nb in nbrs if nb not in visited]
                    if start in nbrs and len(path) >= 3:
                        path.append(start)
                        ok = True
                        break
                    if unvis:
                        nxt = random.choice(unvis)
                    else:
                        if start in nbrs and len(path) >= 3:
                            path.append(start)
                            ok = True
                            break
                        break
                    path.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                if ok and (best is None or len(path) > len(best)):
                    best = path[:]
            return best

        # Greedily extract cycles
        while remaining:
            # Find candidate starts: vertices with degree >= 2 in remaining
            cand = [v for v in adj_rem if len(adj_rem[v]) >= 2]
            if not cand:
                break
            best_cycle = None
            # Try a few random starts
            random.shuffle(cand)
            for start in cand[:min(10, len(cand))]:
                c = find_cycle(start)
                if c and (best_cycle is None or len(c) > len(best_cycle)):
                    best_cycle = c
            if best_cycle is None:
                break
            # Extract
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                remaining.discard((min(u, v), max(u, v)))
            pieces.append(best_cycle[:-1][:])
            adj_rem = build_adj(remaining)

        # Remaining edges become single-edge pieces
        for e in sorted(remaining):
            pieces.append([e[0], e[1]])

        # Step 3: Local search improvement - try to merge single edges into cycles
        # or replace short cycles with longer ones
        # We'll do iterative improvement: try to find a cycle in the "residual" 
        # structure that includes some single edges

        # Build a mapping: which edges are in which piece
        edge_to_piece = {}
        for i, piece in enumerate(pieces):
            if len(piece) >= 3:
                for j in range(len(piece)):
                    u, v = piece[j], piece[(j + 1) % len(piece)]
                    edge_to_piece[(min(u, v), max(u, v))] = i
            else:
                u, v = piece[0], piece[1]
                edge_to_piece[(min(u, v), max(u, v))] = i

        # Try to find cycles that use edges from multiple pieces
        # This is complex; instead, do a simpler improvement:
        # For each pair of adjacent single edges (sharing a vertex), check if they
        # can be part of a longer cycle with existing cycle edges

        # Alternative: use the Euler trail approach
        # Add "virtual" edges between matched odd vertices to make all degrees even
        # Then find Eulerian circuits, which decompose into cycles

        # Let's try a different approach: iterative cycle extraction with
        # simulated annealing-like acceptance

        # Actually, let's try: find the maximum set of edge-disjoint cycles
        # using a greedy + local search on the cycle space

        # For time efficiency, do a few rounds of "cycle improvement"
        # where we try to replace a short cycle + adjacent singles with a longer cycle

        improved = True
        max_rounds = 20
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try to find a longer cycle by combining edges from multiple pieces
            # Build a "combined" graph from all edges
            all_edges = set()
            for piece in pieces:
                if len(piece) >= 3:
                    for j in range(len(piece)):
                        u, v = piece[j], piece[(j + 1) % len(piece)]
                        all_edges.add((min(u, v), max(u, v)))
                else:
                    all_edges.add((piece[0], piece[1]))

            # Try to find a cycle that uses edges from at least 2 different pieces
            # and is longer than the sum of the pieces it replaces
            # This is NP-hard in general, so use heuristic

            # Find a vertex with high degree in the combined graph
            comb_adj = defaultdict(set)
            for u, v in all_edges:
                comb_adj[u].add(v)
                comb_adj[v].add(u)

            # Try random walks to find cycles that span multiple pieces
            for attempt in range(30):
                if not all_edges:
                    break
                start = random.choice(list(comb_adj.keys())) if comb_adj else None
                if start is None:
                    break
                # Random walk trying to close a cycle
                path = [start]
                visited = {start}
                cur = start
                success = False
                for _ in range(60):
                    nbrs = [nb for nb in comb_adj.get(cur, [])
                            if (min(cur, nb), max(cur, nb)) in all_edges]
                    if not nbrs:
                        break
                    unvis = [nb for nb in nbrs if nb not in visited]
                    if start in nbrs and len(path) >= 3:
                        path.append(start)
                        success = True
                        break
                    if unvis:
                        nxt = random.choice(unvis)
                    else:
                        if start in nbrs and len(path) >= 3:
                            path.append(start)
                            success = True
                            break
                        break
                    path.append(nxt)
                    visited.add(nxt)
                    cur = nxt

                if not success or len(path) < 4:
                    continue

                # Check which pieces this cycle uses edges from
                cycle_edges = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    cycle_edges.append((min(u, v), max(u, v)))

                # Find which pieces contain these edges
                pieces_used = set()
                for e in cycle_edges:
                    if e in edge_to_piece:
                        pieces_used.add(edge_to_piece[e])

                if len(pieces_used) < 2:
                    continue

                # This cycle uses edges from multiple pieces
                # Replace those pieces with this cycle + leftover edges
                old_piece_count = len(pieces_used)
                # New pieces: the cycle + any leftover edges from the used pieces
                new_pieces = [path[:-1][:]]
                for pi in pieces_used:
                    old_piece = pieces[pi]
                    if len(old_piece) >= 3:
                        # Get edges not in the cycle
                        old_edges = set()
                        for j in range(len(old_piece)):
                            u, v = old_piece[j], old_piece[(j + 1) % len(old_piece)]
                            old_edges.add((min(u, v), max(u, v)))
                        leftover = old_edges - set(cycle_edges)
                        for le in leftover:
                            new_pieces.append([le[0], le[1]])
                    else:
                        e = (min(old_piece[0], old_piece[1]), max(old_piece[0], old_piece[1]))
                        if e not in set(cycle_edges):
                            new_pieces.append(old_piece[:])

                # Check if this is an improvement
                if len(new_pieces) < old_piece_count:
                    # Apply the change
                    # Remove old pieces
                    for pi in sorted(pieces_used, reverse=True):
                        pieces.pop(pi)
                    # Add new pieces
                    pieces.extend(new_pieces)
                    # Rebuild edge_to_piece
                    edge_to_piece = {}
                    for i, piece in enumerate(pieces):
                        if len(piece) >= 3:
                            for j in range(len(piece)):
                                u, v = piece[j], piece[(j + 1) % len(piece)]
                                edge_to_piece[(min(u, v), max(u, v))] = i
                        else:
                            u, v = piece[0], piece[1]
                            edge_to_piece[(min(u, v), max(u, v))] = i
                    improved = True
                    break

            if not improved:
                break

        return pieces

    def _bfs_distance(start, target, edge_set, n):
        """BFS distance in the graph."""
        if start == target:
            return 0
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in adj.get(node, []):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return float('inf')

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
