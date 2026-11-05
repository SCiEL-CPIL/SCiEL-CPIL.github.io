"""Edge decomposition into cycles and edges: ILP over a pool of cycles."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_in_graph(n, edge_set, max_cycles=5000, max_len=20):
        """Find many simple cycles using randomized DFS."""
        adj = defaultdict(list)
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        for lst in adj.values():
            lst.sort()

        cycles = []
        seen = set()
        rng = random.Random(42)

        # Try starting from each vertex
        for start in range(n):
            if not adj[start]:
                continue
            # Randomized DFS to find cycles
            stack = [(start, [start], {start})]
            found = 0
            while stack and found < max_cycles // max(1, len(adj)):
                if time.time() > deadline - 5:
                    break
                node, path, visited = stack.pop()
                found += 1
                if len(path) >= 3 and node == start:
                    # Found a cycle
                    cycle = tuple(sorted(path[:-1]))
                    if cycle not in seen and len(cycle) >= 3:
                        seen.add(cycle)
                        cycles.append(path[:-1])
                if len(path) < max_len:
                    for nb in adj[node]:
                        if nb not in visited:
                            stack.append((nb, path + [nb], visited | {nb}))
        return cycles

    def decompose_graph(n, edge_list):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        edge_set = set(tuple(sorted(e)) for e in edge_list)
        m = len(edge_set)

        if m == 0:
            return []

        # Generate candidate cycles
        cycles = find_cycles_in_graph(n, edge_set, max_cycles=3000, max_len=15)

        # Convert cycles to edge sets
        cycle_edges = []
        for cyc in cycles:
            edges_in_cycle = set()
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                e = tuple(sorted((u, v)))
                if e in edge_set:
                    edges_in_cycle.add(e)
                else:
                    edges_in_cycle = None
                    break
            if edges_in_cycle and len(edges_in_cycle) == len(cyc):
                cycle_edges.append(frozenset(edges_in_cycle))

        # Remove duplicate cycles
        cycle_edges = list(set(cycle_edges))

        # If no cycles found, return all edges as single pieces
        if not cycle_edges:
            return [list(e) for e in edge_list]

        # Greedy approach: repeatedly pick the cycle that covers the most remaining edges
        remaining = set(edge_set)
        pieces = []
        used_cycle_indices = set()

        # Sort cycles by length descending (longer cycles remove more edges per piece)
        cycle_order = sorted(range(len(cycle_edges)), key=lambda i: -len(cycle_edges[i]))

        # Greedy: pick cycles that cover many remaining edges
        while remaining:
            best_idx = -1
            best_covered = 0
            for idx in cycle_order:
                if idx in used_cycle_indices:
                    continue
                covered = len(cycle_edges[idx] & remaining)
                if covered > best_covered:
                    best_covered = covered
                    best_idx = idx
                if covered == len(cycle_edges[idx]):
                    # Can't do better than full coverage
                    break

            if best_idx == -1 or best_covered == 0:
                break

            used_cycle_indices.add(best_idx)
            # Reconstruct the cycle path for output
            cyc_edges = cycle_edges[best_idx]
            # Find the original cycle path
            for cyc in cycles:
                ce = set()
                valid = True
                for i in range(len(cyc)):
                    u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set:
                        valid = False
                        break
                    ce.add(e)
                if valid and frozenset(ce) == cyc_edges:
                    pieces.append(cyc)
                    break
            else:
                # Fallback: just use one edge
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
                continue

            remaining -= cycle_edges[best_idx]

        # Add remaining edges as single pieces
        for e in remaining:
            pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 10:
            # Not enough time left; fall back to trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose_graph(n, edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
