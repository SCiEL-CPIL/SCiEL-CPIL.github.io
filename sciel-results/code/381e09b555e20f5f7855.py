"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        """Return the set of edges in the 2-core of the graph defined by edge_set."""
        if not edge_set:
            return set()
        
        # Build degree map and adjacency
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        # Queue vertices with degree < 2
        queue = deque([v for v in deg if deg[v] < 2])
        in_core = set(deg.keys())
        
        while queue:
            v = queue.popleft()
            if v not in in_core:
                continue
            in_core.discard(v)
            for nb in adj[v]:
                if nb in in_core:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        queue.append(nb)
                        
        # Reconstruct edge set from core
        core_edges = set()
        for u, v in edge_set:
            if u in in_core and v in in_core:
                core_edges.add(tuple(sorted((u, v))))
        return core_edges

    def decompose(edges, n, time_budget):
        """
        Decompose the edge set into cycles (>=3 distinct vertices) and single edges.
        Strategy:
          1. Identify the 2-core. Edges outside the 2-core are trees/forests and must be single pieces.
          2. Repeatedly find a simple cycle in the current core using randomized DFS.
          3. Remove the cycle and update the core.
          4. If no cycle exists, the remaining edges form a forest; each is a piece.
        We try many random restarts to find fewer pieces.
        """
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1

        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_cycle_dfs(adj, rem, start=None):
            """Find a simple cycle in the subgraph induced by rem. Returns list of vertices or None."""
            if start is None:
                # Pick a random vertex of degree >= 2
                candidates = [u for u in adj if len(adj[u]) >= 2 and
                              any((min(u, nb), max(u, nb)) in rem for nb in adj[u])]
                if not candidates:
                    return None
                start = random.choice(candidates)

            visited = {start}
            parent = {start: None}
            stack = [start]
            found_cycle = None

            while stack:
                u = stack[-1]
                nbrs = [nb for nb in adj[u] if (min(u, nb), max(u, nb)) in rem]
                # Shuffle for randomness
                random.shuffle(nbrs)
                advanced = False
                for nb in nbrs:
                    if nb == start and len(stack) >= 2:
                        # Cycle found
                        found_cycle = stack[:]
                        break
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = u
                        stack.append(nb)
                        advanced = True
                        break
                if found_cycle:
                    break
                if not advanced:
                    stack.pop()

            return found_cycle

        def greedy_decompose(rem_set):
            """Greedy: repeatedly extract a cycle; leftover edges are individual pieces."""
            rem = set(rem_set)
            pieces = []
            
            # First, separate the 2-core from the forest part
            core_edges = get_2_core(rem)
            forest_edges = rem - core_edges
            
            # All forest edges must be single pieces
            for e in forest_edges:
                pieces.append(list(e))
            
            # Now work on the core
            rem = core_edges
            if not rem:
                return pieces

            while rem:
                adj = build_adj(rem)
                cycle = None
                # Try a few random starts
                for _ in range(10):
                    verts = set()
                    for u, v in rem:
                        verts.add(u)
                        verts.add(v)
                    if not verts:
                        break
                    start = random.choice(list(verts))
                    cycle = find_cycle_dfs(adj, rem, start)
                    if cycle and len(cycle) >= 3:
                        break
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        rem.discard(e)
                    pieces.append(cycle[:])
                else:
                    # No cycle found; take one edge
                    e = tuple(next(iter(rem)))
                    rem.discard(e)
                    pieces.append(list(e))
            return pieces

        # Multiple restarts with random edge ordering
        max_restarts = 30
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            # Shuffle the order in which we might pick edges (for tie-breaking diversity)
            pieces = greedy_decompose(edge_set)
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best_pieces = [p[:] for p in pieces]

        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
