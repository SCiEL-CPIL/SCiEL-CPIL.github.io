"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # 5s safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hungarian(cost_matrix):
        """
        Solve assignment problem (minimization) using Hungarian algorithm.
        Returns (cost, assignment) where assignment[i] = j.
        Complexity: O(n^3).
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = -1
                
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost_matrix[i0-1][j-1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                
                j0 = j1
                if p[j0] == 0:
                    break
            
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        
        assignment = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment[p[j]-1] = j - 1
        
        cost = 0
        for i in range(n):
            cost += cost_matrix[i][assignment[i]]
            
        return cost, assignment

    def solve_graph(edges):
        """
        Decompose graph into minimum number of paths using matching on line graph.
        """
        if not edges:
            return []

        edge_indices = list(range(len(edges)))
        edge_map = {}
        for i, (u, v) in enumerate(edges):
            key = (u, v) if u < v else (v, u)
            edge_map[key] = i

        # Build line graph adjacency
        # L[i] = list of (j, shared_vertex)
        # We only care about compatibility for matching
        adj = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                if u == x or u == y:
                    adj[i].append((j, u))
                elif v == x or v == y:
                    adj[i].append((j, v))

        # Hungarian algorithm setup
        # We want to maximize weight (path length), so we minimize negative weight.
        # But we also need to handle the constraint that a vertex cannot be matched
        # to more than 2 edges in the line graph (degree constraint in original graph).
        # Standard matching doesn't enforce degree constraints directly.
        # However, if we just maximize sum of weights where weight=1 for compatible edges,
        # the resulting structure is a set of paths and cycles in the line graph.
        # Since the original graph is a simple graph, cycles in the line graph correspond
        # to cycles in the original graph (if length >= 3) or multi-edges (not allowed here).
        # Wait, a cycle in the line graph of length k corresponds to a cycle of length k in G.
        # Gallai's conjecture is about paths. A cycle can be decomposed into 1 path? No,
        # a cycle is not a path. But we can break it?
        # Actually, the problem asks for a path cover.
        # If we find a maximum matching in the line graph, we get a subgraph where each edge
        # (original) is matched to at most one other edge at each end.
        # This forms a collection of paths and cycles in the original graph.
        # Any cycle can be "broken" into a path by removing one edge? No, that would leave
        # an edge uncovered or change the count.
        # Actually, if we have a cycle, it's a valid "trail" but not a "simple path".
        # The problem requires "simple paths".
        # However, in a simple graph, a cycle of length >= 3 is not a simple path.
        # But wait, if we have a cycle, we can just treat it as a path by... wait.
        # A path cover usually allows cycles to be converted?
        # No, the definition of path cover is a set of vertex-disjoint paths that cover all vertices?
        # No, "edge-disjoint and cover all edges".
        # "Paths must be simple".
        # So we cannot have cycles.
        
        # How to ensure no cycles?
        # If the matching creates a cycle, we can break it by unmatching one edge in the cycle.
        # This reduces the number of matches by 1, increasing the number of paths by 1.
        # But maybe we can find a better matching that avoids cycles?
        # For now, let's assume the matching might produce cycles, and we'll break them.
        
        # To maximize path length, we assign weight 1 to compatible edges.
        # We want to maximize the number of matches.
        # Cost matrix: 0 if compatible, large value if not.
        # Since Hungarian solves minimization, we use cost = -1 for compatible, 0 for not?
        # No, we need to prevent invalid matches.
        # Let's use cost = 0 for compatible, 1 for not compatible.
        # Then we minimize the number of "bad" matches? No, we want to maximize good matches.
        # So cost = 0 for good, 1 for bad?
        # If we minimize sum of costs, we prefer 0s.
        # But we must assign every row to a column.
        # So every edge i is assigned to some edge j.
        # If j is compatible, cost 0. If not, cost 1.
        # We want to minimize total cost, i.e., maximize compatible matches.
        
        n_edges = len(edges)
        cost_matrix = [[1] * n_edges for _ in range(n_edges)]
        
        for i in range(n_edges):
            for j in range(n_edges):
                if i == j:
                    cost_matrix[i][j] = 1000  # Self match is bad
                else:
                    # Check if i and j are compatible
                    # They are compatible if they share a vertex.
                    # In the line graph, an edge exists if they share a vertex.
                    # We built adj[i] as list of (j, v).
                    # So if j is in adj[i], they are compatible.
                    # But we need to be careful: adj[i] contains all j sharing a vertex with i.
                    # So we can just check if j is in the set of neighbors.
                    pass
        
        # Precompute neighbor sets for fast lookup
        neighbors = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            for j, _ in adj[i]:
                neighbors[i].add(j)
        
        for i in range(n_edges):
            for j in range(n_edges):
                if i == j:
                    cost_matrix[i][j] = 1000
                elif j in neighbors[i]:
                    cost_matrix[i][j] = 0
                else:
                    cost_matrix[i][j] = 1

        # Run Hungarian
        min_cost, assignment = hungarian(cost_matrix)
        
        # Reconstruct paths
        # assignment[i] = j means edge i is followed by edge j in a path.
        # This defines a functional graph. Since it's a permutation (bijection),
        # it consists of disjoint cycles.
        # Each cycle in the permutation corresponds to a cycle in the line graph.
        # We need to convert these to paths.
        
        visited = [False] * n_edges
        paths = []
        
        for i in range(n_edges):
            if not visited[i]:
                # Trace the cycle/path
                curr = i
                path_edges = []
                while not visited[curr]:
                    visited[curr] = True
                    path_edges.append(curr)
                    curr = assignment[curr]
                
                # We have a sequence of edge indices.
                # Check if it's a cycle or a path.
                # Since assignment is a permutation, it's always a cycle in the permutation sense.
                # But does it form a valid path in the graph?
                # The sequence e1, e2, ..., ek is valid if e1 shares a vertex with e2, etc.
                # And ek shares a vertex with e1 (closing the cycle).
                # If it's a cycle in the graph, we need to break it into a path.
                # How? Remove one edge from the cycle? No, that would leave it uncovered.
                # Wait, if we have a cycle of edges e1-e2-e3-e1, we can form a path e1-e2-e3?
                # But then e3 is not connected to e1 in the path.
                # The path is just a sequence of edges.
                # If the edges form a cycle in the graph, we can list them in order.
                # The "path" is the sequence of vertices.
                # Example: Triangle 1-2-3-1. Edges (1,2), (2,3), (3,1).
                # Matching: (1,2)->(2,3), (2,3)->(3,1), (3,1)->(1,2).
                # Sequence: 1, 2, 3.
                # Vertices: 1-2-3-1. This is a cycle, not a simple path.
                # The problem requires simple paths.
                # So we must break the cycle.
                # If we break it, we get a path 1-2-3. But edge (3,1) is not in the path?
                # No, the edge (3,1) is in the sequence.
                # The path is defined by the vertices.
                # If we take the sequence of edges, we can construct the vertex sequence.
                # If the vertex sequence has distinct vertices, it's a simple path.
                # If it repeats the start vertex at the end, it's a cycle.
                # To make it a simple path, we can just stop before repeating the start?
                # No, that would drop an edge.
                # Actually, if we have a cycle, we can't represent it as a single simple path
                # that covers all its edges.
                # Wait, a cycle of length k can be decomposed into 1 path? No.
                # It can be decomposed into k paths of length 1? No.
                # It can be decomposed into 1 path of length k-1 and 1 path of length 1?
                # No, edge-disjoint.
                # A cycle of length k requires at least 1 path? No, a cycle is not a path.
                # So we must break the cycle into paths.
                # The minimum number of paths to cover a cycle of length k is 1? No.
                # It is 1 if we allow cycles as paths? No, "simple paths".
                # So a cycle of length k must be broken.
                # If we break one edge, we get a path of length k-1.
                # But the broken edge is still there.
                # So we have a path of length k-1 and a path of length 1.
                # Total 2 paths.
                # Can we do better?
                # No, because the cycle is connected.
                # So each cycle in the matching corresponds to 2 paths?
                # Or 1 path if we consider the cycle as a path? No.
                
                # Let's check the vertex sequence.
                # If the vertex sequence is simple (no repeats), it's a path.
                # If it repeats, it's a cycle.
                
                # Construct vertex sequence
                verts = []
                for idx, e_idx in enumerate(path_edges):
                    u, v = edges[e_idx]
                    if idx == 0:
                        verts.append(u)
                        verts.append(v)
                    else:
                        # Check connection
                        prev_u, prev_v = edges[path_edges[idx-1]]
                        # Determine orientation
                        if prev_v == u:
                            verts.append(v)
                        elif prev_v == v:
                            verts.append(u)
                        else:
                            # Disconnected? Should not happen if matching is valid.
                            # But if it does, we have a problem.
                            pass
                
                # Check if simple
                if len(set(verts)) == len(verts):
                    paths.append(verts)
                else:
                    # It's a cycle. Break it.
                    # Remove the last edge to make it a path?
                    # Then we have a path and a single edge.
                    # Path: verts[0:-1]
                    # Single edge: last edge
                    if len(path_edges) > 1:
                        # Path part
                        p_verts = verts[:-1]
                        if len(p_verts) >= 2:
                            paths.append(p_verts)
                        # Single edge part
                        last_e = edges[path_edges[-1]]
                        paths.append([last_e[0], last_e[1]])
                    else:
                        paths.append(verts)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 5:
            # Fallback: trivial cover
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(*spec)
        try:
            paths = solve_graph(edges)
            
            # Validate
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u < v else (v, u))
            
            covered = set()
            valid = True
            for p in paths:
                if len(p) < 2:
                    valid = False
                    break
                if len(set(p)) != len(p):
                    valid = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u < v else (v, u)
                    if key not in edge_set:
                        valid = False
                        break
                    if key in covered:
                        valid = False
                        break
                    covered.add(key)
                if not valid:
                    break
            
            if not valid or covered != edge_set:
                # Fallback: trivial cover
                paths = [[u, v] for u, v in edges]
                
        except Exception:
            paths = [[u, v] for u, v in edges]

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
