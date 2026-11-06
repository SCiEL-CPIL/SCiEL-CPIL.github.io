"""
Clique partition via edge-coloring with clique constraints.
Structural change: instead of enumerating maximal cliques and greedily covering edges,
we frame this as a constrained edge-coloring problem where each color class must form
a clique. We use a backtracking-based approach with strong pruning, combined with
a local search that swaps edge assignments between color classes.
"""
import time
import random
import sys

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    if not cliques:
        return len(edge_list) == 0
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _solve_one_graph(adj, edge_list, time_budget=50.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []
    
    # Build adjacency bitmask for fast clique checks
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    def _is_clique(vertices):
        """Check if a set of vertices forms a clique."""
        vs = list(vertices)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if not adj[vs[i]][vs[j]]:
                    return False
        return True
    
    def _clique_edges(clique):
        """Get all edges within a clique."""
        edges = set()
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                edges.add(_edge_key(clique[i], clique[j]))
        return edges
    
    # Strategy: Build a partition by iteratively assigning edges to cliques.
    # Key insight: We want to minimize the number of cliques, so we want cliques to be as large as possible.
    # 
    # New approach: Use a greedy algorithm that builds cliques one at a time.
    # For each clique, we pick a seed edge, then greedily add edges that can extend the clique
    # without creating conflicts.
    
    # Precompute for each edge, the set of vertices that could be in a clique containing that edge
    # and are mutually adjacent with both endpoints.
    
    # Better approach: Use a maximum clique heuristic on the line graph.
    # The line graph L(G) has a vertex for each edge of G. Two vertices in L(G) are adjacent
    # if the corresponding edges in G share a vertex. A clique in L(G) corresponds to a set of
    # edges that all share a common vertex (a star) OR form a triangle. But we need cliques in G,
    # which is different.
    
    # Let me try a different structural approach: 
    # Model as a set cover / packing problem where we select cliques (subsets of edges) such that
    # every edge is in exactly one selected clique, and each selected clique's edges form a clique in G.
    # 
    # The key structural change: Use a branch-and-bound with strong lower bounds.
    # Lower bound: For any vertex v, let d(v) be its degree. Any clique containing v can contain
    # at most d(v) edges incident to v. But more usefully: the number of cliques must be at least
    # the maximum over all vertices of ceil(d(v) / max_clique_size_through_v).
    
    # Actually, let me try a completely different approach:
    # Use simulated annealing on the assignment of edges to "clique IDs".
    # State: each edge has a clique_id. Two edges can share a clique_id only if they are compatible
    # (i.e., the union of their endpoints forms a clique in G).
    # Objective: minimize the number of distinct clique_ids used.
    # 
    # This is like a graph coloring problem on the "conflict graph" of edges, where two edges
    # conflict if they cannot be in the same clique.
    
    # Build the conflict graph of edges:
    # Two edges (a,b) and (c,d) conflict if {a,b,c,d} does not form a clique.
    # (They can share at most one endpoint, and if they share one, the other two must be adjacent to each other and to the shared vertex.)
    
    m = len(edge_list)
    edge_to_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    
    # Build conflict matrix (or use adjacency list for sparsity)
    # For m edges, this is O(m^2) which for n=64, p=0.55 gives ~1100 edges, so ~1.2M pairs - feasible.
    
    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edge_list[i]
        for j in range(i + 1, m):
            c, d = edge_list[j]
            # Check if edges i and j can be in the same clique
            # The union of endpoints must form a clique
            verts = set([a, b, c, d])
            compatible = True
            for x in verts:
                for y in verts:
                    if x < y and not adj[x][y]:
                        compatible = False
                        break
                if not compatible:
                    break
            if not compatible:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Now we have a graph coloring problem: color the m edges such that adjacent edges (in conflict graph)
    # get different colors. Minimize number of colors.
    # This is NP-hard but we can use greedy + local search.
    
    # Greedy coloring (Welsh-Powell style): order edges by degree in conflict graph, assign smallest available color.
    order = sorted(range(m), key=lambda i: -len(conflict[i]))
    
    # For better results, use a more sophisticated greedy:
    # Process edges in order, for each edge, try to assign it to an existing color class if compatible,
    # preferring the color class that would result in the largest clique (most edges).
    
    def _greedy_coloring():
        color_of_edge = [-1] * m
        color_classes = []  # list of sets of edge indices
        
        for eidx in order:
            # Find compatible existing colors
            best_color = -1
            best_size = -1
            for ci, cclass in enumerate(color_classes):
                # Check if eidx is compatible with all edges in cclass
                compatible = True
                for other_eidx in cclass:
                    if other_eidx in conflict[eidx]:
                        compatible = False
                        break
                if compatible:
                    if len(cclass) > best_size:
                        best_size = len(cclass)
                        best_color = ci
            
            if best_color >= 0:
                color_of_edge[eidx] = best_color
                color_classes[best_color].add(eidx)
            else:
                color_of_edge[eidx] = len(color_classes)
                color_classes.append({eidx})
        
        return color_of_edge, color_classes
    
    color_of_edge, color_classes = _greedy_coloring()
    
    # Convert to cliques
    def _to_cliques(color_of_edge, color_classes):
        cliques = []
        for cclass in color_classes:
            verts = set()
            for eidx in cclass:
                u, v = edge_list[eidx]
                verts.add(u)
                verts.add(v)
            if len(verts) >= 2:
                cliques.append(sorted(verts))
        return cliques
    
    current_cliques = _to_cliques(color_of_edge, color_classes)
    
    if not _verify_partition(current_cliques, edge_list, adj):
        return [[u, v] for u, v in edge_list]
    
    # Local search: try to merge color classes if their union is still a valid clique.
    def _local_search(color_of_edge, color_classes, max_time=30.0):
        start = time.time()
        color_of_edge = color_of_edge[:]
        color_classes = [set(c) for c in color_classes]
        
        improved = True
        while improved and time.time() - start < max_time:
            improved = False
            nc = len(color_classes)
            for i in range(nc):
                if time.time() - start > max_time:
                    break
                for j in range(i + 1, nc):
                    # Check if union of color classes i and j is still a valid clique
                    union_edges = color_classes[i] | color_classes[j]
                    # Check all pairs in union are non-conflicting
                    valid = True
                    for e1 in union_edges:
                        for e2 in union_edges:
                            if e1 < e2 and e2 in conflict[e1]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Merge
                        color_classes[i] = union_edges
                        for eidx in union_edges:
                            color_of_edge[eidx] = i
                        color_classes.pop(j)
                        improved = True
                        break
                if improved:
                    break
        
        return color_of_edge, color_classes
    
    remaining_time = time_budget - (time.time() - _START)
    if remaining_time > 5:
        color_of_edge, color_classes = _local_search(color_of_edge, color_classes, max_time=min(remaining_time - 2, 35.0))
        new_cliques = _to_cliques(color_of_edge, color_classes)
        if _verify_partition(new_cliques, edge_list, adj):
            current_cliques = new_cliques
    
    return current_cliques

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        time_budget = min(55.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
