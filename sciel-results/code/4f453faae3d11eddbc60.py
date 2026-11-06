import time
from random import Random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        n_edges = len(edges)
        
        # For each edge, we have two endpoints.
        # We want to orient each edge (choose which endpoint is start, which is end)
        # and chain edges together to form paths, minimizing the number of paths.
        
        # Key insight: The minimum number of paths = |E| - (max number of edge-to-edge links)
        # A link between edge i and edge j exists if they share a vertex v, and we orient
        # edge i to end at v and edge j to start at v.
        
        # This is a maximum matching problem in a bipartite graph where:
        # - Left side: for each edge e, one node representing "e as a potential end"
        # - Right side: for each edge e, one node representing "e as a potential start"  
        # - But we need to track which vertex the link is at.
        
        # More precisely:
        # - Left side: for each edge e=(u,v), two nodes: e_u (e ends at u) and e_v (e ends at v)
        #   But each edge can only be used once as an end, so we need to choose one.
        #
        # This is getting complicated. Let me use a different formulation.
        
        # Alternative formulation using flow:
        # - Source -> edge nodes -> vertex nodes -> edge nodes -> sink
        # - But this is for vertex-disjoint paths.
        
        # Let me use the following correct formulation:
        # Create a directed graph where each undirected edge (u,v) becomes two directed edges u->v and v->u.
        # We want to select one direction for each undirected edge such that:
        # - At each vertex, in-degree = out-degree (for internal vertices of paths)
        #   OR in-degree = out-degree + 1 (for start of a path)
        #   OR out-degree = in-degree + 1 (for end of a path)
        # - Minimize the number of path starts (= number of path ends)
        
        # This is equivalent to: maximize the number of edges that are "internal" 
        # (i.e., their start vertex is also the end vertex of another edge).
        
        # The maximum number of such internal connections is found by:
        # For each vertex v, let d(v) be the degree. The maximum number of edges that
        # can end at v is floor(d(v)/2) if we pair them up, but actually it's more subtle.
        
        # I'll use a maximum matching approach on a specific bipartite graph.
        
        # Bipartite graph construction:
        # Left nodes: one for each edge e, representing "edge e is the last edge before a vertex"
        # Right nodes: one for each edge e, representing "edge e is the first edge after a vertex"
        # Edge from left[i] to right[j] if edge i and edge j share a vertex v,
        # and we can orient edge i to end at v and edge j to start at v.
        
        # But an edge has two endpoints, so for a pair of edges sharing vertex v,
        # there's exactly one way to chain them through v (edge i ends at v, edge j starts at v).
        
        # However, an edge can only be chained once at each endpoint. So edge e can be
        # the "last edge" at most once (at one of its endpoints) and the "first edge" at most once.
        
        # This means: create a bipartite graph with:
        # - Left: n_edges nodes (each edge can be a "last edge" at most once)
        # - Right: n_edges nodes (each edge can be a "first edge" at most once)
        # - Edge from left[i] to right[j] if edges i and j share a vertex.
        #   But which vertex? If they share multiple vertices (possible in multigraph, but not here),
        #   we'd have multiple edges. In a simple graph, two edges share at most one vertex.
        
        # Wait, in a simple graph, two distinct edges share at most one vertex. So for each pair
        # of edges sharing a vertex v, there's one potential link. But we need to ensure that
        # the orientation is consistent: edge i ends at v and edge j starts at v.
        
        # The issue: an edge e=(u,v) could be chained at u or at v, but not both as a "last edge".
        # So we need to track which endpoint is used.
        
        # Refined formulation:
        # Left side: for each edge e=(u,v), two nodes: e@u (e ends at u) and e@v (e ends at v)
        # Right side: for each edge e=(u,v), two nodes: e@u (e starts at u) and e@v (e starts at v)
        # But we can only use one of e@u or e@v on the left side, and one on the right side.
        
        # This is a more complex matching problem. Let me use the following approach:
        # Since n is small (<=30), I'll enumerate all possible orientations and use a maximum matching.
        
        # Actually, here's a cleaner way:
        # The problem is equivalent to finding a maximum set of edge-to-edge adjacencies
        # where each edge is used at most twice (once at each endpoint).
        
        # This can be formulated as a maximum matching in a graph where:
        # - Vertices: all "ports" (edge, endpoint) pairs. There are 2*n_edges ports.
        # - Edges: between two ports if they belong to the same vertex v in G and belong to different edges.
        # - We want a matching where each port is used at most once.
        # - The size of the matching = number of adjacencies.
        # - Number of paths = n_edges - matching_size.
        
        # This is exactly what I had before! Let me implement it correctly with a proper max matching.
        
        # Collect ports
        ports = []  # (edge_idx, vertex)
        for idx, (u, v) in enumerate(edges):
            ports.append((idx, u))
            ports.append((idx, v))
        
        n_ports = len(ports)
        
        # Group ports by vertex
        vertex_ports = {}
        for i, (e_idx, v) in enumerate(ports):
            if v not in vertex_ports:
                vertex_ports[v] = []
            vertex_ports[v].append(i)
        
        # Build adjacency for bipartite matching
        # Left node i can match to right node j if:
        #   - They are at the same vertex v
        #   - They belong to different edges
        # Note: left and right are the same set of ports, so we're matching within each vertex's ports.
        
        # For each vertex v, the ports at v form a complete bipartite graph between
        # left ports at v and right ports at v, excluding self-matches and same-edge matches.
        
        # Since the graph is a union of disjoint complete bipartite subgraphs (one per vertex),
        # the maximum matching can be found by solving each vertex independently... 
        # BUT ports can appear in multiple vertices? No, each port (e, v) is associated with exactly one vertex v.
        
        # Wait, each port is (edge_idx, vertex), and each edge has two ports at two different vertices.
        # So port (e, u) is only at vertex u, and port (e, v) is only at vertex v.
        # Therefore, the bipartite graph is a disjoint union of complete bipartite graphs, one per vertex.
        
        # In a complete bipartite graph K_{m,m}, the maximum matching size is m.
        # But we have constraints: left port (e, v) and right port (e, v) cannot be matched together
        # (same edge), and also left port (e1, v) cannot be matched to right port (e1, v).
        
        # Actually, in the complete bipartite graph for vertex v, we have:
        # Left nodes: all ports (e, v) for edges e incident to v
        # Right nodes: all ports (e, v) for edges e incident to v
        # We cannot match left (e, v) to right (e, v) because that would mean edge e is chained to itself.
        
        # So it's a complete bipartite graph minus a perfect matching (the self-matches).
        # The maximum matching in K_{m,m} minus a perfect matching is m-1 if m>=2, and 0 if m=1.
        # Wait, no. K_{m,m} has maximum matching m. Removing a perfect matching (m edges),
        # the remaining graph still has a matching of size m-1? Let me think...
        
        # For m=2: K_{2,2} has 4 edges. Remove 2 diagonal edges. Remaining: 2 edges, which form a matching of size 2.
        # For m=3: K_{3,3} has 9 edges. Remove 3 diagonal edges. Remaining: 6 edges. Max matching is 3? 
        #   Left: L1,L2,L3. Right: R1,R2,R3. Edges: L1-R2, L1-R3, L2-R1, L2-R3, L3-R1, L3-R2.
        #   Matching: L1-R2, L2-R3, L3-R1. Size 3. Yes, it's m.
        
        # Actually, K_{m,m} minus a perfect matching still has a perfect matching for m>=2.
        # So the maximum matching at each vertex v with degree d(v) is d(v) if d(v)>=2, and 0 if d(v)=1.
        
        # Wait, that can't be right because it would give matching_size = sum_v max(0, d(v)-1) or something.
        
        # Let me reconsider. At vertex v with degree d, we have d ports on the left and d ports on the right.
        # We want to match left ports to right ports, but not left port i to right port i (same edge).
        # This is equivalent to finding a derangement of d elements, which exists for d>=2.
        # The maximum matching size is d for d>=2, and 0 for d=1.
        
        # But wait, that would mean matching_size = sum over vertices v of (d(v) if d(v)>=2 else 0).
        # And number of paths = n_edges - matching_size.
        
        # Let's verify with a simple example: a path of 3 vertices (2 edges).
        # Vertices: 0-1-2. Edges: (0,1), (1,2).
        # Degrees: d(0)=1, d(1)=2, d(2)=1.
        # Matching at v=0: d=1, matching=0.
        # Matching at v=1: d=2, matching=2.
        # Matching at v=2: d=1, matching=0.
        # Total matching = 2. Number of paths = 2 - 2 = 0. But we need 1 path!
        
        # Something is wrong. The issue is that each port can only be used once in the matching,
        # but in my formulation, port (edge0, 1) appears at vertex 1, and port (edge1, 1) also appears at vertex 1.
        # If we match them at vertex 1, that's one adjacency. But we have two ports at vertex 1, so we can only form one adjacency there.
        
        # I think I'm confusing myself. Let me restart the formulation.
        
        # Each adjacency uses two ports at the same vertex: one from edge e1 and one from edge e2.
        # Each port can be used in at most one adjacency.
        # At vertex v with degree d(v), there are d(v) ports. The maximum number of adjacencies at v is floor(d(v)/2).
        
        # So total maximum adjacencies = sum over all vertices v of floor(d(v)/2).
        
        # For the path 0-1-2: d(0)=1, d(1)=2, d(2)=1. Sum of floor(d/2) = 0 + 1 + 0 = 1.
        # Number of paths = 2 - 1 = 1. Correct!
        
        # For a triangle (3 edges): d(0)=2, d(1)=2, d(2)=2. Sum = 1+1+1=3. Paths = 3-3=0. But we need 1 path (or 3 paths if we don't chain).
        # A triangle can be covered by 1 path of length 3? No, a path of 3 edges in a triangle would visit 4 vertices, but we only have 3.
        # Actually, a triangle's edges can be covered by 1 path: e.g., 0-1-2-0 is a cycle, not a path. 
        # A path is a sequence of vertices where consecutive vertices are connected by an edge, and no vertex is repeated.
        # In a triangle, the longest path has 2 edges (3 vertices). So we need at least 2 paths to cover 3 edges? 
        # Path 1: 0-1 (edge 0-1). Path 2: 1-2-0 (edges 1-2 and 2-0). That's 2 paths.
        # Or: Path 1: 0-1-2 (edges 0-1, 1-2). Path 2: 2-0 (edge 2-0). That's 2 paths.
        # So minimum is 2 paths.
        
        # Using my formula: sum floor(d/2) = 3. Paths = 3-3 = 0. That's wrong.
        
        # The issue: an edge can be used in at most 2 adjacencies (one at each endpoint), but in the triangle,
        # if we use all 3 adjacencies, we'd have each edge used twice, which would form a cycle, not paths.
        
        # So the constraint is not just floor(d(v)/2) per vertex, but also that the resulting structure must be a collection of paths, not cycles.
        
        # This is getting too complex. Let me just use a greedy approach that works well in practice.
        
        # Greedy approach:
        # 1. Start with all edges as separate paths of length 1.
        # 2. Repeatedly try to merge two paths by finding an endpoint of one path and an endpoint of another
        #    that are connected by an edge. Merge them.
        # 3. Continue until no more merges are possible.
        
        # This is a simple greedy that should give good results.
        
        # Represent each path as a list of vertices.
        # Two paths can be merged if:
        #   - The last vertex of path A is connected to the first vertex of path B
        #   - The first vertex of path A is connected to the last vertex of path B
        #   - etc.
        
        # Actually, for edge-disjoint paths covering all edges, we need to be more careful.
        
        # Let me use the following approach:
        # 1. Build the graph.
        # 2. Find an Eulerian trail in each connected component after adding dummy edges to make all degrees even.
        # 3. The number of paths in the decomposition is related to the number of odd-degree vertices.
        
        # Standard result: The minimum number of paths to cover all edges of a graph is:
        #   k = max(1, ceil(odd_vertices / 2)) if there are edges, where odd_vertices is the number of vertices with odd degree.
        # Wait, that's not quite right either.
        
        # The correct formula: For a connected graph with E edges and O odd-degree vertices,
        # the minimum number of paths is max(1, O/2) if O > 0, else 1.
        # But this assumes we can rearrange the edges freely, which we can since we're just covering edges.
        
        # Actually, the minimum path decomposition of a graph's edges is:
        #   If the graph has no edges: 0
        #   If the graph has edges: max(1, number_of_odd_degree_vertices / 2)
        #   But this is only for connected graphs? No, it works for disconnected graphs too,
        #   with the understanding that each connected component contributes.
        
        # Let me verify: Path 0-1-2: odd vertices = {0, 2}, count=2. Min paths = max(1, 2/2) = 1. Correct.
        # Triangle: odd vertices = {} (all degree 2). Min paths = max(1, 0) = 1. But we showed it's 2!
        
        # So the formula is wrong for the triangle. The issue is that a cycle cannot be decomposed into 1 path;
        # it needs at least 1 path if we allow the path to not repeat vertices, but a cycle of length 3 cannot be a single path.
        
        # I think the correct approach is to use the fact that:
        #   Min paths = (E - max_matching) where max_matching is the maximum number of edge adjacencies.
        # And this can be found via maximum matching in the port graph, but with the constraint that
        # the result is acyclic (no cycles).
        
        # Given the complexity, let me just use a practical greedy algorithm that works well.
        
        # Greedy algorithm:
        # 1. Start with each edge as a separate path.
        # 2. While there exist two paths that can be merged:
        #      - Find the pair that gives the longest merged path (or just any merge).
        #      - Merge them.
        # 3. Return the paths.
        
        # To merge two paths P1 = [v1, v2, ..., vk] and P2 = [w1, w2, ..., wm]:
        #   - If vk is connected to w1, merge to [v1, ..., vk, w1, ..., wm]
        #   - If vk is connected to wm, merge to [v1, ..., vk, wm, ..., w1]
        #   - If v1 is connected to w1, merge to [w1, ..., wm, v1, ..., vk] (reverse P1)
        #   - If v1 is connected to wm, merge to [w1, ..., wm, vk, ..., v1] (reverse P1)
        
        # This is O(n^2) per merge, and we do at most E-1 merges, so O(E^2 * n) total, which is fine for E<=30.
        
        # Build adjacency set for quick lookup
        adj_set = set()
        for u, v in edges:
            adj_set.add((u, v))
            adj_set.add((v, u))
        
        # Initialize paths: each edge is a path of 2 vertices
        paths = []
        for u, v in edges:
            paths.append([u, v])
        
        # Try to merge paths
        merged = True
        while merged:
            merged = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    P1 = paths[i]
                    P2 = paths[j]
                    
                    # Check if we can merge P1 and P2
                    new_path = None
                    
                    # Try P1 + P2
                    if (P1[-1], P2[0]) in adj_set:
                        new_path = P1 + P2
                    elif (P1[-1], P2[-1]) in adj_set:
                        new_path = P1 + P2[::-1]
                    elif (P1[0], P2[0]) in adj_set:
                        new_path = P1[::-1] + P2
                    elif (P1[0], P2[-1]) in adj_set:
                        new_path = P1[::-1] + P2[::-1]
                    
                    if new_path is not None:
                        # Check if new_path is simple (no repeated vertices)
                        if len(set(new_path)) == len(new_path):
                            # Check that all consecutive vertices are connected
                            valid = True
                            for k in range(len(new_path) - 1):
                                if (new_path[k], new_path[k+1]) not in adj_set:
                                    valid = False
                                    break
                            if valid:
                                paths[i] = new_path
                                paths.pop(j)
                                merged = True
                                break
                if merged:
                    break
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
