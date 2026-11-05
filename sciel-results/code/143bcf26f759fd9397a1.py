"""
Path decomposition via minimum-cost pairing of odd-degree vertices + Eulerian trail extraction.
This is a structural change from iterative greedy merging to a global optimization approach.

The key insight: any edge decomposition into paths is equivalent to choosing, at each vertex,
how to pair up the incident half-edges. Vertices with even degree can be fully paired (Eulerian
components). Vertices with odd degree must have one "unpaired" half-edge (a path endpoint).
The number of paths = (# odd-degree vertices) / 2.

To minimize paths, we want to minimize the number of path endpoints, which means we want
as many vertices as possible to have even degree in our "pairing structure". But wait - the
degrees are fixed by the graph! So the number of odd-degree vertices is fixed, and the
minimum number of paths is exactly (# odd-degree vertices) / 2.

Actually, that's the key: the minimum path cover of a connected graph is (# odd-degree vertices)/2.
This is a well-known result. So we just need to find an optimal pairing at each vertex.

Algorithm:
1. Compute degrees. Identify odd-degree vertices.
2. At each vertex, pair up incident edges. For odd-degree vertices, leave one edge unpaired.
   The choice of which edge to leave unpaired affects which vertices become path endpoints.
3. We need to choose, for each odd-degree vertex, which incident edge is "unpaired" (the path endpoint),
   such that the resulting structure forms valid paths (no cycles, no branching).

Actually, a simpler approach: 
- The minimum number of paths is exactly (number of odd-degree vertices) / 2.
- To achieve this, we need to find a pairing of half-edges at each vertex such that the resulting
  directed structure is a collection of paths (no cycles).

A clean algorithm:
1. For each vertex, list its incident edges.
2. At even-degree vertices: pair all edges arbitrarily (or optimally).
3. At odd-degree vertices: pair all but one edge. The unpaired edge determines the path endpoint.
4. The challenge is to choose which edge is unpaired at each odd vertex so that the overall structure
   is a set of paths (no cycles).

A standard approach:
- Pick a spanning tree or use a flow-based method.
- Alternatively: randomly assign the "unpaired" edge at each odd vertex, then check for cycles.
  If cycles exist, fix them by swapping pairings.

Actually, the simplest correct approach:
1. Compute the set of odd-degree vertices: O.
2. The minimum number of paths is |O|/2.
3. To construct the paths:
   a. For each vertex, create a list of incident edge indices.
   b. At even-degree vertices: pair up the incident edges (order doesn't matter for path count, only for which paths form).
   c. At odd-degree vertices: leave one edge unpaired.
   d. The unpaired edges at odd vertices are the path endpoints.
   e. Trace paths by following the pairings.

The key decision: at each odd vertex, which edge is unpaired?
This determines the path endpoints. We need to choose so that no cycles are formed.

A clean method:
1. Assign to each odd vertex a "preferred" unpaired edge (e.g., the first incident edge).
2. Build the pairing structure.
3. Trace all paths. If any cycle is found (a closed loop with no endpoints), we have a problem.
4. To avoid cycles: we can use a global approach. 

Actually, here's a clean constructive proof:
- Pick any spanning tree T of the graph.
- At each vertex, pair the incident edges of T in a specific way, and pair the non-tree edges among themselves.
- This guarantees no cycles (since the tree structure is preserved in a path-like way).

Let me think more carefully. Here's a standard algorithm for minimum path cover:

1. Let O be the set of odd-degree vertices. |O| is even.
2. We need to pair up the |O| odd vertices into |O|/2 pairs, where each pair (u, v) means there's a path from u to v.
3. But the pairing must be "realizable" - there must be edge-disjoint paths connecting each pair.

Actually, the simpler view: 
- The minimum path cover size is |O|/2.
- To construct it: 
  a. At each vertex v, we need to pair up the incident edges. If deg(v) is even, pair all. If deg(v) is odd, pair all but one (the "open" edge).
  b. The "open" edges at odd vertices are the path endpoints.
  c. We need to choose which edge is "open" at each odd vertex.
  d. Then trace the paths.

The constraint is: the resulting structure must be a set of paths (no cycles, no branching).

A safe approach: 
1. For each odd vertex, tentatively choose the first incident edge as "open".
2. Build the pairing.
3. Trace paths. If a cycle is detected, we need to fix it.
4. To fix: find a path that can be "rotated" to break the cycle.

Actually, let me use a cleaner method based on the fact that we can always achieve |O|/2 paths:

Method:
1. Compute degrees and identify odd vertices.
2. For each vertex, create a list of incident edges.
3. At each vertex, pair up the incident edges in a specific order. For odd vertices, the last edge is left unpaired.
4. The key is to choose the pairing order to avoid cycles.

Here's a robust algorithm:
1. For each vertex v, sort its incident edges by some criterion.
2. At even-degree vertices: pair (e0,e1), (e2,e3), etc.
3. At odd-degree vertices: pair (e0,e1), (e2,e3), ..., and leave the last edge unpaired.
4. This creates a pairing structure. Trace the paths.
5. If cycles are formed, they can be broken by swapping pairings.

For simplicity and correctness, I'll use the following approach:
1. Build the pairing structure with a specific ordering.
2. Trace paths.
3. If any cycle is found (a component with no endpoints), it means we have a cycle plus paths. We can break the cycle by "opening" it at any vertex, which increases the path count by 1 (from 0 paths for the cycle to 1 path). But this would give us more than |O|/2 paths.

Wait, I need to be more careful. Let me just implement the standard approach:

1. The minimum number of paths is |O|/2 where O is the set of odd-degree vertices.
2. To achieve this:
   a. For each vertex, list incident edges.
   b. At each vertex, pair up incident edges. At odd vertices, one edge is left unpaired.
   c. The unpaired edges at odd vertices are the path endpoints.
   d. We need to ensure no cycles.

A clean constructive method:
1. Pick a spanning tree T.
2. Root T at an arbitrary vertex r.
3. For each vertex v (processed in BFS/DFS order from r):
   - Let the children of v in T be c1, c2, ..., ck.
   - Let the non-tree edges incident to v be f1, f2, ..., fm.
   - If v is the root:
     - If deg(v) is even: pair all incident edges.
     - If deg(v) is odd: leave one edge unpaired (this makes r a path endpoint).
   - If v is not the root:
     - The edge to the parent of v is "reserved" (it will be paired with something at v).
     - Pair the remaining edges.

This is getting complex. Let me just use a simple heuristic that works well in practice:

1. For each vertex, sort incident edges by the other endpoint's index.
2. At even-degree vertices: pair consecutively.
3. At odd-degree vertices: pair consecutively, leave the last one unpaired.
4. Trace paths. If cycles are found, fix them by swapping.

For the given small graphs (n=20,25,30), this should work well. Let me implement it.
"""
import time
from collections import defaultdict, deque
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_one(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency: vertex -> list of (neighbor, edge_index)
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        # Compute degrees
        deg = {v: len(adj[v]) for v in adj}
        odd_vertices = [v for v in adj if deg[v] % 2 == 1]
        
        # The minimum number of paths is len(odd_vertices) // 2
        # We need to pair up incident edges at each vertex.
        # At odd vertices, one edge is left unpaired (path endpoint).
        
        # Strategy: 
        # 1. For each vertex, sort incident edges by the other endpoint.
        # 2. At even-degree vertices: pair (0,1), (2,3), etc.
        # 3. At odd-degree vertices: pair (0,1), (2,3), ..., leave last unpaired.
        # 4. This creates a pairing structure. Trace paths.
        # 5. If cycles are found, we need to fix them.
        
        # Build pairing: for each vertex, a list of pairs of edge indices.
        # pairing[v] = list of (edge_idx1, edge_idx2) pairs, and possibly one unpaired edge.
        
        pairing = {}
        unpaired = {}  # vertex -> edge_index (only for odd vertices)
        
        for v in adj:
            # Sort incident edges by the other endpoint
            incident = sorted(adj[v], key=lambda x: x[0])
            d = len(incident)
            
            if d % 2 == 0:
                # Pair all
                pairs = []
                for i in range(0, d, 2):
                    pairs.append((incident[i][1], incident[i+1][1]))
                pairing[v] = pairs
            else:
                # Pair all but last
                pairs = []
                for i in range(0, d-1, 2):
                    pairs.append((incident[i][1], incident[i+1][1]))
                pairing[v] = pairs
                unpaired[v] = incident[-1][1]
        
        # Now trace paths.
        # Each edge has two endpoints. At each endpoint, the edge is either paired with another edge or unpaired.
        # We can build a graph where nodes are edge endpoints, and edges connect paired endpoints.
        # Then paths in the original graph correspond to paths in this "pairing graph".
        
        # Actually, let's trace directly:
        # Start from each unpaired edge endpoint. Follow the pairing to build the path.
        
        # For each edge, we need to know: at each endpoint, which edge is it paired with?
        # edge_partner[edge_idx][0] = the edge that edge_idx is paired with at its first endpoint (or -1 if unpaired)
        # edge_partner[edge_idx][1] = the edge that edge_idx is paired with at its second endpoint (or -1 if unpaired)
        
        # Let's build this.
        # For each vertex v, and each pair (e1, e2) in pairing[v]:
        #   For e1: at the endpoint corresponding to v, it's paired with e2.
        #   For e2: at the endpoint corresponding to v, it's paired with e1.
        
        # We need to know, for each edge, which endpoint corresponds to which vertex.
        # edges[e] = (u, v). So endpoint 0 is u, endpoint 1 is v.
        
        # edge_partner[e] = [partner_at_u, partner_at_v]
        
        edge_partner = [[-1, -1] for _ in range(len(edges))]
        
        for v in adj:
            for e1, e2 in pairing[v]:
                # For edge e1, at vertex v, it's paired with e2.
                # Determine which endpoint of e1 corresponds to v.
                u1, v1 = edges[e1]
                if u1 == v:
                    edge_partner[e1][0] = e2
                else:
                    edge_partner[e1][1] = e2
                
                # For edge e2, at vertex v, it's paired with e1.
                u2, v2 = edges[e2]
                if u2 == v:
                    edge_partner[e2][0] = e1
                else:
                    edge_partner[e2][1] = e1
        
        # Now, trace paths.
        # A path starts at an unpaired endpoint and follows the pairing.
        # We need to track visited edges to avoid infinite loops.
        
        visited_edges = set()
        paths = []
        
        # Start from each unpaired endpoint
        for v in unpaired:
            e_start = unpaired[v]
            if e_start in visited_edges:
                continue
            
            # Trace the path starting from edge e_start at vertex v.
            path_vertices = []
            current_edge = e_start
            current_vertex = v
            direction = 0  # 0 means we're at the first endpoint of the edge, 1 means second
            
            while True:
                # Add the current vertex to the path
                if not path_vertices:
                    path_vertices.append(current_vertex)
                else:
                    # Check if current_vertex is already in the path (would indicate a cycle)
                    if current_vertex in path_vertices:
                        # This shouldn't happen if the pairing is correct.
                        # But if it does, we have a cycle. Break.
                        break
                    path_vertices.append(current_vertex)
                
                # Move to the next edge via the pairing at current_vertex.
                # Determine which endpoint of current_edge corresponds to current_vertex.
                u, w = edges[current_edge]
                if u == current_vertex:
                    next_edge = edge_partner[current_edge][0]
                    next_vertex = w
                else:
                    next_edge = edge_partner[current_edge][1]
                    next_vertex = u
                
                visited_edges.add(current_edge)
                
                if next_edge == -1:
                    # This is the end of the path.
                    break
                
                current_edge = next_edge
                current_vertex = next_vertex
        
        # Check for cycles: any edge not visited is part of a cycle.
        # This shouldn't happen if we have the correct number of unpaired endpoints.
        # But just in case, let's handle it.
        for e in range(len(edges)):
            if e not in visited_edges:
                # This edge is part of a cycle. Trace the cycle.
                if time.time() > DEADLINE - 5:
                    break
                path_vertices = []
                current_edge = e
                u, w = edges[current_edge]
                current_vertex = u
                while True:
                    path_vertices.append(current_vertex)
                    visited_edges.add(current_edge)
                    
                    # Move to next edge
                    if u == current_vertex:
                        next_edge = edge_partner[current_edge][0]
                        next_vertex = w
                    else:
                        next_edge = edge_partner[current_edge][1]
                        next_vertex = u
                    
                    if next_edge == -1 or next_edge in visited_edges:
                        break
                    
                    current_edge = next_edge
                    current_vertex = next_vertex
                
                if path_vertices:
                    paths.append(path_vertices)
        
        # Remove any empty paths
        paths = [p for p in paths if len(p) >= 2]
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 5:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        n, p, seed = spec
        paths = solve_one(n, p, seed)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    r = construct()
    for i, lst in enumerate(r):
        print(i, len(lst))
    print("Total:", sum(len(lst) for lst in r))
