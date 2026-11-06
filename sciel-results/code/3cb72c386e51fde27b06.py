import time
from collections import deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def max_flow_2_matching(n, edges):
        """
        Finds a maximum 2-matching in the graph (n, edges).
        Returns a set of edges that form a 2-matching (maximal set of edges 
        such that each vertex is incident to at most 2 of them).
        """
        # Build Flow Network
        # Nodes: 0..n-1 (in), n..2n-1 (out)
        # Source: 2n, Sink: 2n+1
        # Edge v_in -> v_out capacity 2
        # Edge u_out -> v_in capacity 1 for each undirected edge {u,v}
        # Edge Source -> v_out capacity 2
        # Edge v_in -> Sink capacity 2
        
        N = 2 * n + 2
        source = 2 * n
        sink = 2 * n + 1
        
        # Adjacency list for residual graph
        # graph[u] = list of [v, cap, rev_index]
        graph = [[] for _ in range(N)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
            
        # Vertex capacities
        for v in range(n):
            v_in = v
            v_out = n + v
            add_edge(v_in, v_out, 2)
            add_edge(source, v_out, 2)
            add_edge(v_in, sink, 2)
            
        # Graph edges
        edge_map = {} # flow edge index -> original edge
        for u, v in edges:
            u_out = n + u
            v_in = v
            u_in = u
            v_out = n + v
            
            # u_out -> v_in
            idx1 = len(graph[u_out])
            add_edge(u_out, v_in, 1)
            edge_map[(u_out, v_in)] = (u, v)
            
            # v_out -> u_in
            idx2 = len(graph[v_out])
            add_edge(v_out, u_in, 1)
            edge_map[(v_out, u_in)] = (v, u)

        # BFS to find augmenting paths (Edmonds-Karp is fine for small n)
        def bfs():
            parent = [-1] * N
            q = deque()
            q.append(source)
            parent[source] = -2
            while q:
                u = q.popleft()
                for v, cap, rev in graph[u]:
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == sink:
                            return parent
                        q.append(v)
            return None

        # Find max flow
        while True:
            parent = bfs()
            if parent is None:
                break
            # Find min residual capacity on path
            flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                # Find edge u->v
                for edge in graph[u]:
                    if edge[0] == v:
                        flow = min(flow, edge[1])
                        break
                v = u
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                for i, edge in enumerate(graph[u]):
                    if edge[0] == v:
                        graph[u][i][1] -= flow
                        graph[v][graph[u][i][2]][1] += flow
                        break
                v = u

        # Extract edges with flow
        # An edge in original graph has flow if the corresponding directed edge has cap 0 (and original cap was 1)
        # Or simpler: check if flow was pushed.
        # We can reconstruct by checking which edges have residual cap 0 in the forward direction?
        # No, because we might have sent flow back.
        # Better: track flow explicitly or check if the edge is "saturated" in a way that indicates selection.
        # Actually, for 2-matching, an edge is selected if it carries flow in the final state.
        # Since capacities are 1, if the edge is selected, the residual forward cap is 0 and reverse cap is 1.
        
        selected_edges = set()
        for u, v in edges:
            u_out = n + u
            v_in = v
            # Check u_out -> v_in
            for edge in graph[u_out]:
                if edge[0] == v_in:
                    if edge[1] == 0: # Forward cap 0 means flow 1 (since cap was 1)
                        selected_edges.add((u, v))
                    break
            # Check v_out -> u_in (just in case, though symmetric logic applies)
            # We only need to add the undirected edge once
            # The check above is sufficient if we iterate over unique undirected edges
            
        return selected_edges

    def extract_cycles(n, edges):
        """
        Given a set of edges forming a 2-matching (deg <= 2),
        extract all cycles.
        Returns list of cycles (each cycle is a list of vertices).
        """
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        cycles = []
        
        for start in range(n):
            if start in visited or len(adj[start]) < 2:
                continue
            # Try to find a cycle starting from start
            # Since it's a 2-matching, it's a collection of paths and cycles.
            # We can just walk the graph.
            
            # Find a node in a cycle.
            # A node is in a cycle if following the path leads back to itself.
            # In a 2-matching, every component is a path or a cycle.
            # We can identify cycles by finding nodes where the two neighbors are distinct and part of a loop.
            
            # Simple approach: DFS/Path tracing
            # Since deg <= 2, we can just trace.
            if start in visited:
                continue
                
            # Trace path
            path = [start]
            visited.add(start)
            prev = -1
            curr = start
            
            # We need to be careful not to infinite loop if it's a path.
            # But if it's a cycle, we will return to start.
            
            # Let's just find cycles by checking if a component is a cycle.
            # A component is a cycle if all nodes in it have degree 2.
            
            # Get component
            comp = set()
            q = deque([start])
            comp.add(start)
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if nei not in comp:
                        comp.add(nei)
                        q.append(nei)
                        
            # Check if all nodes in comp have degree 2
            is_cycle = True
            for node in comp:
                if len(adj[node]) != 2:
                    is_cycle = False
                    break
                    
            if is_cycle:
                # Reconstruct cycle order
                # Start from start, go to one neighbor, keep going until back to start
                cycle = [start]
                visited.update(comp)
                curr = start
                prev = -1
                while True:
                    nexts = [x for x in adj[curr] if x != prev]
                    if not nexts:
                        break
                    next_node = nexts[0]
                    if next_node == start:
                        break
                    cycle.append(next_node)
                    visited.add(next_node)
                    prev = curr
                    curr = next_node
                if len(cycle) >= 3:
                    cycles.append(cycle)
            else:
                # Mark as visited to avoid re-processing
                visited.update(comp)
                
        return cycles

    def decompose(edges, n):
        remaining = set(edges)
        pieces = []
        
        # Iteratively extract max 2-matchings and their cycles
        # This is a heuristic: taking the max 2-matching of the remaining graph
        # tends to pack edges into cycles.
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            if not remaining:
                break
                
            # Find max 2-matching on remaining edges
            matching_edges = max_flow_2_matching(n, remaining)
            
            if not matching_edges:
                break
                
            # Extract cycles from this matching
            cycles = extract_cycles(n, matching_edges)
            
            if not cycles:
                # If no cycles found, but matching edges exist, they are paths.
                # We cannot use paths as cycles.
                # We might be stuck if we just remove matching edges.
                # But we want to keep edges that can form cycles later?
                # Actually, if the max 2-matching has no cycles, it means the graph is a forest of paths.
                # In that case, no cycle can be formed from the remaining edges?
                # Not necessarily. A cycle might be "blocked" by the matching structure?
                # No, if a cycle exists in the graph, it is a 2-matching itself.
                # So if max 2-matching has no cycles, there are no cycles in the graph.
                # Proof: Any cycle is a valid 2-matching. If a cycle existed, the max 2-matching
                # (which maximizes edge count) would likely include it or a superset?
                # Actually, max 2-matching maximizes the number of edges.
                # If a cycle of length k exists, it has k edges.
                # If the max 2-matching has fewer edges than a potential cycle packing, we are not optimal.
                # But if the max 2-matching contains NO cycles, it consists of paths.
                # Does that imply no cycles in G?
                # Consider a triangle. Max 2-matching is the triangle (3 edges). It is a cycle.
                # Consider a square with a diagonal. Max 2-matching might be 4 edges (the square) or 3 edges (triangle).
                # If it picks the square, it's a cycle.
                # If the graph has no cycles, it's a forest. Max 2-matching is a spanning forest (max edges in forest is n-c).
                # So yes, if max 2-matching has no cycles, G has no cycles.
                break
                
            # Add cycles to pieces
            for cycle in cycles:
                pieces.append(cycle)
                # Remove cycle edges from remaining
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                        
        # Remaining edges are singletons
        for u, v in sorted(remaining):
            pieces.append([u, v])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
