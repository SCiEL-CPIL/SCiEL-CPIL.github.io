"""New approach: Greedy vertex selection with Max-Flow verification for disjoint paths."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = int(P['n'])
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Helper: Max Flow (Edmonds-Karp) with vertex splitting
    # Nodes in flow network:
    # For each vertex v in the "available" set, create v_in, v_out
    # Edge v_in -> v_out with capacity 1
    # For each edge (u, v) in original graph, if both are available, add u_out -> v_in and v_out -> u_in with capacity INF
    # Source: v_out (for the candidate v)
    # Sinks: u_in (for each u in B)
    # Actually, to handle source/sinks properly:
    # We want to send flow from v to all u in B.
    # Create a super sink T.
    # Connect u_out to T with capacity 1 for each u in B?
    # No, the path ends at u. So flow enters u_in, goes to u_out, then to T.
    # So yes, connect u_out to T with cap 1.
    # Source is v_in? No, v is the start.
    # Connect S to v_in with cap 1.
    
    def max_flow(n_nodes, adj_list, source, sink, node_capacities):
        """
        n_nodes: number of nodes in flow network (after splitting)
        adj_list: list of lists of (neighbor, capacity)
        source, sink: node indices
        node_capacities: not used directly, capacities are in edges
        """
        # Edmonds-Karp
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * n_nodes
            capacity = [0] * n_nodes
            queue = deque([source])
            parent[source] = -2
            capacity[source] = float('inf')
            
            found = False
            while queue:
                u = queue.popleft()
                for v, cap in adj_list[u]:
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        capacity[v] = min(capacity[u], cap)
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
                if found:
                    break
            
            if not found:
                break
            
            # Update residual graph
            flow += capacity[sink]
            v = sink
            while v != source:
                u = parent[v]
                # Find the edge u->v
                for i, (neighbor, cap) in enumerate(adj_list[u]):
                    if neighbor == v:
                        adj_list[u][i] = (neighbor, cap - capacity[sink])
                        break
                # Reverse edge
                for i, (neighbor, cap) in enumerate(adj_list[v]):
                    if neighbor == u:
                        adj_list[v][i] = (neighbor, cap + capacity[sink])
                        break
                v = u
                
        return flow

    def can_add_vertex(v, B, U):
        """
        Check if v can be added to B given current branches B and used interior vertices U.
        Returns (True, paths) if possible, else (False, None).
        """
        if not B:
            return True, []
            
        # Nodes available for flow: V \ (B \cup U) \cup {v} \cup B
        # We only care about paths from v to B.
        # The interiors must be in V \ (B \cup U).
        
        # Let's map original vertices to flow network nodes.
        # Flow network nodes:
        # 0: Source S
        # 1: Sink T
        # For each node x in V \ (B \cup U):
        #   x_in, x_out
        # For v:
        #   v_in, v_out
        # For each u in B:
        #   u_in, u_out
        
        # Actually, we can simplify.
        # The graph for flow is induced by V_free = V \ (B \cup U).
        # Plus v and B.
        
        nodes_in_flow = set()
        nodes_in_flow.add(v)
        nodes_in_flow.update(B)
        nodes_in_flow.update(V_free := set(range(n)) - set(B) - U)
        
        # Mapping from original vertex to flow node indices
        # We need 2 nodes per vertex: in and out.
        # Let's assign indices.
        node_map = {}
        idx = 0
        # Reserve 0 for S, 1 for T? No, let's just add them.
        # S = idx, T = idx+1
        S = 0
        T = 1
        idx = 2
        
        for node in nodes_in_flow:
            node_map[node] = (idx, idx + 1) # in, out
            idx += 2
            
        n_flow_nodes = idx
        
        # Initialize adjacency list for flow network
        flow_adj = [[] for _ in range(n_flow_nodes)]
        
        # Add edges for vertex capacities
        for node in nodes_in_flow:
            in_idx, out_idx = node_map[node]
            flow_adj[in_idx].append((out_idx, 1))
            flow_adj[out_idx].append((in_idx, 0)) # Reverse edge initially 0
            
        # Add edges for graph edges
        # For each edge (u, w) in original graph:
        # If both u and w are in nodes_in_flow:
        #   Add u_out -> w_in with capacity INF
        #   Add w_out -> u_in with capacity INF
        
        INF = 1000
        for u in nodes_in_flow:
            if u not in node_map: continue
            u_in, u_out = node_map[u]
            for w in adj[u]:
                if w in node_map:
                    w_in, w_out = node_map[w]
                    flow_adj[u_out].append((w_in, INF))
                    flow_adj[w_out].append((u_in, INF)) # Wait, this adds reverse edges twice if we iterate all u?
                    # No, flow_adj is directed.
                    # We need to add the reverse edge for the residual graph.
                    # My max_flow function expects the residual graph to be built?
                    # No, my max_flow function modifies the adj_list in place.
                    # So I must initialize it with forward edges and 0-capacity reverse edges.
                    
        # Re-initialize flow_adj properly
        flow_adj = [[] for _ in range(n_flow_nodes)]
        
        # Helper to add edge
        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap])
            flow_adj[v].append([u, 0])
            
        # Add vertex capacity edges
        for node in nodes_in_flow:
            in_idx, out_idx = node_map[node]
            add_edge(in_idx, out_idx, 1)
            
        # Add graph edges
        for u in nodes_in_flow:
            if u not in node_map: continue
            u_in, u_out = node_map[u]
            for w in adj[u]:
                if w in node_map:
                    w_in, w_out = node_map[w]
                    add_edge(u_out, w_in, INF)
                    
        # Connect Source to v
        v_in, v_out = node_map[v]
        add_edge(S, v_in, 1)
        
        # Connect B nodes to Sink
        for u in B:
            u_in, u_out = node_map[u]
            add_edge(u_out, T, 1)
            
        # Run Max Flow
        max_f = max_flow(n_flow_nodes, flow_adj, S, T, None)
        
        if max_f == len(B):
            # Extract paths
            # This is complex. We need to trace the flow.
            # For now, return True and dummy paths?
            # The task requires returning the actual paths.
            # We need to reconstruct the paths from the residual graph.
            # This is tricky with the current max_flow implementation.
            
            # Let's try a simpler approach for path extraction:
            # Since n is small, we can just do BFS in the original graph 
            # restricted to available vertices, greedily assigning vertices to paths.
            
            # Re-do: Greedy BFS for paths
            # Available vertices: V_free
            # We need to connect v to each u in B.
            # We'll maintain a set of 'blocked' vertices (U).
            # For each u in B, find a path from v to u in G - (B \cup U_blocked).
            # If found, add interior vertices to U_blocked.
            
            # But this greedy approach might fail even if a solution exists.
            # However, for the purpose of this heuristic, if Max Flow says YES, 
            # a greedy BFS often finds a valid set of paths if we order them well.
            
            # Let's try to extract paths from the flow graph.
            # The flow is 1 on each path.
            # We can trace back from T to S for each unit of flow.
            
            paths = []
            # We need to find the specific edges used.
            # The max_flow function destroys the original capacities.
            # We should have kept a copy or returned the flow decomposition.
            
            # Given the complexity, let's assume if Max Flow succeeds, 
            # we can find the paths via a secondary BFS on the "used" edges.
            # An edge (u_out, w_in) has flow 1 if the residual capacity is INF-1.
            
            used_edges = []
            for u in nodes_in_flow:
                if u not in node_map: continue
                u_in, u_out = node_map[u]
                for w in adj[u]:
                    if w in node_map:
                        w_in, w_out = node_map[w]
                        # Check if flow u_out -> w_in is 1
                        # In flow_adj[u_out], find edge to w_in.
                        # If capacity is INF-1, then flow is 1.
                        for edge in flow_adj[u_out]:
                            if edge[0] == w_in:
                                if edge[1] == INF - 1:
                                    used_edges.append((u, w))
                                break
                        
            # Also check edges from S to v_in and u_out to T
            # S -> v_in
            for edge in flow_adj[S]:
                if edge[0] == v_in and edge[1] == 0: # Flow 1
                    pass # v is start
            
            for u in B:
                u_in, u_out = node_map[u]
                for edge in flow_adj[u_out]:
                    if edge[0] == T and edge[1] == 0:
                        pass # u is end
            
            # Now we have a set of edges that form disjoint paths from v to B.
            # We can reconstruct the paths by following the edges.
            # Since they are disjoint, we can just find the path from v to each u.
            
            # Build a directed graph of used edges
            path_adj = {node: [] for node in nodes_in_flow}
            for u, w in used_edges:
                path_adj[u].append(w)
                
            # Add start/end connections
            # v is start.
            # For each u in B, u is end.
            
            # Find path from v to each u
            for u in B:
                # BFS from v to u in path_adj
                visited = {v: None}
                queue = deque([v])
                while queue:
                    cur = queue.popleft()
                    if cur == u:
                        break
                    for next_node in path_adj[cur]:
                        if next_node not in visited:
                            visited[next_node] = cur
                            queue.append(next_node)
                            
                if u in visited:
                    # Reconstruct path
                    path = []
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = visited[cur]
                    path.reverse()
                    paths.append(path)
                else:
                    return False, None
                    
            return True, paths
            
        else:
            return False, None

    # Greedy Selection
    # Order vertices by degree (ascending)
    vertices = sorted(range(n), key=lambda x: len(adj[x]))
    
    B = []
    U = set()
    all_paths = []
    
    for v in vertices:
        if v in B or v in U:
            continue
            
        ok, paths = can_add_vertex(v, B, U)
        if ok:
            B.append(v)
            # Update U with interior vertices of the new paths
            for p in paths:
                for node in p[1:-1]:
                    U.add(node)
            all_paths.extend(paths)
            
    # If B is empty or has 1 vertex, fallback
    if len(B) < 2:
        # Fallback to first edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    # Reconstruct all paths for all pairs in B?
    # The task requires paths for ALL unordered pairs of branches.
    # My greedy method only found paths from the *last added* vertex to existing ones.
    # It did NOT find paths between existing branches!
    # This is a major flaw.
    
    # The paths must exist for ALL pairs.
    # My max-flow check only verified connectivity of the NEW vertex to OLD ones.
    # It did NOT verify that OLD branches are connected to each other.
    # But if they were added previously, they should have been connected.
    # Wait, when I added branch B[1], I connected it to B[0].
    # When I added B[2], I connected it to B[0] and B[1].
    # So all pairs involving the new vertex are connected.
    # The pairs among old vertices were connected when they were added.
    # So the set of paths `all_paths` should contain paths for all pairs?
    # No, `all_paths` accumulates paths.
    # When B[1] was added, path (B[0], B[1]) was added.
    # When B[2] was added, paths (B[0], B[2]) and (B[1], B[2]) were added.
    # So yes, `all_paths` contains paths for all pairs.
    
    # However, the interiors must be disjoint.
    # My update `U.add(node)` ensures that new paths don't use old interior nodes.
    # And the max-flow check ensured new paths are disjoint from each other.
    # So the total set of paths is valid.
    
    branches = [int(x) for x in B]
    paths = [list(map(int, p)) for p in all_paths]
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
