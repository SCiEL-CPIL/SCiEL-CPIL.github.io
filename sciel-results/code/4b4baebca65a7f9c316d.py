"""Difference triangle set (7,5) solver using min-cost max-flow assignment."""
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    
    Uses min-cost max-flow to assign difference values to rows,
    then reconstructs valid rows from the assigned differences.
    """
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # For (7,5): 105 total differences, 15 per row
    # Best known: 112, counting bound: 105
    
    def min_cost_max_flow(n_nodes, edges, source, sink, required_flow):
        """
        Min-cost max-flow using successive shortest paths with potentials.
        edges: list of (u, v, capacity, cost)
        Returns (flow, cost) or (None, None) if required_flow not achievable.
        """
        # Build adjacency list
        adj = [[] for _ in range(n_nodes)]
        for idx, (u, v, cap, cost) in enumerate(edges):
            adj[u].append([v, cap, cost, idx])  # to, cap, cost, edge_idx
            adj[v].append([u, 0, -cost, idx])    # reverse edge
        
        # Edge data for updating
        edge_data = []
        for (u, v, cap, cost) in edges:
            edge_data.append([cap, cost])
        
        def bellman_ford_or_dijkstra(n_nodes, adj, source, potentials):
            """Find shortest paths using Dijkstra with potentials."""
            dist = [float('inf')] * n_nodes
            parent_node = [-1] * n_nodes
            parent_edge = [-1] * n_nodes
            dist[source] = 0
            visited = [False] * n_nodes
            
            # Simple O(V^2) Dijkstra (fine for small graphs)
            for _ in range(n_nodes):
                u = -1
                min_dist = float('inf')
                for i in range(n_nodes):
                    if not visited[i] and dist[i] < min_dist:
                        min_dist = dist[i]
                        u = i
                if u == -1:
                    break
                visited[u] = True
                for v, cap, cost, edge_idx in adj[u]:
                    if cap > 0 and dist[v] > dist[u] + cost + potentials[u] - potentials[v]:
                        dist[v] = dist[u] + cost + potentials[u] - potentials[v]
                        parent_node[v] = u
                        parent_edge[v] = edge_idx
            
            # Update potentials
            for i in range(n_nodes):
                if dist[i] < float('inf'):
                    potentials[i] += dist[i]
            
            return dist, parent_node, parent_edge
        
        flow = 0
        cost = 0
        potentials = [0] * n_nodes
        
        while flow < required_flow:
            dist, parent_node, parent_edge = bellman_ford_or_dijkstra(
                n_nodes, adj, source, potentials
            )
            if dist[sink] >= float('inf'):
                break  # No more augmenting paths
            
            # Find bottleneck
            path_flow = required_flow - flow
            v = sink
            while v != source:
                u = parent_node[v]
                edge_idx = parent_edge[v]
                # Get the forward edge capacity
                for entry in adj[u]:
                    if entry[0] == v and entry[3] == edge_idx:
                        path_flow = min(path_flow, entry[1])
                        break
                v = u
            
            # Augment
            v = sink
            while v != source:
                u = parent_node[v]
                edge_idx = parent_edge[v]
                # Update forward edge
                for i, entry in enumerate(adj[u]):
                    if entry[0] == v and entry[3] == edge_idx:
                        adj[u][i][1] -= path_flow
                        break
                # Update reverse edge
                for i, entry in enumerate(adj[v]):
                    if entry[0] == u and entry[3] == edge_idx:
                        adj[v][i][1] += path_flow
                        break
                v = u
            
            flow += path_flow
            # Calculate actual cost (using original costs, not reduced)
            v = sink
            while v != source:
                u = parent_node[v]
                edge_idx = parent_edge[v]
                for entry in adj[u]:
                    if entry[0] == v and entry[3] == edge_idx:
                        cost += path_flow * entry[2]
                        break
                v = u
        
        if flow >= required_flow:
            return flow, cost
        return None, None
    
    def build_flow_network(M, n_rows, k):
        """
        Build min-cost max-flow network.
        
        Nodes:
        - Source: 0
        - Row nodes: 1 to n_rows (row i is node i)
        - Difference value nodes: n_rows+1 to n_rows+M (value v is node n_rows+v)
        - Sink: n_rows+M+1
        
        Edges:
        - Source -> row_i: capacity 15 (n_diffs_per_row), cost 0
        - row_i -> diff_v: capacity 1, cost = penalty(v, i)
        - diff_v -> sink: capacity 1, cost 0
        
        The penalty encourages using larger differences for later rows
        to preserve small differences for earlier rows.
        """
        n_nodes = 1 + n_rows + M + 1  # source + rows + diffs + sink
        source = 0
        sink = n_nodes - 1
        
        edges = []
        
        # Source to rows
        for i in range(n_rows):
            row_node = 1 + i
            edges.append((source, row_node, n_diffs_per_row, 0))
        
        # Rows to difference values
        for i in range(n_rows):
            row_node = 1 + i
            for v in range(1, M + 1):
                diff_node = 1 + n_rows + v
                # Cost: penalize small values more for later rows
                # This encourages spreading out the assignments
                # Use a convex cost to encourage balance
                cost = v * (i + 1)  # later rows pay more for small diffs
                edges.append((row_node, diff_node, 1, cost))
        
        # Difference values to sink
        for v in range(1, M + 1):
            diff_node = 1 + n_rows + v
            edges.append((diff_node, sink, 1, 0))
        
        return n_nodes, edges, source, sink
    
    def reconstruct_row_from_diffs(diffs, k):
        """
        Given a set of 15 differences for k=5, reconstruct the row.
        The row has 6 elements: 0 = a0 < a1 < a2 < a3 < a4 < a5
        Differences are {a_j - a_i : i < j}
        
        We need to find a valid ordering. This is a constraint satisfaction problem.
        """
        diffs = sorted(diffs)
        if len(diffs) != k * (k + 1) // 2:
            return None
        
        # For k=5, we have 15 differences from 6 elements
        # The largest difference is a5 - a0 = a5
        # Let's use backtracking to find the row
        
        def backtrack(row, used_local):
            if len(row) == k + 1:
                return row[:]
            
            last = row[-1]
            # Try next values
            # The next value x must create new differences x - a_i for all a_i in row
            # These must all be in the diffs set and not in used_local
            
            # Find candidate x values
            # x - a_i must be in diffs for all i
            # So x = a_i + d for some d in diffs
            candidates = set()
            for a in row:
                for d in diffs:
                    x = a + d
                    if x > last:
                        candidates.add(x)
            
            for x in sorted(candidates):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d not in diffs or d in used_local:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check that all new_diffs are actually in the remaining diffs
                remaining = set(diffs) - used_local
                if not new_diffs.issubset(remaining):
                    continue
                
                row.append(x)
                result = backtrack(row, used_local | new_diffs)
                if result is not None:
                    return result
                row.pop()
            
            return None
        
        return backtrack([0], set())
    
    # Try increasing M values
    best_result = None
    best_scope = float('inf')
    
    # Start from counting bound
    start_M = total_diffs  # 105
    max_M = 150
    
    for M in range(start_M, max_M + 1):
        n_nodes, edges, source, sink = build_flow_network(M, n_rows, k)
        
        flow, cost = min_cost_max_flow(n_nodes, edges, source, sink, total_diffs)
        
        if flow is None or flow < total_diffs:
            continue
        
        # Extract which differences were assigned to which row
        # We need to trace back through the flow
        # Rebuild the network to extract flow
        adj = [[] for _ in range(n_nodes)]
        for idx, (u, v, cap, cost_e) in enumerate(edges):
            adj[u].append([v, cap, cost_e, idx])
            adj[v].append([u, 0, -cost_e, idx])
        
        # Run the flow again to get the actual assignments
        # Actually, let's just run it fresh and track
        
        # Simpler: run min-cost flow and track assignments
        adj2 = [[] for _ in range(n_nodes)]
        for idx, (u, v, cap, cost_e) in enumerate(edges):
            adj2[u].append([v, cap, cost_e, idx])
            adj2[v].append([u, 0, -cost_e, idx])
        
        # Run successive shortest paths and track
        potentials = [0] * n_nodes
        flow2 = 0
        assignments = {i: [] for i in range(n_rows)}  # row -> list of diffs
        
        while flow2 < total_diffs:
            # Dijkstra
            dist = [float('inf')] * n_nodes
            parent_node = [-1] * n_nodes
            parent_edge = [-1] * n_nodes
            dist[source] = 0
            visited = [False] * n_nodes
            
            for _ in range(n_nodes):
                u = -1
                min_d = float('inf')
                for i in range(n_nodes):
                    if not visited[i] and dist[i] < min_d:
                        min_d = dist[i]
                        u = i
                if u == -1:
                    break
                visited[u] = True
                for v, cap, c, ei in adj2[u]:
                    if cap > 0 and dist[v] > dist[u] + c + potentials[u] - potentials[v]:
                        dist[v] = dist[u] + c + potentials[u] - potentials[v]
                        parent_node[v] = u
                        parent_edge[v] = ei
            
            if dist[sink] >= float('inf'):
                break
            
            for i in range(n_nodes):
                if dist[i] < float('inf'):
                    potentials[i] += dist[i]
            
            # Find path and augment
            path_flow = total_diffs - flow2
            v = sink
            while v != source:
                u = parent_node[v]
                ei = parent_edge[v]
                for entry in adj2[u]:
                    if entry[0] == v and entry[3] == ei:
                        path_flow = min(path_flow, entry[1])
                        break
                v = u
            
            v = sink
            while v != source:
                u = parent_node[v]
                ei = parent_edge[v]
                for i, entry in enumerate(adj2[u]):
                    if entry[0] == v and entry[3] == ei:
                        adj2[u][i][1] -= path_flow
                        # If this is a row->diff edge, record assignment
                        if u >= 1 and u <= n_rows:
                            row_idx = u - 1
                            diff_val = v - (1 + n_rows)
                            if 1 <= diff_val <= M:
                                assignments[row_idx].append(diff_val)
                        break
                for i, entry in enumerate(adj2[v]):
                    if entry[0] == u and entry[3] == ei:
                        adj2[v][i][1] += path_flow
                        break
                v = u
            
            flow2 += path_flow
        
        if flow2 < total_diffs:
            continue
        
        # Reconstruct rows
        rows = []
        valid = True
        for i in range(n_rows):
            diffs = assignments[i]
            if len(diffs) != n_diffs_per_row:
                valid = False
                break
            row = reconstruct_row_from_diffs(set(diffs), k)
            if row is None:
                valid = False
                break
            rows.append(row)
        
        if not valid:
            continue
        
        scope = max(row[-1] for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_result = rows
            if scope == start_M:  # Can't do better than counting bound
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
