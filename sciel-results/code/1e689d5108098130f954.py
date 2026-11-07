import itertools
import sys
from collections import deque
import networkx as nx # Wait, networkx is not guaranteed? "Python standard library, NumPy and SciPy are available". No NetworkX.
# I must implement max-flow myself or use a simpler heuristic.
# Implementing max-flow (Dinic or Edmonds-Karp) is doable but tedious.
# Alternative: Since N=24, and we are checking disjoint paths for a small number of targets (e.g., 5-8), 
# we can use a backtracking search for the specific paths, similar to the seed but smarter.

# However, the seed already did backtracking and scored 2.33. 
# The issue might be that it didn't find the optimal set.
# Let's try to find the Maximum Clique first. That gives a guaranteed baseline.
# Then try to expand it.

# Let's write a simple Max-Flow (Edmonds-Karp) implementation as it's short enough.
# Or, simpler: Use the fact that for small k, we can just do BFS for each path in a specific order? 
# No, order matters.

# Let's stick to the Max-Flow approach for the "can we add vertex" check.
# It is more robust than greedy path picking.

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Helper: Find Max Clique using Bron-Kerbosch
    def bron_kerbosch(R, P, X):
        if not P and not X:
            yield R
        # Pivot selection: choose vertex in P union X with max degree in P
        pivot = None
        max_deg = -1
        for v in P | X:
            deg = len(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is not None:
            candidates = P - adj[pivot]
        else:
            candidates = P
            
        for v in candidates:
            yield from bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    # Find all maximal cliques
    all_cliques = list(bron_kerbosch(set(), set(range(n)), set()))
    
    # Sort cliques by size descending
    all_cliques.sort(key=lambda c: len(c), reverse=True)
    
    best_branches = None
    best_paths = None
    best_size = 0

    # Function to check if we can add vertex 'new_v' to branch set 'branches'
    # given existing 'paths' (list of paths).
    # We need to find |branches| internally disjoint paths from new_v to each b in branches.
    # These paths must avoid the interiors of existing paths.
    # We use Max Flow.
    
    def can_add_vertex(new_v, branches, existing_paths):
        # Identify blocked vertices: interiors of existing paths
        blocked = set()
        for path in existing_paths:
            # path is a list [u, ..., v]. Interiors are path[1:-1]
            for u in path[1:-1]:
                blocked.add(u)
        
        # Also, we cannot pass through other branch vertices except the target.
        # But in the flow network, we treat branch vertices specially.
        
        # Build Flow Network
        # Nodes:
        # Source: new_v
        # Sinks: each b in branches
        # To handle vertex capacities, split each vertex v into v_in, v_out.
        # Edge v_in -> v_out with capacity 1 (if v not blocked and v != new_v and v not in branches? 
        # Wait, if v is in branches, it is a sink. We don't need to pass *through* it to get to another sink?
        # Actually, paths are from new_v to b. They end at b.
        # So for any vertex x:
        # if x == new_v: capacity INF (or |branches|)
        # if x in branches: capacity 1 (it's a sink, but we connect to super-sink)
        # else: capacity 1 if x not in blocked, else 0.
        
        # Let's map graph vertices to flow nodes.
        # Flow node ID: (vertex, 0) for in, (vertex, 1) for out.
        # Total nodes: 2*n + 2 (source, sink)
        
        # Simplification: Since n=24, we can just use a dict for the flow graph.
        
        # Let's implement a simple BFS-based Max Flow (Edmonds-Karp)
        
        # Nodes in flow graph:
        # 0: Source (new_v)
        # 1: Sink (Super Sink)
        # 2..2n-1: Vertex splits. 
        # Let's use 2*v for in, 2*v+1 for out.
        
        N_flow = 2 * n + 2
        source = 2 * n
        sink = 2 * n + 1
        
        # Adjacency list for flow graph: {u: {v: capacity}}
        flow_adj = {i: {} for i in range(N_flow)}
        
        # Add edges
        # 1. Split vertices
        for v in range(n):
            in_node = 2 * v
            out_node = 2 * v + 1
            
            if v == new_v:
                cap = len(branches) # Can send all flow out
            elif v in blocked:
                cap = 0
            else:
                cap = 1
            
            # Add edge in -> out
            if cap > 0:
                flow_adj[in_node][out_node] = cap
                flow_adj[out_node][in_node] = 0 # Reverse edge initially 0
                
        # 2. Graph edges
        for u in range(n):
            for v in adj[u]:
                if u == v: continue
                u_out = 2 * u + 1
                v_in = 2 * v
                # Capacity INF (or large)
                flow_adj[u_out][v_in] = 100
                flow_adj[v_in][u_out] = 0
                
        # 3. Connect Source to new_v
        new_v_in = 2 * new_v
        # Wait, source is external. Connect source to new_v_in? 
        # Usually source connects to the 'in' or 'out'? 
        # Flow starts at new_v. It goes out. 
        # Let's connect source to new_v_in? No, new_v_in -> new_v_out is the capacity edge.
        # If we connect source to new_v_in, flow goes in->out. 
        # But new_v is the start. 
        # Let's connect source to new_v_out directly? 
        # If we connect source to new_v_in, and cap(in->out) is len(branches), it works.
        flow_adj[source][new_v_in] = len(branches)
        flow_adj[new_v_in][source] = 0
        
        # 4. Connect Branch vertices to Sink
        for b in branches:
            b_in = 2 * b
            # Flow enters b_in, then goes to sink? 
            # We need to terminate at b. 
            # So connect b_in to sink? Or b_out to sink?
            # If flow goes u_out -> b_in -> b_out -> ... it passes through.
            # We want to stop at b.
            # So connect b_in to sink.
            # But we also have edge b_in -> b_out.
            # If we connect b_in to sink, flow can exit there.
            flow_adj[b_in][sink] = 1
            flow_adj[sink][b_in] = 0
            
        # Now run Max Flow
        # Edmonds-Karp
        def bfs_flow():
            # Returns augmenting path or None
            parent = [-1] * N_flow
            parent[source] = -2
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, cap in flow_adj[u].items():
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == sink:
                            return parent
                        queue.append(v)
            return None
            
        total_flow = 0
        while True:
            parent = bfs_flow()
            if parent is None:
                break
            # Find min capacity on path
            min_cap = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                min_cap = min(min_cap, flow_adj[u][v])
                v = u
            # Augment
            v = sink
            while v != source:
                u = parent[v]
                flow_adj[u][v] -= min_cap
                flow_adj[v][u] += min_cap
                v = u
            total_flow += min_cap
            
        if total_flow == len(branches):
            # Extract paths
            # We need to trace back the flow from sink to source for each unit of flow?
            # This is tricky with the aggregated sink.
            # Alternative: Since we know a solution exists, we can find the paths by
            # looking at the flow on the edges.
            # For each branch b, there is 1 unit of flow entering b_in from some neighbor.
            # We can reconstruct by tracing back from b_in to source.
            
            paths = {}
            # We need to know which edge carried the flow to b_in.
            # The flow into b_in comes from some u_out.
            # Let's find the predecessor of b_in in the flow graph that has positive flow.
            
            for b in branches:
                b_in = 2 * b
                # Find u such that flow(u_out, b_in) > 0
                pred = None
                for u in range(n):
                    u_out = 2 * u + 1
                    # Check if flow_adj[u_out][b_in] is part of the residual? 
                    # We modified flow_adj. 
                    # The actual flow on edge (u_out, b_in) is original_cap - residual_cap.
                    # Original cap was 100.
                    # So if flow_adj[u_out][b_in] < 100, there is flow.
                    # But there might be multiple? No, capacity of b_in->sink is 1.
                    # So only 1 unit enters b_in.
                    if u_out in flow_adj and b_in in flow_adj[u_out]:
                         # Residual is flow_adj[u_out][b_in]
                         # If residual < 100, flow > 0.
                         # Actually, we can just check if the reverse edge flow_adj[b_in][u_out] > 0
                         # Because flow(u->v) = reverse_edge_flow(v->u)
                         if flow_adj[b_in][u_out] > 0:
                             pred = u
                             break
                if pred is None:
                    # Direct edge from source? No, source connects to new_v_in.
                    # If new_v == b, path is [new_v, b] which is length 1? 
                    # But new_v is not in branches initially.
                    # If new_v is adjacent to b, pred might be new_v.
                    # If new_v == b, impossible since new_v not in branches.
                    return None
                # Trace back from pred
                path = [b, pred]
                curr = pred
                while curr != new_v:
                    curr_out = 2 * curr + 1
                    # Find predecessor of curr_in? 
                    # We are at curr. We came from some x such that x_out -> curr_in had flow.
                    # So we need to find x where flow_adj[curr_in][x_out] > 0 (reverse flow)
                    curr_in = 2 * curr
                    next_pred = None
                    for x in range(n):
                        x_out = 2 * x + 1
                        if flow_adj[curr_in][x_out] > 0:
                            next_pred = x
                            break
                    if next_pred is None:
                        # Maybe we reached new_v?
                        # If curr is new_v, loop ends.
                        # If not, error.
                        return None
                    path.append(next_pred)
                    curr = next_pred
                # path is [b, ..., new_v]
                path.reverse()
                # path should be [new_v, ..., b]
                paths[b] = path
                
            return paths
        return None

    # Process each clique
    for clique in all_cliques:
        if len(clique) <= best_size:
            continue
            
        current_branches = list(clique)
        # Initialize paths for the clique
        current_paths = []
        for u, v in itertools.combinations(current_branches, 2):
            current_paths.append([u, v])
            
        # Try to expand
        expanded = True
        while expanded:
            expanded = False
            # Try adding each vertex not in current_branches
            candidates = [v for v in range(n) if v not in current_branches]
            # Sort candidates by degree to current branches?
            candidates.sort(key=lambda v: sum(1 for b in current_branches if v in adj[b]), reverse=True)
            
            for v in candidates:
                new_paths = can_add_vertex(v, current_branches, current_paths)
                if new_paths:
                    # Add v to branches
                    current_branches.append(v)
                    # Add new paths to current_paths
                    current_paths.extend(new_paths.values())
                    expanded = True
                    break # Restart candidate loop with new set
        
        if len(current_branches) > best_size:
            best_size = len(current_branches)
            best_branches = current_branches
            best_paths = current_paths

    # If no clique found (shouldn't happen), fallback
    if best_branches is None:
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    return {'branches': best_branches, 'paths': best_paths}

# EVOLVE-BLOCK-END
