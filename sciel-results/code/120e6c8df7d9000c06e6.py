import time
import random
import sys
from collections import defaultdict
import heapq

# Increase recursion limit just in case, though we use iterative methods
sys.setrecursionlimit(2000)

_DEADLINE_SECONDS = 145.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_flow(n_nodes, edges, source, sink):
    """
    Computes maximum flow using Dinic's algorithm.
    n_nodes: number of nodes
    edges: list of (u, v, capacity)
    """
    # Build adjacency list with edge indices
    graph = [[] for _ in range(n_nodes)]
    edge_list = []
    
    def add_edge(u, v, cap):
        edge_list.append([u, v, cap])
        graph[u].append(len(edge_list) - 1)
        edge_list.append([v, u, 0])
        graph[v].append(len(edge_list) - 1)

    for u, v, cap in edges:
        add_edge(u, v, cap)
        
    flow = 0
    INF = 10**9
    
    while True:
        # BFS to find augmenting paths
        level = [-1] * n_nodes
        queue = [source]
        level[source] = 0
        for u in queue:
            for i in graph[u]:
                v = edge_list[i][1]
                if level[v] < 0 and edge_list[i][2] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
                    
        if level[sink] < 0:
            break
            
        # DFS to find blocking flow
        iter_idx = [0] * n_nodes
        
        def dfs(u, f):
            if u == sink:
                return f
            for i in range(iter_idx[u], len(graph[u])):
                iter_idx[u] = i
                e_idx = graph[u][i]
                v = edge_list[e_idx][1]
                if edge_list[e_idx][2] > 0 and level[u] < level[v]:
                    d = dfs(v, min(f, edge_list[e_idx][2]))
                    if d > 0:
                        edge_list[e_idx][2] -= d
                        edge_list[e_idx ^ 1][2] += d
                        return d
            return 0
            
        while True:
            f = dfs(source, INF)
            if f == 0:
                break
            flow += f
            
    return flow, edge_list, graph

def _decompose_graph(edges, n, time_limit):
    if not edges:
        return []
        
    start_time = time.time()
    
    # 1. Construct Bipartite Double Cover for Max 2-Matching
    # Nodes:
    # 0 to n-1: v_out
    # n to 2n-1: v_in
    # 2n: source
    # 2n+1: sink
    
    N = 2 * n + 2
    source = 2 * n
    sink = 2 * n + 1
    
    flow_edges = []
    
    # Source to v_out (cap 2)
    for v in range(n):
        flow_edges.append((source, v, 2))
        
    # v_in to Sink (cap 2)
    for v in range(n):
        flow_edges.append((n + v, sink, 2))
        
    # v_out to w_in for each edge (v, w)
    # Edge (u, v) in original graph corresponds to u_out -> v_in and v_out -> u_in
    for u, v in edges:
        flow_edges.append((u, n + v, 1))
        flow_edges.append((v, n + u, 1))
        
    # Run Max Flow
    flow_val, edge_list, graph = _max_flow(N, flow_edges, source, sink)
    
    # 2. Extract the 2-matching
    # An edge (u, v) is in the 2-matching if flow on u_out -> v_in is 1 OR flow on v_out -> u_in is 1?
    # Actually, in the flow network, if flow is 1 on u_out -> v_in, it means edge (u,v) is selected and "entered" v from u.
    # Since the graph is undirected, the selection is symmetric in the 2-matching sense.
    # If flow(u_out, v_in) == 1, then (u,v) is in the matching.
    # Note: It is possible that flow(u_out, v_in) == 1 and flow(v_out, u_in) == 1?
    # No, because capacity is 1. And in a simple graph, we only add one direction for the flow?
    # Wait, I added both:
    # flow_edges.append((u, n + v, 1))
    # flow_edges.append((v, n + u, 1))
    # These are distinct edges in the flow network.
    # If both have flow 1, that would mean u sends to v AND v sends to u.
    # This would imply degree 2 for both?
    # In a 2-matching, an edge is either in or out.
    # If (u,v) is in the matching, it consumes 1 unit of capacity from u_out and 1 from v_in.
    # It also consumes 1 from v_out and 1 from u_in?
    # No, the edge (u,v) is just an edge.
    # In the double cover, selecting edge (u,v) in the 2-matching corresponds to sending flow 1 along u_out -> v_in.
    # Does it also require flow on v_out -> u_in?
    # No. The 2-matching is an undirected subgraph.
    # The flow formulation:
    # x_e = 1 if edge e is in the 2-matching.
    # Constraints: sum_{e incident to v} x_e <= 2.
    # This is a b-matching problem.
    # The reduction to bipartite matching for b-matching:
    # For each vertex v, split into v_in and v_out.
    # For each edge (u,v), add edge u_out -> v_in with cap 1.
    # Source -> v_out with cap b(v).
    # v_in -> Sink with cap b(v).
    # Max flow gives the b-matching.
    # The set of edges with flow 1 in the middle layer form the matching.
    
    matching_edges = set()
    for i, (u, v, cap) in enumerate(flow_edges):
        # Check if this is a middle edge (u_out -> v_in)
        # u is in [0, n), v is in [n, 2n)
        if 0 <= u < n and n <= v < 2*n:
            if edge_list[i][2] == 0: # Original cap was 1, now 0 means flow 1
                orig_u = u
                orig_v = v - n
                matching_edges.add(tuple(sorted((orig_u, orig_v))))
                
    # 3. Find cycles in the matching
    # Build adjacency list for the matching
    match_adj = defaultdict(list)
    for u, v in matching_edges:
        match_adj[u].append(v)
        match_adj[v].append(u)
        
    visited = [False] * n
    cycles = []
    
    for i in range(n):
        if not visited[i] and match_adj[i]:
            # Try to find a cycle starting from i
            # Since it's a 2-matching, components are paths or cycles.
            # If it's a cycle, all nodes in the component have degree 2.
            # If it's a path, endpoints have degree 1.
            
            # Check if component is a cycle
            # Do a DFS/BFS to find the component
            component = []
            stack = [i]
            visited[i] = True
            is_cycle = True
            
            while stack:
                node = stack.pop()
                component.append(node)
                for nb in match_adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
                    elif nb != i: # Simple check, not robust for complex graphs but ok for 2-matching
                        pass
                        
            # In a 2-matching, if all nodes in component have degree 2, it's a cycle.
            # Otherwise it's a path.
            all_deg_2 = True
            for node in component:
                if len(match_adj[node]) != 2:
                    all_deg_2 = False
                    break
                    
            if all_deg_2:
                # Reconstruct cycle order
                # Start at i, follow neighbors
                cycle = [i]
                prev = -1
                curr = i
                while True:
                    nexts = match_adj[curr]
                    next_node = nexts[0] if nexts[0] != prev else nexts[1]
                    if next_node == i:
                        break
                    cycle.append(next_node)
                    prev = curr
                    curr = next_node
                cycles.append(cycle)
            else:
                # It's a path, edges are single pieces
                pass

    # 4. Construct final pieces
    pieces = []
    for c in cycles:
        pieces.append(c)
        
    # Add remaining edges as single edges
    for u, v in edges:
        if (u, v) not in matching_edges:
            pieces.append([u, v])
            
    return pieces

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Allocate time for this graph
        # Max flow is very fast for N=30-60 nodes.
        # We can use most of the remaining time if we want to add local search improvements later,
        # but for now, the max flow is the main step.
        graph_time = remaining * 0.95
        
        pieces = _decompose_graph(edges, n, graph_time)
        
        results.append(pieces)

    return results
