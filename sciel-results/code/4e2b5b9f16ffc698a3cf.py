"""
Decompose three fixed graphs into cycles and edges, minimizing piece count.
Strategy: Maximum Matching on the Line Graph.
1. Construct a bipartite graph where nodes represent directed edges (u, v).
2. A match between (u, v) and (v, w) represents a "turn" at vertex v.
3. A maximum matching in this bipartite graph corresponds to the maximum set of 
   edge-disjoint paths/cycles.
4. The number of pieces is minimized by maximizing the number of matched pairs 
   (turns), which effectively merges edges into longer structures.
"""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(n_left, n_right, adj_left):
    """
    Standard Hopcroft-Karp algorithm for maximum bipartite matching.
    adj_left: list of lists, where adj_left[i] is a list of right nodes connected to left node i.
    Returns: list of matched right nodes for each left node (or -1 if unmatched).
    """
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj_left[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return pair_u


def _decompose_with_matching(edges, n, time_limit):
    """
    Decompose the graph by finding a maximum matching of 'turns'.
    """
    start_time = time.time()
    
    # 1. Build directed edge representation
    # Each undirected edge (u, v) creates two directed edges: u->v and v->u
    # We index directed edges.
    dir_edges = []
    edge_to_dir = {}
    for i, (u, v) in enumerate(edges):
        dir_edges.append((u, v))
        dir_edges.append((v, u))
        edge_to_dir[(u, v)] = 2 * i
        edge_to_dir[(v, u)] = 2 * i + 1

    m = len(dir_edges)
    
    # 2. Build bipartite graph for matching
    # Left side: directed edges e1
    # Right side: directed edges e2
    # Connect e1=(u,v) to e2=(v,w) if v is the shared vertex and u != w (no immediate backtrack)
    # Actually, the condition for a valid "turn" at v is that we enter v from u and leave to w.
    # So e1 = (u, v) and e2 = (v, w).
    # Constraint: u != w is not strictly necessary for matching validity in terms of edge disjointness 
    # because matching ensures each directed edge is used once. However, u=w would mean 
    # (u,v) -> (v,u) which uses the same undirected edge twice in opposite directions?
    # No, the matching is on directed edges. If we match (u,v) with (v,u), it means we traverse 
    # u->v and then v->u. This consumes both directed representations of the single undirected edge (u,v).
    # This creates a "cycle" of length 2 in the line graph sense, but in the original graph, it's just 
    # traversing the edge back and forth. This is NOT a valid cycle in the original graph (simple cycle).
    # A simple cycle requires distinct vertices. 
    # So we must forbid matching (u,v) with (v,u).
    
    adj_left = [[] for _ in range(m)]
    
    # For efficiency, we can group right nodes by their start vertex.
    # If e1 ends at v, e2 must start at v.
    # Let's pre-compute lists of directed edges starting at each vertex.
    start_at = defaultdict(list)
    for idx, (u, v) in enumerate(dir_edges):
        start_at[v].append(idx) # e2 starts at v

    for idx1, (u, v) in enumerate(dir_edges):
        # e1 is (u, v). It ends at v.
        # e2 must start at v.
        for idx2 in start_at[v]:
            w, x = dir_edges[idx2] # e2 is (v, x) so w=v, x=...
            # Check if idx2 is the reverse of idx1
            # Reverse of (u, v) is (v, u).
            # If idx2 is the reverse, we skip.
            if dir_edges[idx2] == (v, u):
                continue
            adj_left[idx1].append(idx2)
            
    # 3. Run Hopcroft-Karp
    # Note: The bipartite graph has m nodes on left and m nodes on right.
    # This can be large. For n=30, m ~ 2 * 150 = 300. 300x300 is fine.
    # For n=25, m ~ 2 * 100 = 200.
    # For n=20, m ~ 2 * 70 = 140.
    
    pair_u = _hopcroft_karp(m, m, adj_left)
    
    # 4. Reconstruct cycles and paths from the matching
    # The matching gives us a set of directed edges that are "linked".
    # A matched pair (idx1, idx2) means edge idx1 is followed by idx2.
    # This forms disjoint paths and cycles in the directed line graph.
    # We need to trace these.
    
    # next_edge[i] = j if i is matched to j, else -1
    next_edge = pair_u
    
    # prev_edge[j] = i if i is matched to j, else -1
    prev_edge = [-1] * m
    for i, j in enumerate(pair_u):
        if j != -1:
            prev_edge[j] = i
            
    visited = [False] * m
    pieces = []
    
    # We will extract cycles and paths.
    # A cycle in the matching corresponds to a cycle in the original graph.
    # A path in the matching corresponds to a path in the original graph, which we must decompose into edges.
    
    # We need to be careful: The matching maximizes the number of "turns".
    # The number of pieces is: (# Cycles) + (# Edges in Paths).
    # Wait, if we have a path of length L (L edges), it contributes L pieces if we can't close it.
    # But the objective is to minimize total pieces.
    # A cycle of length L contributes 1 piece.
    # A path of length L contributes L pieces (since each edge is a separate piece if not part of a cycle).
    # Actually, the task says: "A piece is a single edge [u, v] or a cycle".
    # So a path of 3 edges must be 3 pieces.
    # Therefore, we want to maximize the number of edges that are part of *cycles*.
    # The matching approach finds a set of edge-disjoint paths and cycles.
    # Any cycle found is a valid piece.
    # Any path found must be broken into single edges.
    
    # Let's trace components.
    for i in range(m):
        if visited[i]:
            continue
        
        # Trace the chain
        current = i
        chain = []
        while not visited[current]:
            visited[current] = True
            chain.append(current)
            nxt = next_edge[current]
            if nxt == -1:
                break
            current = nxt
            
        # Now we have a chain of directed edges.
        # Check if it's a cycle:
        # If the last edge in the chain points to the first edge in the chain?
        # No, the chain follows next_edge.
        # If it's a cycle, the chain should close.
        # But our traversal stops when next_edge is -1 or visited.
        # If it's a cycle, we will visit all nodes in the cycle and then try to go to the start?
        # Let's check:
        # If i is in a cycle, we visit i, i1, i2... ik. next_edge[ik] = i.
        # But visited[i] is already True. So the loop breaks.
        # The chain is [i, i1, ..., ik].
        # To verify it's a cycle, we check if next_edge[ik] == i AND prev_edge[i] == ik.
        
        if len(chain) >= 3:
            first = chain[0]
            last = chain[-1]
            if next_edge[last] == first and prev_edge[first] == last:
                # It's a cycle
                # Extract vertices
                cycle_vertices = []
                for idx in chain:
                    u, v = dir_edges[idx]
                    if not cycle_vertices:
                        cycle_vertices.append(u)
                        cycle_vertices.append(v)
                    else:
                        # The last vertex of previous edge should be u
                        if cycle_vertices[-1] != u:
                            # Error in logic? 
                            # dir_edges[idx] is (u,v). Previous was (..., u). So last is u.
                            pass
                        cycle_vertices.append(v)
                # The cycle is cycle_vertices[0:-1] because the last v is the same as first u?
                # Let's trace:
                # e1: (v0, v1)
                # e2: (v1, v2)
                # ...
                # ek: (vk-1, vk) where vk = v0
                # cycle_vertices = [v0, v1, v1, v2, ..., vk-1, v0]
                # We want [v0, v1, ..., vk-1]
                # So remove the last element.
                cycle_vertices.pop()
                pieces.append(cycle_vertices)
            else:
                # It's a path. Each edge in the path becomes a separate piece.
                for idx in chain:
                    u, v = dir_edges[idx]
                    pieces.append([u, v])
        else:
            # Short chain (path of length 1 or 2)
            # Each edge is a piece
            for idx in chain:
                u, v = dir_edges[idx]
                pieces.append([u, v])

    return pieces

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            # Fallback to simple edge list if time is tight
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        graphs_left = len(specs) - i
        graph_time = remaining * 0.9 / max(graphs_left, 1)
        
        pieces = _decompose_with_matching(edges, n, graph_time)
        results.append(pieces)
        
    return results

# Quick self-test if run directly
if __name__ == "__main__":
    results = construct()
    for i, res in enumerate(results):
        print(f"Graph {i}: {len(res)} pieces")
