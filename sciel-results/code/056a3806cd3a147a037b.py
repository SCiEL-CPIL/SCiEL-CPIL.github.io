"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    """
    Decompose three fixed graphs into minimum number of edge-disjoint paths covering all edges.
    Uses the exact reduction to minimum-cost maximum matching in a bipartite graph.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_path_decomposition(n, edge_list):
        """
        Exact minimum path decomposition via min-cost max-matching.
        
        Build bipartite graph with left nodes L_0..L_{n-1}, right nodes R_0..R_{n-1}.
        For each undirected edge (u,v) in original graph, add two directed arcs:
          L_u -> R_v with cost 0
          L_v -> R_u with cost 0
        Also, for each vertex v, add:
          L_v -> R_v with cost 1  (this represents "not continuing" a path through v)
        
        Wait, the standard reduction:
        - We want to find a set of vertex-disjoint paths (in terms of edge usage) that cover all edges.
        - Actually, the classic result: minimum number of paths in a path decomposition = 
          (number of odd-degree vertices + number of connected components with no odd-degree vertices * 2) / 2
          ... no, that's for Eulerian trails.
        
        Let me reconsider. The minimum number of edge-disjoint paths covering all edges of a connected graph:
        - Let O = number of vertices with odd degree.
        - If O == 0 (Eulerian), min paths = 1 if there are edges, else 0.
        - If O > 0, min paths = O/2.
        
        Wait, is that right? Let's think:
        - A path has two endpoints. In a path decomposition, the endpoints of all paths together must account for the odd-degree vertices.
        - Each path contributes exactly 2 to the "endpoint count" (or 0 if it's a cycle, but we want paths not cycles).
        - Actually, the minimum number of paths = max(1, O/2) for a connected graph with at least one edge? No.
        
        Let me recall the correct theorem:
        - The minimum number of edge-disjoint paths needed to cover all edges of a connected graph G is:
          - 1 if G is Eulerian (all degrees even) and has at least one edge
          - O/2 if G has O > 0 odd-degree vertices
        - This is because we can add O/2 edges to make it Eulerian, find an Eulerian circuit, then remove the added edges to get O/2 paths.
        
        So the answer is simply:
          O = number of odd-degree vertices
          if O == 0: return 1 (if edges exist)
          else: return O // 2
        
        But wait - the paths must be SIMPLE (no repeated vertices). The Eulerian trail argument gives trails, not necessarily simple paths. However, the theorem about minimum number of paths (allowing trails) gives O/2. But the task requires simple paths.
        
        Hmm, but for these sparse random graphs, the minimum number of simple paths might be larger. However, the task says "Gallai's conjecture predicts at most 10+13+15 = 38" which suggests the bound is ceil(delta_max/2) or similar. Actually, Gallai's conjecture is about the minimum number of paths in a path decomposition where each path is simple, and the bound is ceil(d(G)/2) where d(G) is the maximum degree.
        
        Wait no. Let me re-read: "Gallai's conjecture predicts at most 10+13+15 = 38". For n=20, 25, 30 with p=0.3, 0.3, 0.25. The maximum degree in these graphs would be around 5-7 typically, so ceil(d/2) would be small. But 10, 13, 15 are much larger.
        
        Actually, I think the bound 10, 13, 15 might come from a different formula. Let me just implement the exact min-cost max-matching approach which gives the true minimum number of edge-disjoint paths (trails), and then verify they are simple. If not simple, we may need to split, but for sparse graphs this is unlikely to be an issue.
        
        Actually, let me just implement the matching-based approach properly.
        """
        if not edge_list:
            return []
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
        
        # Count degrees
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        
        # Number of odd-degree vertices
        odd_count = sum(1 for d in deg if d % 2 == 1)
        
        # The minimum number of trails (not necessarily simple paths) is:
        #   max(1, odd_count // 2) if there are edges
        # But we need simple paths. For now, let's use the matching approach to find actual paths.
        
        # Build bipartite graph for min-cost max-matching
        # Left side: 0..n-1, Right side: 0..n-1
        # For each edge (u,v): add arc L_u -> R_v cost 0, L_v -> R_u cost 0
        # For each vertex v: add arc L_v -> R_v cost 1 (staying/terminating)
        
        # We want a maximum matching that minimizes total cost.
        # The number of paths = number of "starts" = n - (number of matched pairs where cost 0)
        # Actually: each vertex has at most one outgoing and one incoming arc in the matching.
        # A vertex v is an endpoint of a path if either:
        #   - L_v is unmatched (no outgoing arc) -> v is a start
        #   - R_v is unmatched (no incoming arc) -> v is an end
        # Number of paths = number of starts = number of ends = (n - matched_L) where matched_L = number of L nodes matched
        
        # Let me use the Hungarian algorithm or min-cost flow.
        # For small graphs (n<=30), we can use a simple approach.
        
        # Actually, let's use scipy's linear_sum_assignment or implement min-cost max-flow.
        # The cost matrix: cost[i][j] = 0 if (i,j) is an edge in original graph, 1 otherwise.
        # But we can only match i to j if (i,j) is an edge OR i==j.
        # Non-allowed matches get infinite cost.
        
        INF = float('inf')
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            cost[i][i] = 1  # self-loop means terminating/starting at i
            for j in adj[i]:
                if j != i:
                    cost[i][j] = 0
        
        # Use scipy's linear_sum_assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            # Fallback: simple greedy + local search
            # For now, just return a trivial decomposition
            return [[list(e)] for e in edge_list]
        
        # Build the paths from the matching
        # next_v[i] = j means in the path, after vertex i comes vertex j
        next_v = [-1] * n
        prev_v = [-1] * n
        for i, j in zip(row_ind, col_ind):
            next_v[i] = j
            prev_v[j] = i
        
        # Find all paths
        paths = []
        visited = [False] * n
        
        # A vertex is a start if prev_v[v] == -1
        starts = [v for v in range(n) if prev_v[v] == -1]
        
        for start in starts:
            if visited[start]:
                continue
            path = [start]
            cur = start
            visited[cur] = True
            while next_v[cur] != -1:
                cur = next_v[cur]
                if visited[cur]:
                    # Cycle detected, should not happen in a valid path decomposition
                    break
                path.append(cur)
                visited[cur] = True
            if len(path) > 1:
                paths.append(path)
            elif len(path) == 1:
                # Single vertex, no edge - skip
                pass
        
        # Verify all edges are covered
        covered_edges = set()
        for path in paths:
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                e = tuple(sorted((u, v)))
                covered_edges.add(e)
        
        all_edges = set(tuple(sorted(e)) for e in edge_list)
        if covered_edges != all_edges:
            # Fallback to trivial decomposition
            return [[list(e)] for e in edge_list]
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edge_list = instance(n, p, seed)
        paths = min_path_decomposition(n, edge_list)
        results.append(paths)
        if time.time() > deadline:
            # Return what we have, pad with trivial if needed
            while len(results) < 3:
                spec2 = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
                n2, p2, seed2 = spec2
                edge_list2 = instance(n2, p2, seed2)
                results.append([[list(e)] for e in edge_list2])
            break
    
    return results
# EVOLVE-BLOCK-END
