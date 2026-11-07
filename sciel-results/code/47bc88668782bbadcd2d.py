# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

from collections import deque
import random

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degrees for heuristic
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def max_flow_sparse(s, t, adj_flow, N):
        # adj_flow is a list of lists of [neighbor, capacity]
        # We need to handle residual graph updates carefully.
        # For simplicity and speed with small N, let's stick to dense matrix but optimize BFS.
        # Actually, let's use a dictionary for edges to allow sparse updates if needed, 
        # but dense is faster for lookup if N < 200.
        # Let's use a dense matrix but pre-compute neighbors? No, capacities change.
        # Let's use the standard Edmonds-Karp with dense matrix, but break early if possible.
        
        # To speed up, we can use Dinic's algorithm which is faster in practice for unit capacities.
        # Since most capacities are 1 or INF, Dinic is a good choice.
        
        # However, implementing Dinic correctly in a short time is risky.
        # Let's optimize Edmonds-Karp by only iterating over neighbors with cap > 0.
        # We can maintain a set of active neighbors for each node? 
        # Or just iterate all N nodes. N ~ 100. 100*100 = 10k ops per BFS.
        # Number of BFS calls in Edmonds-Karp is bounded by V*E. 
        # V=100, E~1000. 100k BFS calls? No, that's worst case.
        # In practice, it's much less.
        
        # Let's try to improve the search strategy first, as that might yield the biggest gain.
        # If the flow check is the bottleneck, we will see it.
        
        pass # Placeholder, we will replace the whole construct body below.

    def try_branch_set(B):
        """Try to find vertex-disjoint paths for all pairs in B. Returns paths dict or None."""
        k = len(B)
        Bset = set(B)
        num_pairs = k * (k - 1) // 2
        if num_pairs == 0:
            return {}
            
        pair_list = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_list.append((i, j))
        
        # Node indices:
        # 0..n-1: v_in
        # n..2n-1: v_out
        # 2n..2n+num_pairs-1: sources
        # 2n+num_pairs..2n+2*num_pairs-1: sinks
        # 2n+2*num_pairs: Super Source
        # 2n+2*num_pairs+1: Super Sink
        N = 2 * n + 2 * num_pairs + 2
        SS = N - 2
        TT = N - 1
        
        # Use a dictionary of dictionaries for sparse graph? 
        # Or a list of lists for neighbors, and a 2D array for capacities?
        # Let's use a 2D array for capacities, it's fast for small N.
        cap = [[0] * N for _ in range(N)]
        
        def add_edge(u, v, c):
            cap[u][v] += c
            # Residual edge is implicitly handled by cap[v][u] in Edmonds-Karp 
            # if we treat it as a directed graph with reverse edges.
            # Standard EK: cap[u][v] is forward capacity. 
            # When we augment, we do cap[u][v] -= f, cap[v][u] += f.
            # So we must initialize cap[v][u] = 0 for all non-existent edges.
            # This is already done by initialization.
            
        # 1. Vertex splitting
        for v in range(n):
            if v in Bset:
                add_edge(v, v + n, 10**9) # INF
            else:
                add_edge(v, v + n, 1)
        
        # 2. Graph edges
        for u, v in edges:
            # u -> v and v -> u in original graph
            # We need to add directed edges in the flow network
            
            # Case 1: u in B, v in B
            # If both are branches, they are connected via the super source/sink logic?
            # No, the path goes from src -> u_in -> u_out -> ... -> v_in -> v_out -> sink
            # The edge between u and v in the original graph should allow flow from u_out to v_in and v_out to u_in.
            # However, if u and v are both branches, do we allow direct connection?
            # The path must be simple and avoid other branches.
            # If u and v are the endpoints, the path is just u-v? 
            # If u and v are branches, and we need a path between them, it can be the direct edge u-v.
            # In the flow network:
            # src -> u_in (INF)
            # u_in -> u_out (INF)
            # u_out -> v_in (1)  <-- This edge exists if (u,v) in edges
            # v_in -> v_out (INF)
            # v_out -> sink (INF)
            # This works.
            
            if u in Bset and v in Bset:
                # Both branches
                add_edge(u + n, v, 1)
                add_edge(v + n, u, 1)
            elif u in Bset:
                # u branch, v non-branch
                # u_out -> v_in
                add_edge(u + n, v, 1)
                # v_out -> u_in ? No, flow goes towards the other branch.
                # If the path is u -> ... -> v (where v is non-branch, then to another branch w), 
                # flow comes from u_out to v_in.
                # Can flow come from v_out to u_in? Only if the path is w -> ... -> v -> u.
                # Since the graph is undirected, we need bidirectional edges for non-branch connections.
                # But for branch connections, we only need entry to non-branch and exit from non-branch.
                # Wait, if u is a branch, flow can only LEAVE u (via u_out) or ENTER u (via u_in from source).
                # It cannot enter u from a non-branch vertex because u_in only has incoming from Source and from non-branch vertices?
                # No, in the standard vertex splitting:
                # Edges are v_in -> v_out.
                # Original edge (u,v) becomes u_out -> v_in and v_out -> u_in.
                # If u is a branch, u_in has incoming from Super Source. 
                # Does u_in have incoming from other vertices? 
                # If we add v_out -> u_in for all neighbors v, then flow could enter u from a non-branch vertex.
                # This would mean the path ends at u. 
                # Since u is a branch, and it's an endpoint, this is correct!
                # So we MUST add v_out -> u_in for all neighbors v of branch u.
                
                add_edge(v + n, u, 1) # v_out -> u_in
            elif v in Bset:
                # v branch, u non-branch
                add_edge(u + n, v, 1) # u_out -> v_in
                add_edge(v + n, u, 1) # v_out -> u_in
            else:
                # Both non-branch
                add_edge(u + n, v, 1)
                add_edge(v + n, u, 1)
        
        # 3. Sources and Sinks
        for idx, (i, j) in enumerate(pair_list):
            b1, b2 = B[i], B[j]
            src = 2 * n + idx
            sink = 2 * n + num_pairs + idx
            
            add_edge(SS, src, 1)
            add_edge(src, b1, 10**9)
            add_edge(src, b2, 10**9)
            
            add_edge(b1 + n, sink, 10**9)
            add_edge(b2 + n, sink, 10**9)
            add_edge(sink, TT, 1)
            
        # Run Max Flow
        flow = 0
        while True:
            parent = [-1] * N
            parent[SS] = SS
            q = deque([SS])
            while q and parent[TT] == -1:
                u = q.popleft()
                # Optimize: only check neighbors with cap > 0
                for v in range(N):
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        q.append(v)
            if parent[TT] == -1:
                break
            # Find bottleneck
            path_flow = 10**9
            v = TT
            while v != SS:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            # Augment
            v = TT
            while v != SS:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            flow += path_flow
            
        if flow != num_pairs:
            return None
        
        # Extract paths
        paths = {}
        for idx, (i, j) in enumerate(pair_list):
            src = 2 * n + idx
            sink = 2 * n + num_pairs + idx
            b1, b2 = B[i], B[j]
            
            # Determine entry branch
            # Flow from SS to src to b1 or b2
            # Check residual: if cap[b1][src] > 0, flow went src->b1
            # Note: cap[src][b1] was INF. If flow 1 went through, cap[b1][src] becomes 1.
            if cap[b1][src] > 0:
                first, second = b1, b2
            else:
                first, second = b2, b1
                
            # Find path from first to second
            # We need to trace the flow from first to second.
            # The flow goes: first_in -> first_out -> ... -> second_in -> second_out -> sink
            # We can BFS from first_out to second_in in the residual graph?
            # No, we need to find the path that was used.
            # The edges used are those where cap[v][u] > 0 (reverse residual) for forward edges u->v?
            # Actually, simpler: 
            # We know the path starts at `first`.
            # We can do a BFS from `first` in the original graph, but constrained by the flow.
            # An edge (u, v) in the original graph is part of the path if the corresponding flow edge has positive flow.
            # Flow edge u_out -> v_in has flow if cap[v][u+n] > 0 (residual back edge).
            
            path = [first]
            visited = {first}
            q = deque([(first, [first])])
            found = False
            while q:
                u, p = q.popleft()
                if u == second:
                    path = p
                    found = True
                    break
                for v in adj[u]:
                    if v in visited or v in Bset:
                        continue
                    # Check if flow went from u to v
                    # Edge in flow network: u_out -> v_in
                    # Residual: cap[v][u+n] > 0
                    if cap[v][u + n] > 0:
                        visited.add(v)
                        q.append((v, p + [v]))
            if not found:
                # Fallback: maybe the path was direct?
                # If first and second are adjacent, the path is [first, second]
                if (first, second) in [(u,v) for u,v in edges] or (second, first) in [(u,v) for u,v in edges]:
                     path = [first, second]
                     found = True
            if not found:
                return None
            paths[(i, j)] = path
        
        return paths

    best_B = []
    best_paths = {}
    
    # Greedy construction with sorted vertices
    # Start with empty set
    # Try to add vertices in order of degree
    # To ensure we find a good set, we can try multiple orders or just the degree order.
    
    # Initialize with a single high-degree vertex? 
    # The problem requires at least 2 branches.
    # Let's try to build up from a pair.
    
    # Find a good starting pair
    # Try all pairs of high-degree vertices?
    # Or just start with the top 2 vertices.
    
    # Strategy:
    # 1. Sort vertices by degree desc.
    # 2. Try to add them one by one to the current set.
    # 3. If a vertex can't be added, skip it? Or try to swap?
    # Simple greedy: 
    # B = []
    # for v in sorted_vertices:
    #    if try(B + [v]): B = B + [v]
    
    # This might fail if the first vertex is isolated from the rest.
    # But high-degree vertices are likely connected.
    
    # Let's try to find a triangle first to ensure connectivity, then expand.
    # Or just start with the top 2.
    
    B = []
    for v in sorted_vertices:
        if not B:
            B.append(v)
            continue
        # Try adding v
        new_B = B + [v]
        if len(new_B) > 2: # Only check flow for size >= 2? 
             # Actually, for size 2, we just need an edge.
             pass
        paths = try_branch_set(new_B)
        if paths is not None:
            B = new_B
            best_B = B
            best_paths = paths
    
    # If B is too small, try to find a better start
    if len(B) < 3:
        # Try all pairs
        found_pair = False
        for i in range(len(sorted_vertices)):
            for j in range(i+1, len(sorted_vertices)):
                u = sorted_vertices[i]
                v = sorted_vertices[j]
                if (u,v) in [(a,b) for a,b in edges] or (v,u) in [(a,b) for a,b in edges]:
                     # Check if they are connected
                     # try_branch_set([u,v]) will verify
                     paths = try_branch_set([u,v])
                     if paths is not None:
                         B = [u, v]
                         best_B = B
                         best_paths = paths
                         found_pair = True
                         break
            if found_pair:
                break
        if not found_pair:
             # Fallback
             return {'branches': [0, 1], 'paths': [[0, 1]]}
        
        # Now expand from this pair
        for v in sorted_vertices:
            if v in B:
                continue
            new_B = B + [v]
            paths = try_branch_set(new_B)
            if paths is not None:
                B = new_B
                best_B = B
                best_paths = paths

    # Final check and result construction
    if not best_B or len(best_B) < 2:
        # Absolute fallback
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    branches = list(best_B)
    result_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            result_paths.append(list(best_paths[(i, j)]))
            
    return {'branches': branches, 'paths': result_paths}
# EVOLVE-BLOCK-END
