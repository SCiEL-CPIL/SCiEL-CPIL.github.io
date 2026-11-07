import time
from collections import deque
import sys

# Increase recursion limit just in case, though we use iterative flow
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Global deadline
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer
    
    # Dinic's Max Flow Implementation
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
            self.level = [0] * n
            self.it = [0] * n
        
        def add_edge(self, fr, to, cap):
            forward = [to, cap, None]
            backward = [fr, 0, forward]
            forward[2] = backward
            self.graph[fr].append(forward)
            self.graph[to].append(backward)
        
        def bfs(self, s, t):
            self.level = [-1] * self.n
            q = deque([s])
            self.level[s] = 0
            while q:
                v = q.popleft()
                for e in self.graph[v]:
                    if e[1] > 0 and self.level[e[0]] < 0:
                        self.level[e[0]] = self.level[v] + 1
                        q.append(e[0])
            return self.level[t] >= 0
        
        def dfs(self, v, t, f):
            if v == t:
                return f
            for i in range(self.it[v], len(self.graph[v])):
                self.it[v] = i
                e = self.graph[v][i]
                if e[1] > 0 and self.level[v] < self.level[e[0]]:
                    d = self.dfs(e[0], t, min(f, e[1]))
                    if d > 0:
                        e[1] -= d
                        e[2][1] += d
                        return d
            return 0
        
        def max_flow(self, s, t):
            flow = 0
            INF = 10**18
            while self.bfs(s, t):
                self.it = [0] * self.n
                f = INF
                while f:
                    f = self.dfs(s, t, f)
                    if not f:
                        break
                    flow += f
            return flow

    def check_extension(new_branch, current_branches, used_internal):
        """
        Check if new_branch can be added to current_branches.
        We need paths from new_branch to each b in current_branches.
        These paths must be vertex-disjoint (except at endpoints) and avoid used_internal and other branches.
        
        Flow Network Construction:
        Nodes:
        - Source S (index 0)
        - Sink T (index 1)
        - For each vertex v in 0..n-1:
          - v_in = 2 + v
          - v_out = 2 + n + v
        Total nodes: 2 + 2*n
        
        Edges:
        - S -> new_branch_in : Cap = len(current_branches) (allow multiple paths to leave)
          Actually, paths start at new_branch. So S -> new_branch_out?
          Let's say flow starts at new_branch.
          We want to send 1 unit of flow to each b in current_branches.
          So S -> new_branch_in with Cap = K.
          new_branch_in -> new_branch_out with Cap = K (since it's the source, it can emit K flows).
          
        - For each b in current_branches:
          b_in -> b_out with Cap = 1 (each branch is a destination for exactly 1 path)
          b_out -> T with Cap = 1
        
        - For all other vertices v (not new_branch, not in current_branches, not in used_internal):
          v_in -> v_out with Cap = 1 (vertex capacity 1)
          
        - For all edges (u, v) in graph:
          If u is not blocked and v is not blocked:
            u_out -> v_in with Cap = INF
            v_out -> u_in with Cap = INF
            
        Blocked vertices:
        - used_internal
        - current_branches (except they act as sinks, so their internal edges are handled specifically)
        - Wait, the paths must NOT go through other branches.
          So for b in current_branches, we treat them as terminals.
          Flow goes TO b. It should not pass THROUGH b to another node.
          So b_in -> b_out has Cap 1, and b_out -> T has Cap 1.
          Do we allow flow to leave b_out to other nodes?
          If we add edges from b_out to neighbors, flow could pass through b.
          We must prevent this.
          So, for b in current_branches, we should NOT add edges from b_out to any neighbor_in.
          Only b_out -> T.
          
        - What about new_branch?
          new_branch is the source.
          Flow enters new_branch_in from S.
          Leaves new_branch_out to neighbors.
          We must prevent flow from entering new_branch from neighbors?
          S -> new_branch_in is the only entry.
          So we don't add edges to new_branch_in from other nodes?
          Actually, if we add edges u_out -> new_branch_in, flow could cycle or enter from elsewhere.
          But since S is the only source with capacity, and we are checking max flow from S to T,
          flow conservation applies.
          However, to be safe and simple:
          We only add edges u_out -> v_in if v is not the source (new_branch) and u is not a sink (current_branch).
          Actually, standard vertex-disjoint path flow:
          Split all vertices.
          Source: new_branch. Sink: set of current_branches.
          Remove all edges entering new_branch (except from S).
          Remove all edges leaving current_branches (except to T).
          
        Let's refine the graph construction for the flow:
        
        Nodes:
        S = 0
        T = 1
        For v in 0..n-1:
            in_v = 2 + v
            out_v = 2 + n + v
            
        1. Split edges:
           For all v:
               if v == new_branch:
                   add_edge(in_v, out_v, K) # K = len(current_branches)
               elif v in current_branches:
                   add_edge(in_v, out_v, 1)
               elif v not in used_internal:
                   add_edge(in_v, out_v, 1)
               else:
                   # blocked, no edge or 0 cap
                   pass
        
        2. Source connection:
           add_edge(S, in_new_branch, K)
           
        3. Sink connections:
           For b in current_branches:
               add_edge(out_b, T, 1)
               
        4. Graph edges:
           For u, v in edges:
               # u -> v
               if u != new_branch and v != new_branch: # Don't enter source
                   if u not in current_branches: # Don't leave sink
                       if u not in used_internal and v not in used_internal:
                           add_edge(out_u, in_v, INF)
               # v -> u
               if v != new_branch and u != new_branch:
                   if v not in current_branches:
                       if v not in used_internal and u not in used_internal:
                           add_edge(out_v, in_u, INF)
        
        Run Max Flow.
        If flow == K, it's valid.
        Extract paths from the flow.
        """
        K = len(current_branches)
        if K == 0:
            return True, {}
            
        # Node mapping
        def in_node(v): return 2 + v
        def out_node(v): return 2 + n + v
        
        N_flow = 2 + 2 * n
        dinic = Dinic(N_flow)
        S = 0
        T = 1
        INF = 10**9
        
        # 1. Vertex split capacities
        for v in range(n):
            if v == new_branch:
                dinic.add_edge(in_node(v), out_node(v), K)
            elif v in current_branches:
                dinic.add_edge(in_node(v), out_node(v), 1)
            elif v not in used_internal:
                dinic.add_edge(in_node(v), out_node(v), 1)
            # else: blocked, no edge added (capacity 0 effectively)
        
        # 2. Source
        dinic.add_edge(S, in_node(new_branch), K)
        
        # 3. Sinks
        for b in current_branches:
            dinic.add_edge(out_node(b), T, 1)
            
        # 4. Original graph edges
        for u, v in edges:
            # u -> v
            # Flow can go from u to v if:
            # - u is not a sink (current branch) -> flow shouldn't leave a sink
            # - v is not the source (new branch) -> flow shouldn't enter source
            # - u, v not blocked (used_internal)
            if u not in current_branches and v != new_branch and u not in used_internal and v not in used_internal:
                dinic.add_edge(out_node(u), in_node(v), INF)
            
            # v -> u
            if v not in current_branches and u != new_branch and v not in used_internal and u not in used_internal:
                dinic.add_edge(out_node(v), in_node(u), INF)
                
        flow = dinic.max_flow(S, T)
        
        if flow < K:
            return False, {}
        
        # Extract paths
        # We need to find the K paths from new_branch to each b in current_branches.
        # The flow is integral. We can trace back from T.
        # Or trace forward from S.
        # Since capacities on vertex splits are 1 (except source), the paths are unique in terms of vertex usage.
        
        paths = {}
        # We need to identify which path goes to which branch.
        # The edges into T are from out_node(b).
        # So we can trace back from T via the edge with flow 1.
        
        # Build a reverse graph of used edges?
        # Simpler: For each b in current_branches, find the path from new_branch to b in the flow residual?
        # Actually, we can just DFS from S in the graph of edges with flow > 0.
        # But we need to separate the K paths.
        
        # Let's collect the edges with flow.
        # Since it's a flow network, we can find paths by following flow.
        # Start at S.
        # S -> in_new_branch (Flow K)
        # From in_new_branch, go to out_new_branch.
        # From out_new_branch, we have K outgoing edges with flow 1 (to neighbors).
        
        # Let's do a DFS from S, but we need to keep track of "which path" we are on.
        # Since the network is a DAG of flow (cycles would have been canceled or not used in max flow? Not necessarily, but vertex capacities 1 prevent cycles in the *used* subgraph? No, cycles are possible in flow, but we can extract simple paths).
        # Actually, with vertex capacity 1, the subgraph of edges with flow > 0 is a collection of paths and cycles.
        # Since S is the only source and T the only sink, and flow is K, it's K edge-disjoint paths (which implies vertex-disjoint due to split capacities).
        
        # Let's find the paths by tracing from T backwards.
        # For each b in current_branches:
        #   prev = T
        #   Find edge (u, T) with flow 1. u must be out_node(b).
        #   Then find edge (u_in, u_out) with flow 1.
        #   Then find edge (w_out, u_in) with flow 1.
        #   ... until we reach in_node(new_branch).
        #   Then S -> in_node(new_branch).
        
        # To make this efficient, let's build a list of (from, to, cap, flow) for all edges.
        # Dinic stores edges as [to, cap, back].
        # We can iterate through dinic.graph.
        
        # Map from (u, v) to flow?
        # Note: parallel edges are not an issue here as we add at most 1 edge per pair (u_out, v_in).
        # Wait, we added edges for u->v and v->u. They are distinct (u_out->v_in vs v_out->u_in).
        
        # Let's create a map: node -> list of (prev_node, flow)
        # We need to be careful with the split nodes.
        
        # Let's just do a DFS from S in the "flow graph".
        # We need to distinguish the K paths.
        
        # Since K is small, we can just do this:
        # For each target b in current_branches:
        #   Find a path from new_branch to b in the graph of edges with residual capacity < original capacity?
        #   Actually, easier: The flow is 1 on the specific edges used.
        #   Let's identify the set of edges with flow > 0.
        
        used_edges = []
        for u in range(N_flow):
            for e in dinic.graph[u]:
                # e[1] is current capacity. Original capacity was added.
                # We didn't store original capacity separately in the edge object easily.
                # But we know:
                # - S->in_nb: cap K
                # - in_v->out_v: cap 1 or K
                # - out_b->T: cap 1
                # - graph edges: cap INF
                # 
                # If e[1] < original_cap, then flow = original_cap - e[1].
                # We need original_cap.
                # Let's re-construct or store it.
                # For now, let's assume we can check if flow > 0 by checking if e[2][1] > 0 (backward capacity).
                # Backward capacity is exactly the flow sent!
                if e[2][1] > 0:
                    used_edges.append((u, e[0], e[2][1]))
        
        # Now we have a list of (u, v, flow).
        # We want to find K paths from S to T.
        # Since it's a flow, we can just pick any path from S to T in the used_edges graph, remove those edges, repeat K times.
        
        # Build adjacency list for used edges
        adj_flow = [[] for _ in range(N_flow)]
        for u, v, f in used_edges:
            if f > 0:
                adj_flow[u].append(v)
                
        found_paths = []
        for _ in range(K):
            # Find path from S to T in adj_flow
            visited = set()
            path = []
            def dfs_flow(v):
                if v == T:
                    return True
                visited.add(v)
                for w in adj_flow[v]:
                    if w not in visited:
                        path.append(w)
                        if dfs_flow(w):
                            return True
                        path.pop()
                return False
            
            path.append(S)
            if dfs_flow(S):
                # path is S -> ... -> T
                # Extract vertex sequence
                # Nodes in path: S, in_nb, out_nb, ..., in_b, out_b, T
                # Map flow nodes to original vertices
                # in_v = 2+v, out_v = 2+n+v
                # S=0, T=1
                orig_path = []
                for node in path:
                    if node == S or node == T:
                        continue
                    if node < 2 + n:
                        orig_path.append(node - 2)
                    else:
                        orig_path.append(node - 2 - n)
                # The path should start with new_branch and end with some b
                # But wait, the flow graph might have merged paths?
                # No, vertex capacities are 1, so paths are vertex-disjoint.
                # The path found is a simple path in the flow graph.
                # It corresponds to one of the K paths.
                # We need to associate it with the correct branch b.
                # The last vertex before T is out_b.
                # So b = orig_path[-1]
                b = orig_path[-1]
                # The path in original graph is orig_path
                # But we need to ensure it starts at new_branch.
                # orig_path[0] should be new_branch.
                if orig_path[0] == new_branch and b in current_branches:
                    # Store path from new_branch to b
                    # The path is new_branch -> ... -> b
                    # We store it as [new_branch, ..., b]
                    found_paths.append((b, orig_path))
            
            # Remove used edges from adj_flow for next iteration?
            # Since we found a simple path, and capacities are 1, these edges are "saturated".
            # We can just remove them from adj_flow.
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                if v in adj_flow[u]:
                    adj_flow[u].remove(v)
                    
        # We expect K paths
        if len(found_paths) == K:
            # Convert to dict: branch -> path
            path_dict = {b: p for b, p in found_paths}
            return True, path_dict
        
        return False, {}

    # Greedy Extension
    # 1. Sort vertices by degree
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    current_branches = []
    current_paths = {} # (u, v) -> path
    used_internal = set()
    
    # Initialize with top vertex
    if n > 0:
        current_branches.append(sorted_vertices[0])
        
    # Iterate to add more branches
    max_k = n
    while len(current_branches) < max_k and time.time() < deadline:
        added = False
        # Try adding a new vertex
        # Candidates: vertices not in current_branches
        # Sort candidates by degree or distance to current branches?
        # Let's try all candidates, maybe prioritize high degree.
        candidates = [v for v in sorted_vertices if v not in current_branches]
        
        for v in candidates:
            if time.time() > deadline:
                break
                
            # Check if v can be added
            valid, new_paths = check_extension(v, current_branches, used_internal)
            
            if valid:
                # Add v to branches
                current_branches.append(v)
                
                # Update paths
                # new_paths maps b -> path from v to b
                # We need to add these to current_paths
                # And update used_internal
                for b, path in new_paths.items():
                    # Path is [v, ..., b]
                    # Internal nodes are path[1:-1]
                    # Add to used_internal
                    for node in path[1:-1]:
                        used_internal.add(node)
                    # Store in current_paths
                    # Key: tuple(sorted((v, b)))
                    key = tuple(sorted((v, b)))
                    current_paths[key] = path
                    
                added = True
                break
        
        if not added:
            break
            
    # If we only have 1 branch, we need at least 2.
    # The loop above starts with 1 branch and tries to add a 2nd.
    # If it fails, we have 1 branch. We need to return at least 2.
    # If we can't add a 2nd branch to the first one, maybe the first choice was bad?
    # Or maybe the graph is disconnected?
    # For safety, if we have < 2 branches, try to find any edge.
    
    if len(current_branches) < 2:
        # Fallback: Find any edge
        for u, v in edges:
            # Check if u and v are not in used_internal (which is empty)
            # Just take the first edge
            current_branches = [u, v]
            current_paths = {tuple(sorted((u, v))): [u, v]}
            used_internal = set()
            break
            
    if len(current_branches) < 2:
        # Should not happen
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    # Finalize
    branches_list = [int(b) for b in current_branches]
    paths_list = []
    # We need to return paths for ALL pairs in branches_list
    # current_paths has keys as sorted tuples.
    # We just need to return the list of paths.
    for key, path in current_paths.items():
        paths_list.append([int(v) for v in path])
        
    # Ensure we have exactly C(k, 2) paths?
    # The problem says: "paths': one simple path for each unordered pair of branches"
    # So we must have k*(k-1)/2 paths.
    # My greedy method adds paths for the new vertex to ALL existing branches.
    # So if I have k branches, I have added paths for:
    # Step 1: 1 branch.
    # Step 2: Add b2. Paths: (b1, b2). Total 1.
    # Step 3: Add b3. Paths: (b1, b3), (b2, b3). Total 3.
    # Step k: Add bk. Paths: (b1, bk)...(b_{k-1}, bk). Total 1 + 2 + ... + k-1 = k(k-1)/2.
    # So yes, we have all pairs.
    
    return {
        'branches': branches_list,
        'paths': paths_list
    }

# EVOLVE-BLOCK-END
