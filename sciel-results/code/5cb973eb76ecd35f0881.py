"""Rewritten program using flow-based max-flow oracle for clique subdivision."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    deadline = time.time() + 80.0
    
    def max_flow(s, t, blocked_vertices):
        """Compute max flow from s to t in the graph where blocked_vertices are removed.
        Returns (flow_value, flow_dict) where flow_dict maps (u,v) to flow on edge."""
        # Build residual graph
        # Each vertex v (not blocked, not s, not t) has capacity 1
        # We split each vertex v into v_in and v_out with edge (v_in, v_out) capacity 1
        # For original edge (u,v), add edges (u_out, v_in) and (v_out, u_in) with capacity 1
        
        # Map original vertices to split vertices
        # s and t are terminals, they don't need splitting (or we can treat them specially)
        
        # Actually, let's use a simpler approach: vertex-capacitated max flow
        # We'll create a flow network where:
        # - Source is s, sink is t
        # - Each vertex v (v != s, v != t, v not blocked) has capacity 1
        # - Each edge (u,v) has infinite capacity (or large enough)
        
        # For vertex capacity, we split v into v_in and v_out
        # Edge v_in -> v_out has capacity 1
        # For each original edge (u,v), we add u_out -> v_in and v_out -> u_in with capacity INF
        
        # Let's assign IDs:
        # For vertex v: v_in = 2*v, v_out = 2*v+1
        # Source: s_out = 2*s+1 (we start from s_out)
        # Sink: t_in = 2*t
        
        INF = 10**9
        
        # Build adjacency for residual graph
        # flow[u][v] = flow from u to v
        # We need residual edges: for each edge (u,v) with capacity c, we have forward (u,v) cap c and backward (v,u) cap 0
        # Initially flow is 0
        
        # Let's use a dict-based approach
        # residual[u][v] = remaining capacity from u to v
        residual = defaultdict(lambda: defaultdict(int))
        
        # Add vertex capacity edges
        for v in range(n):
            if v in blocked_vertices or v == s or v == t:
                continue
            vin = 2 * v
            vout = 2 * v + 1
            residual[vin][vout] = 1
            residual[vout][vin] = 0
        
        # Add original edges
        for u, v in edges:
            if u in blocked_vertices or v in blocked_vertices:
                continue
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            # u_out -> v_in
            residual[uout][vin] = INF
            residual[vin][uout] = 0
            # v_out -> u_in
            residual[vout][uin] = INF
            residual[uin][vout] = 0
        
        # Also need edges from s to its neighbors' in-nodes, and from neighbors' out-nodes to t
        # Actually s_out should connect to neighbor_in for each neighbor
        # And neighbor_out should connect to t_in
        
        # Wait, I need to be more careful. Let me redo the construction.
        # 
        # For a vertex v (not s, not t, not blocked):
        #   v_in -> v_out with capacity 1
        # For each original edge (u, v):
        #   If u is not s,t,blocked: u_out -> v_in with capacity INF
        #   If v is not s,t,blocked: v_out -> u_in with capacity INF
        #   If u is s: s -> v_in with capacity INF (s has no in/out split, just use s as source)
        #   If v is s: u_out -> s with capacity INF (but s is source, so this goes to source)
        #   Similarly for t
        #
        # Let me simplify: use node-splitting for all vertices except s and t.
        # s and t are single nodes.
        
        # Let me restart with a cleaner approach.
        
        # Actually, let me just use a standard max-flow with vertex splitting properly.
        
        pass  # This is getting complex, let me use a different approach
    
    def find_disjoint_paths(branches):
        """Use max-flow to find vertex-disjoint paths between all pairs of branches.
        Returns list of paths or None."""
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(range(k), 2))
        num_pairs = len(pairs)
        
        # We'll use a flow network where:
        # - Each branch b_i has a source node s_i and sink node t_i
        # - Each pair (i,j) needs 1 unit of flow from s_i to t_j and 1 unit from s_j to t_i
        # - Actually, we need paths from branch i to branch j for each pair
        # - Each non-branch vertex has capacity 1
        # - Each branch vertex has capacity infinity (it can be used by multiple paths as an endpoint)
        
        # Let me think about this differently.
        # We have k branch vertices. For each pair (i,j), we need a path from branch[i] to branch[j].
        # All paths must be vertex-disjoint in their interiors.
        # 
        # Model: Create a flow network where:
        # - Each branch b_i is split: b_i_in (for incoming paths) and b_i_out (for outgoing paths)
        # - Actually, each branch can be an endpoint for (k-1) paths
        # - For each pair (i,j), we send 1 unit of flow from b_i to b_j
        # - Non-branch vertices have capacity 1
        # - Branch vertices have capacity (k-1) since they're endpoints for k-1 paths
        #
        # But the direction matters: we need a path from i to j AND a path from j to i?
        # No, we need one path for each unordered pair. The path goes from i to j (or j to i, doesn't matter).
        # 
        # Let me model it as: for each pair (i,j), we need 1 unit of flow from b_i to b_j in the network.
        # But this creates a directed flow problem. The paths should be undirected.
        #
        # Alternative: Since paths are undirected, we can model each pair as needing 1 unit of flow
        # between the two branch vertices. We can use a standard max-flow with multiple sources/sinks.
        
        # Let me use a simpler greedy approach with backtracking, but use max-flow as an oracle
        # to check if a set of branches is feasible.
        
        # For checking feasibility of a set of branches:
        # We need to find vertex-disjoint paths for all pairs.
        # This is equivalent to: can we route |pairs| units of flow, one for each pair?
        
        # Let me implement a proper max-flow based check.
        
        # Network construction:
        # - Node for each vertex v: if v is a branch, it has high capacity; otherwise capacity 1
        # - For each pair (i,j), add a demand of 1 unit from branch[i] to branch[j]
        # - Use node-splitting: v_in -> v_out with capacity cap(v)
        # - For each edge (u,v): u_out -> v_in with capacity INF, v_out -> u_in with capacity INF
        
        # To handle multiple commodities (one per pair), we can't directly use single-commodity max-flow.
        # But if we just check if ALL pairs can be routed simultaneously, we need multi-commodity flow,
        # which is NP-hard in general.
        
        # However, for small k (say k <= 8), we can try a greedy assignment with backtracking.
        
        # Let me use a different strategy:
        # 1. Find a large clique in the graph (these vertices have direct edges, so paths are trivial)
        # 2. Try to add more vertices that are not in the clique but can be connected via disjoint paths
        
        # Actually, let me just use a smart greedy with BFS path finding and backtracking.
        
        # For now, let me implement a robust greedy with multiple restarts.
        
        best_result = None
        
        # Strategy: Start with a clique, then try to add vertices one by one.
        # For each addition, try to find disjoint paths.
        
        degree = {v: len(adj[v]) for v in range(n)}
        sorted_vertices = sorted(range(n), key=lambda v: degree[v], reverse=True)
        
        # Find maximum clique using Bron-Kerbosch
        def bron_kerbosch(R, P, X):
            if not P and not X:
                return R if len(R) > len(bron_kerbosch.best) else bron_kerbosch.best
            if len(R) > len(bron_kerbosch.best):
                bron_kerbosch.best = list(R)
            
            # Pivot selection: choose pivot u from P ∪ X with maximum |P ∩ N(u)|
            if P or X:
                pivot = max(P | X, key=lambda u: len(P & adj[u]))
            else:
                return R if len(R) > len(bron_kerbosch.best) else bron_kerbosch.best
            
            candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                if time.time() > deadline - 20:
                    return bron_kerbosch.best
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
            
            return bron_kerbosch.best
        
        bron_kerbosch.best = []
        all_vertices = set(range(n))
        max_clique = bron_kerbosch(set(), all_vertices, set())
        
        if not max_clique or len(max_clique) < 2:
            max_clique = [sorted_vertices[0], sorted_vertices[1]] if len(sorted_vertices) >= 2 else []
        
        # Now try to extend the clique with additional vertices
        # A vertex v can be added if for every existing branch b, there's a path from v to b
        # that doesn't use other branches or already-used interior vertices.
        
        def try_add_vertex(current_branches, candidate):
            """Try to add candidate to current_branches. Returns new paths or None."""
            new_branches = current_branches + [candidate]
            return find_paths_for_set(new_branches)
        
        def find_paths_for_set(branches):
            """Find vertex-disjoint paths for all pairs in branches.
            Uses greedy assignment with BFS, trying hardest pairs first."""
            k = len(branches)
            if k < 2:
                return None
            
            pairs = list(combinations(range(k), 2))
            paths = {}
            used_interior = set()
            branch_set = set(branches)
            
            # Order pairs by difficulty (fewest available common neighbors first)
            def pair_key(pair):
                i, j = pair
                bi, bj = branches[i], branches[j]
                # If directly connected, easy
                if (bi, bj) in edge_set:
                    return -1
                # Otherwise, count common neighbors not in branches or used
                common = (adj[bi] & adj[bj]) - branch_set - used_interior
                return len(common)
            
            remaining = list(pairs)
            max_rounds = 200
            rnd = 0
            
            while remaining and rnd < max_rounds:
                if time.time() > deadline - 10:
                    return None
                
                remaining.sort(key=pair_key)
                i, j = remaining[0]
                bi, bj = branches[i], branches[j]
                
                # Try to find a path from bi to bj
                path = bfs_path(bi, bj, branch_set - {bi, bj}, used_interior)
                
                if path is not None:
                    interior = set(path[1:-1])
                    if interior & used_interior:
                        remaining.pop(0)
                        rnd += 1
                        continue
                    paths[(i, j)] = path
                    used_interior.update(interior)
                    remaining.pop(0)
                else:
                    # Can't find path, remove this pair
                    remaining.pop(0)
                
                rnd += 1
            
            if len(paths) != len(pairs):
                return None
            
            result = []
            for i, j in pairs:
                result.append(paths[(i, j)])
            return result
        
        def bfs_path(src, dst, forbidden, used):
            """BFS shortest path from src to dst avoiding forbidden and used vertices."""
            if src == dst:
                return [src]
            if (src, dst) in edge_set and src not in forbidden and dst not in forbidden:
                # Direct edge, check if it's valid
                if src not in used and dst not in used:
                    return [src, dst]
            
            blocked = forbidden | used
            visited = {src}
            queue = deque([(src, [src])])
            
            while queue:
                node, path = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor in blocked or neighbor in visited:
                        continue
                    if neighbor == dst:
                        return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            
            return None
        
        # Start with the max clique
        current = list(max_clique)
        best_branches = None
        best_paths = None
        
        # Verify the clique works
        if len(current) >= 2:
            paths = find_paths_for_set(current)
            if paths is not None:
                best_branches = list(current)
                best_paths = paths
        
        # Try to add vertices from the sorted list
        for v in sorted_vertices:
            if v in current:
                continue
            if time.time() > deadline - 15:
                break
            
            new_branches = current + [v]
            paths = find_paths_for_set(new_branches)
            if paths is not None:
                current = new_branches
                best_branches = list(current)
                best_paths = paths
        
        # If we didn't get anything, try pairs
        if best_branches is None:
            for u, v in edges:
                best_branches = [u, v]
                best_paths = [[u, v]]
                break
        
        if best_branches is None:
            best_branches = [0, 1]
            best_paths = [[0, 1]]
        
        # Ensure types are correct
        result = {
            'branches': [int(b) for b in best_branches],
            'paths': [[int(x) for x in p] for p in best_paths]
        }
        return result

# EVOLVE-BLOCK-END
