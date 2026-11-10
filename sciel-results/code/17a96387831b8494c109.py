import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    n = 16
    # Precompute bitmasks for vertices
    # adj[i] is a bitmask of neighbors of i
    
    # State:
    # adj: list of int (bitmasks)
    # forbidden[i]: bitmask of vertices j > i such that edge (i,j) cannot be added
    #               because it would create a triangle.
    #               Specifically, if we add edge (i,j), we need (adj[i] & adj[j]) == 0.
    #               Actually, for maximization, we want to add edges.
    #               The constraint is: for any i < j < k, not (adj[i] has j and k) AND (adj[j] has k).
    #               Or: adj[i] & adj[j] & (1<<k) == 0 for all k.
    #               This is equivalent to: adj[i] & adj[j] == 0 for all i != j? No.
    #               Triangle-free means: for any distinct i,j,k, not (i~j and j~k and k~i).
    #               This is equivalent to: for any edge (i,j), N(i) and N(j) are disjoint.
    #               i.e., adj[i] & adj[j] == 0 if adj[i] has bit j.
    
    # Let's use a recursive backtracking that decides for each pair (i,j) whether to include the edge.
    # To maximize edges, we should try to include edges first.
    # But we must maintain triangle-freeness.
    
    # A more efficient representation for triangle-free graphs:
    # We can build the graph vertex by vertex.
    # When adding vertex v, we choose its neighbors among {0, ..., v-1}.
    # Let S be the set of neighbors of v.
    # For the graph to remain triangle-free, S must be an independent set in the subgraph induced by {0, ..., v-1}.
    # i.e., no two vertices in S are adjacent.
    
    # This is a standard way to enumerate triangle-free graphs.
    # We can use branch and bound here.
    # Upper bound on edges: current_edges + sum over remaining vertices of max_possible_neighbors.
    # Max possible neighbors for v is v (all previous vertices), but constrained by independence.
    # A tighter bound: for each remaining vertex v, the max neighbors is the size of the maximum independent set in the prefix?
    # That's expensive to compute.
    # Simpler bound: current_edges + sum_{v=k}^{n-1} (n-1 - v) ? No, that's just remaining potential edges.
    # Since we are maximizing, and we process vertices in order, the max additional edges is the number of remaining pairs.
    # This bound is weak.
    
    # Let's try a different approach: Local Search with a very strong heuristic.
    # The "inspiration" mentioned "Randomized greedy edge addition from a Grötzsch graph core".
    # Let's implement a robust version of that.
    
    # 1. Generate a random maximal triangle-free graph (MTF).
    # 2. Check if it's 4-chromatic.
    # 3. If not, try to modify it.
    
    # Actually, the most reliable way to find dense 4-chromatic triangle-free graphs is to start with known dense structures or use a very aggressive local search.
    
    # Let's try a "Simulated Annealing" but with a much better initial state and move set.
    # Or, better: **Exhaustive Search over Independent Sets**.
    
    # Let's stick to the Branch and Bound on Independent Sets.
    # State:
    # k: current vertex index (0 to n-1)
    # adj: list of bitmasks
    # edges_count: current number of edges
    
    # When placing vertex k, we choose a subset S of {0, ..., k-1} to be its neighbors.
    # S must be an independent set in the graph induced by {0, ..., k-1}.
    # We iterate through all possible independent sets? That's exponential.
    # But n=16 is small.
    
    # Let's implement the recursive construction.
    
    # To speed up 3-colorability check:
    # Use bitmasks for colors.
    # C0, C1, C2 are bitmasks of vertices colored 0, 1, 2.
    # A vertex v can be colored c if (adj[v] & C_c) == 0.
    
    def is_3_colorable(adj):
        # adj is list of bitmasks
        # Returns True if 3-colorable
        # Use backtracking with bitmasks
        
        colors = [-1] * n
        # Order vertices by degree descending
        deg = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Available colors
            # If adj[v] & C0 == 0, color 0 is available
            # We maintain C0, C1, C2 as global or passed in?
            # For speed, let's pass them or use a global state with undo.
            # Since n=16, a simple recursive check with bitmasks for color classes is fast.
            
            # Let's use a different fast check:
            # Try all 3^n colorings? 3^16 = 43 million. Too slow in Python.
            # Backtracking is better.
            
            c0_mask = 0
            c1_mask = 0
            c2_mask = 0
            # Reconstructing masks at each step is O(n).
            # Better to maintain them.
            
            # Let's rewrite backtrack to maintain masks
            return _backtrack(idx, 0, 0, 0)
            
        def _backtrack(idx, c0, c1, c2):
            if idx == n:
                return True
            v = order[idx]
            # Try color 0
            if (adj[v] & c0) == 0:
                if _backtrack(idx + 1, c0 | (1 << v), c1, c2):
                    return True
            # Try color 1
            if (adj[v] & c1) == 0:
                if _backtrack(idx + 1, c0, c1 | (1 << v), c2):
                    return True
            # Try color 2
            if (adj[v] & c2) == 0:
                if _backtrack(idx + 1, c0, c1, c2 | (1 << v)):
                    return True
            return False
            
        return _backtrack(0, 0, 0, 0)

    # Branch and Bound to maximize edges in triangle-free graph that is NOT 3-colorable.
    # We build the graph vertex by vertex.
    # At step k, we choose neighbors for vertex k from {0..k-1}.
    # The neighbors must form an independent set in the current graph.
    
    best_edges = 0
    best_adj = None
    deadline_exceeded = False
    
    # Precompute independent sets for each prefix?
    # No, just check on the fly.
    
    def is_independent_set(S, adj):
        # S is a bitmask of vertices
        # Check if no two vertices in S are adjacent
        # For each v in S, (adj[v] & S) should not contain any other vertex in S?
        # Actually, if u,v in S and u~v, then adj[u] has v.
        # So we need: for all v in S, (adj[v] & S) == 0? No, adj[v] doesn't contain v.
        # So (adj[v] & S) should be 0 for all v in S?
        # If u~v, then v is in adj[u]. So adj[u] & S has bit v.
        # So yes, for all v in S, (adj[v] & S) must be 0.
        # Wait, if S={u,v} and u~v, then adj[u] has v. So adj[u] & S has v.
        # So the condition is: for all v in S, (adj[v] & S) == 0.
        
        # Efficient check:
        # Iterate through bits in S.
        s = S
        while s:
            lsb = s & -s
            v = lsb.bit_length() - 1
            if (adj[v] & S) != 0:
                return False
            s ^= lsb
        return True

    # Recursive search
    # adj: list of bitmasks
    # k: next vertex to add (0 to n)
    # edges: current edge count
    
    def search(adj, k, edges):
        nonlocal best_edges, best_adj, deadline_exceeded
        
        if deadline_exceeded:
            return
            
        if time.time() > DEADLINE:
            deadline_exceeded = True
            return
            
        if k == n:
            # Check if 3-colorable
            if not is_3_colorable(adj):
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj[:]
            return
            
        # Upper bound on edges:
        # Max possible edges from remaining vertices:
        # Each new vertex v can connect to at most v previous vertices.
        # So max additional edges = sum_{i=k}^{n-1} i
        # This is a loose bound.
        # Tighter bound: The max independent set size in the prefix graph.
        # But let's use the loose bound first to see if it prunes.
        
        # Actually, a better bound:
        # The graph on {0..k-1} is fixed.
        # For vertex k, we can choose any independent set from {0..k-1}.
        # Let alpha(G_k) be the max independent set size.
        # Then max edges added by k is alpha(G_k).
        # But we don't compute alpha.
        
        # Let's just enumerate independent sets for vertex k.
        # To do this efficiently, we can iterate through subsets of {0..k-1} that are independent.
        
        # Get all independent sets of G[0..k-1]
        # We can generate them recursively or just iterate 2^k subsets and check.
        # 2^16 = 65536. Checking independence is fast.
        # So for each k, we can precompute or iterate all 2^k masks and filter.
        
        # To save time, let's generate candidates.
        # Candidates for neighbors of k: subsets of mask_k = (1 << k) - 1
        # We want to try larger sets first (to maximize edges).
        
        # Generate all subsets of mask_k, sort by popcount descending.
        # This is expensive if done at every node.
        
        # Alternative: Branch on whether to include specific vertices?
        # No, the "Independent Set" choice is the branching factor.
        
        # Let's optimize the independent set generation.
        # We can use a recursive helper to generate independent sets.
        
        # For now, let's iterate all subsets.
        # Optimization: If k is large, 2^k is large.
        # But n=16, so max k=15. 2^15 = 32768.
        # At each node, we do this.
        # Number of nodes?
        # The search tree size is product of number of independent sets.
        # This could be huge.
        
        # We need better pruning.
        # 1. If edges + max_possible_remaining <= best_edges, prune.
        #    max_possible_remaining = sum_{i=k}^{n-1} i
        #    This is very loose.
        
        # 2. Use a tighter upper bound.
        #    The maximum number of edges in a triangle-free graph on m vertices is floor(m^2/4).
        #    For the remaining vertices k..n-1, plus connections to previous?
        #    This is complex.
        
        # Let's try a different strategy: **Local Search with Random Restarts**.
        # The Branch and Bound might be too slow in Python without C extensions.
        # The "inspiration" says "Randomized greedy edge addition from a Grötzsch graph core".
        
        # Let's implement a high-quality Local Search.
        # 1. Start with a random Maximal Triangle-Free graph.
        # 2. Check if 4-chromatic.
        # 3. If yes, try to add edges (if not maximal) or swap edges.
        # 4. If no, try to modify to make it 4-chromatic?
        
        # Actually, the goal is to MAXIMIZE edges.
        # So we want a graph that is BOTH triangle-free AND 4-chromatic.
        # Most dense triangle-free graphs are 3-colorable (e.g., complete bipartite).
        # We need to "break" 3-colorability while maintaining triangle-freeness.
        
        # Strategy:
        # 1. Generate random MTF graphs.
        # 2. Filter for 4-chromatic.
        # 3. Among those, find the one with max edges.
        # 4. Use local search to improve the best one found.
        
        # Let's go back to the simulated annealing idea but make it much faster.
        # The bottleneck is `is_3_colorable`.
        # My bitmask backtracking is fast.
        
        # Let's combine:
        # 1. Fast generation of random MTF graphs.
        # 2. Fast 3-colorability check.
        # 3. Local search (Simulated Annealing) to maximize edges among 4-chromatic MTFs.
        
        # Wait, the local search must stay in the space of 4-chromatic MTFs.
        # Moves:
        # - Add edge (if it doesn't create a triangle). If it breaks 3-colorability, reject (or accept with prob if we allow temporary violation? No, we must stay valid).
        #   Actually, if we add an edge and it becomes 3-colorable, we reject.
        #   If we add an edge and it remains 4-chromatic, we accept (improves objective).
        # - Remove edge. If it becomes 3-colorable, reject (we want to stay 4-chromatic).
        #   If it remains 4-chromatic, we accept (decreases objective, but might allow future additions).
        # - Swap edge?
        
        # This search space is disconnected?
        # If we start with a 4-chromatic graph, removing an edge might make it 3-colorable.
        # So we can't just remove edges freely.
        
        # Alternative: **Exhaustive Search for Small Subgraphs**.
        # Since n=16 is small, maybe we can use an ILP solver? No, standard library only.
        
        # Let's try the **Branch and Bound** again, but with a much tighter bound.
        # Bound:
        # Current edges: E
        # Remaining vertices: R = {k, ..., n-1}
        # Max edges in triangle-free graph on n vertices is floor(n^2/4).
        # But we have a prefix.
        
        # Let's use the fact that the graph must be 4-chromatic.
        # This is a constraint, not a bound.
        
        # Let's try a different exact method:
        # **Iterate over all partitions of vertices into independent sets?**
        # No.
        
        # Let's try **Integer Programming via Branch and Cut**? Too complex.
        
        # Let's go with **High-Performance Local Search**.
        # The key is to find a dense 4-chromatic triangle-free graph.
        # Known dense examples:
        # - Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free).
        # - Blow-ups of Grötzsch graph?
        #   If we replace each vertex of Grötzsch with an independent set of size k, and replace edges with complete bipartite graphs, the result is triangle-free and 4-chromatic.
        #   Vertices: 11 * k.
        #   For n=16, we can't do uniform blow-up.
        #   We can do non-uniform blow-up.
        #   Let sizes be s1, ..., s11 summing to 16.
        #   Edges: sum_{(i,j) in E(Grotzsch)} si * sj.
        #   We want to maximize this.
        
        # Let's implement this!
        # 1. Get Grötzsch graph edges.
        # 2. Iterate over all compositions of 16 into 11 parts (s1...s11 >= 1? No, can be 0? If 0, vertex removed).
        #    If a part is 0, the vertex is removed.
        #    We need to ensure the resulting graph is 4-chromatic.
        #    If we remove vertices, it might become 3-colorable.
        #    So we must check.
        
        # Number of compositions of 16 into 11 parts: C(16+11-1, 11-1) = C(26, 10) ~ 5 million.
        # This is feasible!
        # For each composition, construct the graph, check 3-colorability, count edges.
        
        # This is a **Structural Change**: using a specific parametric family of graphs (blow-ups of Grötzsch) instead of generic local search.
        
        # Let's implement this.
        
        # Grötzsch graph (Mycielski of C5)
        # Vertices 0..4 (C5), 5..9 (neighbors), 10 (center)
        # Edges:
        # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
        # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
        # Center: (5,10), (6,10), (7,10), (8,10), (9,10)
        # Wait, standard Mycielski:
        # Original v_i. New v'_i connected to v_i and neighbors of v_i?
        # No, v'_i connected to v_i. And v'' connected to all v'_i.
        # And v'_i connected to neighbors of v_i in original?
        # Let's check Grötzsch graph definition.
        # It is triangle-free, 4-chromatic, 11 vertices.
        
        # Let's hardcode the edges of the Grötzsch graph.
        # Reference: 
        # 0-1, 1-2, 2-3, 3-4, 4-0 (C5)
        # 0-5, 1-6, 2-7, 3-8, 4-9 (matching)
        # 5-10, 6-10, 7-10, 8-10, 9-10 (star)
        # Plus: 5-2, 5-3? No.
        # Mycielski construction:
        # Given G, create G'.
        # V(G') = V(G) U V'(G) U {w}
        # E(G') = E(G) U {(v, v') | v in V(G)} U {(v', w) | v' in V'(G)}
        # Wait, that's not right.
        # E(G') = E(G) U {(v, v') | v in V(G)} U {(v', w) | v' in V'(G)} U {(v', u) | (u,v) in E(G)}
        
        # Let's construct it programmatically to be sure.
        
        def build_grotzsch():
            # C5
            edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
            # Mycielski
            # v0..v4
            # v0'..v4'
            # w
            # n=11
            # Edges in G:
            for u, v in edges:
                pass # already added
            
            # Add v' nodes (5..9)
            # v_i' connected to v_i
            for i in range(5):
                edges.append((i, 5+i))
                
            # v_i' connected to neighbors of v_i in G
            # Neighbors of 0: 1, 4 -> 5 connected to 1, 4?
            # Neighbors of 1: 0, 2 -> 6 connected to 0, 2
            # Neighbors of 2: 1, 3 -> 7 connected to 1, 3
            # Neighbors of 3: 2, 4 -> 8 connected to 2, 4
            # Neighbors of 4: 3, 0 -> 9 connected to 3, 0
            
            # Let's verify triangle-free.
            # If 5-1 and 1-0 and 5-0? 5-0 is not an edge.
            # 5 connected to 0, 1, 4.
            # 1 connected to 0, 2, 5, 6? No, 6 connected to 0, 2.
            # Is there a triangle?
            # 0-1-5-0? 5-0 is edge. 1-5 is edge. 0-1 is edge.
            # Yes, 0-1-5 is a triangle!
            # So Mycielski definition must be: v_i' connected to neighbors of v_i, NOT v_i itself?
            # No, Mycielski graph is triangle-free.
            # Let's check the definition again.
            # "v' is adjacent to v and to all neighbors of v" -> This creates triangles if v has neighbors.
            # Correct Mycielski:
            # V' = {v' | v in V}
            # E' = E U {(v, v') | v in V} U {(v', w) | v' in V'} U {(v', u) | (u,v) in E}
            # Wait, if (v, v') is an edge, and (v', u) is an edge, and (u,v) is an edge, then v-v'-u-v is a triangle.
            # So (v, v') should NOT be an edge?
            # Actually, the Mycielski graph M(G) has chromatic number chi(G)+1.
            # If G is C5, M(C5) is Grötzsch.
            # Let's look up the exact edges.
            
            # Correct edges for Grötzsch (11 vertices):
            # 0-1, 1-2, 2-3, 3-4, 4-0
            # 0-5, 1-6, 2-7, 3-8, 4-9
            # 5-1, 5-4
            # 6-0, 6-2
            # 7-1, 7-3
            # 8-2, 8-4
            # 9-0, 9-3
            # 5-10, 6-10, 7-10, 8-10, 9-10
            
            # Let's verify 0-1-5-0.
            # 0-1 is edge.
            # 1-5 is edge (from 5-1).
            # 5-0 is edge (from 0-5).
            # Triangle!
            
            # So the edges must be different.
            # In Mycielski construction, the new vertex v' is adjacent to ALL neighbors of v, but NOT to v itself.
            # And there is a new vertex w adjacent to ALL v'.
            
            # Let's re-construct:
            # Base: 0-1-2-3-4-0
            # New nodes 5,6,7,8,9 corresponding to 0,1,2,3,4.
            # 5 (for 0) connected to neighbors of 0: 1, 4.
            # 6 (for 1) connected to neighbors of 1: 0, 2.
            # 7 (for 2) connected to neighbors of 2: 1, 3.
            # 8 (for 3) connected to neighbors of 3: 2, 4.
            # 9 (for 4) connected to neighbors of 4: 3, 0.
            # New node 10 connected to 5,6,7,8,9.
            
            # Check triangle 0-1-5.
            # 0-1 is edge.
            # 1-5? 5 is connected to 1. Yes.
            # 5-0? 5 is connected to neighbors of 0 (1,4). Not 0.
            # So 5-0 is NOT an edge.
            # So no triangle.
            
            # This is the correct Grötzsch graph.
            
            edges = [
                (0,1), (1,2), (2,3), (3,4), (4,0),
                (5,1), (5,4),
                (6,0), (6,2),
                (7,1), (7,3),
                (8,2), (8,4),
                (9,0), (9,3),
                (5,10), (6,10), (7,10), (8,10), (9,10)
            ]
            return edges

        grotzsch_edges = build_grotzsch()
        
        # Now, iterate over all compositions of 16 into 11 parts.
        # Parts can be 0.
        # If part is 0, the vertex is not present.
        # We need to map the 11 "groups" to the 16 actual vertices.
        
        # To iterate compositions of 16 into 11 parts:
        # Use recursion.
        
        best_score = 0
        best_config = None
        
        def compose(remaining, parts_left, current_parts):
            nonlocal best_score, best_config, deadline_exceeded
            
            if deadline_exceeded:
                return
                
            if time.time() > DEADLINE:
                deadline_exceeded = True
                return
                
            if parts_left == 0:
                if remaining == 0:
                    # Check this configuration
                    # current_parts is list of 11 ints
                    # Construct graph
                    # Map group i to vertices [start_i, end_i)
                    
                    # Check if valid (all groups non-empty? No, can be 0)
                    # If a group is 0, those vertices are missing.
                    # The graph is induced by the non-empty groups.
                    # We need to check if the resulting graph is 4-chromatic.
                    
                    # Construct adj matrix
                    adj = [[0]*n for _ in range(n)]
                    vertex_map = {} # group -> list of vertices
                    v_idx = 0
                    for i, size in enumerate(current_parts):
                        if size > 0:
                            vertex_map[i] = list(range(v_idx, v_idx + size))
                            v_idx += size
                            
                    # Add edges
                    for u_g, v_g in grotzsch_edges:
                        if u_g in vertex_map and v_g in vertex_map:
                            for u in vertex_map[u_g]:
                                for v in vertex_map[v_g]:
                                    adj[u][v] = 1
                                    adj[v][u] = 1
                                    
                    # Count edges
                    e_count = sum(sum(row) for row in adj) // 2
                    
                    # Check 3-colorability
                    if not is_3_colorable_bitmask(adj):
                        if e_count > best_score:
                            best_score = e_count
                            best_config = current_parts[:]
                            
                return
                
            # Pruning:
            # Min possible remaining parts: 0
            # Max possible remaining parts: remaining
            # If parts_left > remaining, we must have zeros.
            # If parts_left == 1, current_parts.append(remaining)
            
            if parts_left == 1:
                compose(0, 0, current_parts + [remaining])
                return
            
            # Iterate possible values for next part
            # Max value: remaining
            # Min value: 0
            # To find dense graphs, we might want larger parts.
            # But we must explore all.
            
            # Optimization: Symmetry?
            # The Grötzsch graph has symmetries.
            # But let's just iterate.
            
            # To speed up, we can bound the value.
            # Max value for next part: remaining
            # Min value: 0
            
            # However, if we pick 0, we are effectively reducing the problem size.
            
            # Let's iterate from high to low to find good solutions earlier?
            # Or low to high?
            # High to low might find dense graphs faster.
            
            for val in range(remaining, -1, -1):
                compose(remaining - val, parts_left - 1, current_parts + [val])
                if deadline_exceeded:
                    return

        # Helper for 3-colorability with bitmask adj
        def is_3_colorable_bitmask(adj):
            # adj is list of lists
            # Convert to bitmask
            adj_bits = [0] * n
            for i in range(n):
                mask = 0
                for j in range(n):
                    if adj[i][j]:
                        mask |= (1 << j)
                adj_bits[i] = mask
                
            deg = [bin(adj_bits[i]).count('1') for i in range(n)]
            order = sorted(range(n), key=lambda x: -deg[x])
            
            def _backtrack(idx, c0, c1, c2):
                if idx == n:
                    return True
                v = order[idx]
                if (adj_bits[v] & c0) == 0:
                    if _backtrack(idx + 1, c0 | (1 << v), c1, c2):
                        return True
                if (adj_bits[v] & c1) == 0:
                    if _backtrack(idx + 1, c0, c1 | (1 << v), c2):
                        return True
                if (adj_bits[v] & c2) == 0:
                    if _backtrack(idx + 1, c0, c1, c2 | (1 << v)):
                        return True
                return False
                
            return _backtrack(0, 0, 0, 0)

        compose(16, 11, [])
        
        if best_config:
            # Reconstruct best graph
            adj = [[0]*n for _ in range(n)]
            vertex_map = {}
            v_idx = 0
            for i, size in enumerate(best_config):
                if size > 0:
                    vertex_map[i] = list(range(v_idx, v_idx + size))
                    v_idx += size
                    
            edges = []
            for u_g, v_g in grotzsch_edges:
                if u_g in vertex_map and v_g in vertex_map:
                    for u in vertex_map[u_g]:
                        for v in vertex_map[v_g]:
                            edges.append([u, v])
                            adj[u][v] = 1
                            adj[v][u] = 1
                            
            return edges
        else:
            # Fallback: Return a known 4-chromatic triangle-free graph
            # Just return the Grötzsch graph edges mapped to 16 vertices (with 5 isolated vertices? No, must be on 0..15)
            # We can just take the first 11 vertices as Grötzsch and leave 11-15 isolated.
            # But the task says "on vertices 0..15". Isolated vertices are allowed?
            # "simple triangle-free graph on vertices 0..15". Yes.
            # But maximizing edges suggests using all vertices.
            # If best_config is None, it means no 4-chromatic blow-up was found?
            # That's unlikely. The Grötzsch graph itself (sizes 1,1,1,1,1,1,1,1,1,1,1) is 4-chromatic.
            # Sum = 11. We have 16. So we can add 5 more vertices to some groups.
            # e.g., 2,1,1,1,1,1,1,1,1,1,5.
            
            # So best_config should not be None.
            
            # If it is, return a valid fallback.
            return [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
                    [5, 1], [5, 4],
                    [6, 0], [6, 2],
                    [7, 1], [7, 3],
                    [8, 2], [8, 4],
                    [9, 0], [9, 3],
                    [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]]

