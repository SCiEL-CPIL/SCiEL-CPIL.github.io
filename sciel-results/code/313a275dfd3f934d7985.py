import random
import time
import sys
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 16
    # Vertices 0..15

    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))

    # We want a triangle-free graph on 16 vertices that is not 3-colorable.
    # Maximize edges.
    # A known upper bound for triangle-free graphs is n^2/4 = 64 (for n=16).
    # But we also need it to be not 3-colorable.
    # The Grötzsch graph is the smallest triangle-free 4-chromatic graph (11 vertices, 20 edges).
    # We can extend it.

    # Strategy:
    # 1. Start with a known 4-chromatic triangle-free graph on fewer vertices, or build one.
    # 2. Use a backtracking search to add edges while maintaining triangle-free and 4-chromatic properties.
    # 3. Or, use a randomized approach to find a dense triangle-free graph and check if it's 4-chromatic.

    # Let's use a more direct approach:
    # Start with the Grötzsch graph on vertices 0..10.
    # Then add vertices 11..15 and connect them optimally.

    def _build_grotzsch_edges():
        # Grötzsch graph (Mycielski of C5)
        # Vertices 0..10
        # C5: 0-1-2-3-4-0
        # Mycielski construction:
        # Original: 0..4
        # Copies: 5..9 (5 is copy of 0, 6 of 1, etc.)
        # Top: 10
        edges = []
        # C5 edges
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([2, 3])
        edges.append([3, 4])
        edges.append([4, 0])
        # Mycielski edges:
        # For each edge (u,v) in C5, add (u, copy(v)) and (v, copy(u))
        # copy(i) = 5+i
        c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in c5_edges:
            edges.append([u, 5+v])
            edges.append([v, 5+u])
        # Top vertex 10 connected to all copies 5..9
        for i in range(5):
            edges.append([10, 5+i])
        return edges

    grotzsch_edges = _build_grotzsch_edges()

    # Check if Grötzsch is triangle-free and 4-chromatic
    # It is known to be so.

    # Now we have 11 vertices. We need to add 5 more vertices (11..15).
    # We want to maximize the total number of edges.

    # Let's represent the graph as a set of edges and an adjacency list/matrix.
    # We will try to add edges involving the new vertices, and possibly rearrange the old ones?
    # No, let's keep the Grötzsch core and add the new vertices.

    # A better approach:
    # Use a general backtracking search to find the maximum edge triangle-free 4-chromatic graph on 16 vertices.
    # This is hard.

    # Alternative:
    # Start with a random triangle-free graph, ensure it's 4-chromatic, and try to add edges.

    # Let's try a hybrid approach:
    # 1. Start with Grötzsch + 5 isolated vertices.
    # 2. Try to add edges from new vertices to old ones and between new vertices.
    # 3. Use local search to optimize.

    # However, the current program (score 2.25) uses a specific construction.
    # Let's look at the current program's logic more closely.
    # It builds Grötzsch, finds max independent sets, and assigns them to new vertices.
    # This is a specific type of extension.

    # Let's try a different extension strategy:
    # The Grötzsch graph has independence number 5.
    # If we add a new vertex connected to an independent set, we might preserve 4-chromaticity if done carefully.

    # Actually, let's try to generate random triangle-free graphs and check their chromatic number.
    # If we find one with many edges and chromatic number >= 4, we can try to add more edges.

    def is_triangle_free(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    def chromatic_number(edges):
        # Simple backtracking to find chromatic number
        # For n=16, this can be slow if not optimized, but we only need to check if it's <= 3
        # So we can just check if it's 3-colorable.
        
        def is_3_colorable():
            # Backtracking with bitmasks
            # color[i] is the color of vertex i (0, 1, 2)
            # We can use a recursive search
            
            # Optimization: Order vertices by degree (descending)
            degrees = [0] * n
            adj = [0] * n
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
            
            order = sorted(range(n), key=lambda x: -degrees[x])
            
            # We can use a more efficient algorithm, but for n=16, backtracking might suffice with pruning.
            # Let's use a simple recursive backtracking with forward checking.
            
            color = [-1] * n
            # Map original vertex to position in order? No, just use original indices.
            
            def backtrack(idx):
                if idx == n:
                    return True
                v = order[idx]
                # Determine available colors
                used_colors = [False] * 3
                for u in range(n):
                    if color[u] != -1 and (adj[v] & (1 << u)):
                        used_colors[color[u]] = True
                
                for c in range(3):
                    if not used_colors[c]:
                        color[v] = c
                        if backtrack(idx + 1):
                            return True
                        color[v] = -1
                return False
            
            return backtrack(0)
        
        if is_3_colorable():
            return 3
        else:
            return 4 # Or higher, but we only care if > 3

    # This chromatic_number check is too slow for many iterations.
    # We need a faster way to check 3-colorability or a better construction.

    # Let's stick to the construction approach but improve it.
    # The current program scores 2.25.
    # It constructs a graph with ~45 edges? (20 from Grötzsch + ~25 new).
    # Max possible is 64.
    
    # Let's try to use the fact that the complement of a triangle-free graph is a graph with no independent set of size 3.
    # So we are looking for a graph with no triangle and no independent set of size 3? No, that's 4-chromatic.
    # Wait, 4-chromatic means it cannot be colored with 3 colors.
    # This is equivalent to saying that for any partition into 3 sets, one set contains an edge.
    # Or, the independence number is not the only constraint.

    # Let's try a randomized local search.
    # 1. Start with a random triangle-free graph.
    # 2. Ensure it is not 3-colorable (if it is, add an edge that breaks 3-colorability but not triangle-free).
    # 3. Try to add random edges that maintain triangle-free property.

    # How to ensure not 3-colorable?
    # Start with Grötzsch.
    # Then add vertices and edges greedily.

    # Let's refine the Grötzsch extension.
    # Grötzsch (11 vertices).
    # Add vertex 11. Connect it to an independent set in Grötzsch.
    # To maintain 4-chromaticity, the new vertex must not be "redundant".
    
    # Actually, there is a known graph: The Grötzsch graph can be extended to 16 vertices with more edges.
    # Let's try to search for the optimal configuration of the 5 new vertices.
    
    # We will use the same framework as the current program but with a more exhaustive search or better heuristics.
    
    # 1. Build Grötzsch
    core_edges = _build_grotzsch_edges()
    core_n = 11
    new_vertices = list(range(core_n, n))
    
    # 2. Find all independent sets in Grötzsch (not just max, but all, or a subset)
    # Actually, for each new vertex, it can be connected to ANY independent set.
    # To maximize edges, we want to connect to LARGE independent sets.
    # But we also need to ensure the whole graph is not 3-colorable.
    
    # Let's enumerate all independent sets of size >= 3 in Grötzsch.
    # There are 2^11 = 2048 subsets. We can check which are independent.
    
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)
        
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        # Check if mask is independent
        # For each pair in mask, check if not adjacent
        # Faster: for each u in mask, check if adj_core[u] & mask == 0? No, adj_core[u] & (mask without u) == 0
        # Even faster: iterate u in mask, if adj_core[u] & mask: return False? 
        # Wait, adj_core[u] contains neighbors. If any neighbor is in mask, it's not independent.
        # So for all u in mask, (adj_core[u] & mask) == 0 is not quite right because u is not in adj_core[u].
        # But if v is in mask and v is neighbor of u, then v is in adj_core[u].
        # So yes, for all u in mask, (adj_core[u] & mask) == 0 implies no edge between any two vertices in mask.
        
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        
        if is_ind and mask != 0:
            # Store the set of vertices
            vertices = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                vertices.append(u)
                m ^= lsb
            independent_sets.append(vertices)
            
    # Sort by size descending
    independent_sets.sort(key=lambda x: -len(x))
    
    # We have 5 new vertices.
    # We want to assign an independent set to each.
    # Let's try all combinations of the top K independent sets.
    # Top 10 might be enough?
    
    # Also, we need to add edges between new vertices.
    # If new vertex i is connected to set S_i and new vertex j to S_j,
    # can we add edge (i, j)?
    # Yes, if S_i and S_j are disjoint? No, that's not the condition.
    # The condition for triangle-free is:
    # If we add edge (i, j), we must ensure there is no k such that (i,k) and (j,k) are edges.
    # k can be in core or new.
    # If k is in core: k must not be in S_i AND S_j simultaneously?
    # Wait. If (i, k) is edge, k in S_i. If (j, k) is edge, k in S_j.
    # If k is in S_i and S_j, then (i, k) and (j, k) exist.
    # If we add (i, j), then (i, j, k) is a triangle.
    # So S_i and S_j must be disjoint if we add edge (i, j).
    
    # So the problem reduces to:
    # Choose 5 independent sets S_0, ..., S_4 from the list of independent sets of Grötzsch.
    # Choose a subset of edges E_new between the 5 new vertices.
    # Maximize |core_edges| + sum(|S_i|) + |E_new|
    # Subject to:
    # 1. For each (i, j) in E_new, S_i and S_j are disjoint.
    # 2. The graph formed by E_new is triangle-free.
    # 3. The whole graph is not 3-colorable.
    
    # Condition 3 is the tricky one.
    # If we just maximize edges, we might end up with a 3-colorable graph.
    # We need to check 3-colorability.
    
    # Let's first find the best configuration ignoring 3-colorability, then check.
    # If it's 3-colorable, we need to modify it (e.g., remove an edge, or change sets).
    
    # Let's limit the search to the top 30 independent sets (expanded from 20)
    top_sets = independent_sets[:30]
    
    best_edges = None
    best_score = -1
    
    # Search over combinations of top_sets for the 5 new vertices
    # 30^5 = 24.3 million. Too many for full enumeration.
    # Use randomized local search with multiple restarts.
    
    rng = random.Random(42)
    
    # Start with a random assignment of top sets
    current_assignment = [rng.choice(top_sets) for _ in range(5)]
    
    # Fast 3-colorability check using bitmasks and constraint propagation
    def _is_3_colorable_fast(adj):
        """Fast 3-colorability check using bitmasks."""
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        # color[v] = bitmask of available colors (bits 0,1,2)
        # Initially all colors available: 0b111 = 7
        colors = [7] * n
        
        # Process vertices in order
        for idx in range(n):
            v = order[idx]
            
            # For each neighbor already colored, remove that color from v's available set
            for u in range(n):
                if u != v and (adj[v] & (1 << u)):
                    # u has been processed before v (or is v itself, but we skip that)
                    # Actually, we need to track which vertices are colored
                    # This approach doesn't work well with static ordering
                    pass
        
        # Better approach: backtracking with forward checking
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            
            # Determine available colors by checking colored neighbors
            used = 0
            for u in range(n):
                if color[u] != -1 and (adj[v] & (1 << u)):
                    used |= (1 << color[u])
            
            available = 7 ^ used  # 3 colors total: 0b111
            
            # Try each available color
            c = 0
            while available:
                if available & 1:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
                c += 1
                available >>= 1
            
            return False
        
        return backtrack(0)

    def _is_3_colorable(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return _is_3_colorable_fast(adj)

    # Function to evaluate a configuration
    def evaluate(assignment):
        # 1. Build edges to core
        edges = list(core_edges)
        for i, v_new in enumerate(new_vertices):
            for v_core in assignment[i]:
                edges.append([v_new, v_core])
        
        # 2. Determine possible edges between new vertices
        possible_new_edges = []
        for i in range(5):
            for j in range(i+1, 5):
                # Check if S_i and S_j are disjoint
                set_i = set(assignment[i])
                set_j = set(assignment[j])
                if set_i.isdisjoint(set_j):
                    possible_new_edges.append((new_vertices[i], new_vertices[j]))
                    
        # 3. Find max triangle-free subset of possible_new_edges
        # Since there are only 5 vertices, we can brute force subsets of possible_new_edges
        # Max 10 edges. 2^10 = 1024 subsets.
        
        max_new_edges = 0
        best_sub = []
        m = len(possible_new_edges)
        for mask in range(1 << m):
            # Build adjacency for new vertices in this subset
            adj_subset = [0]*5
            for k in range(m):
                if mask & (1 << k):
                    u, v = possible_new_edges[k]
                    lu = u - core_n
                    lv = v - core_n
                    adj_subset[lu] |= (1 << lv)
                    adj_subset[lv] |= (1 << lu)
            
            # Check triangles in adj_subset
            tf = True
            for u in range(5):
                for v in range(u+1, 5):
                    if adj_subset[u] & (1 << v):
                        if adj_subset[u] & adj_subset[v]:
                            tf = False
                            break
                if not tf:
                    break
            
            if tf:
                cnt = bin(mask).count("1")
                if cnt > max_new_edges:
                    max_new_edges = cnt
                    best_sub = []
                    for k in range(m):
                        if mask & (1 << k):
                            best_sub.append(possible_new_edges[k])
        
        edges.extend(best_sub)
        
        # 4. Check if the resulting graph is 3-colorable
        if _is_3_colorable(edges, n):
            return -1 # Invalid
        
        return len(edges)

    # Local Search with multiple restarts
    best_score = -1
    best_assignment = None
    
    # Number of restarts: use remaining time budget
    # Each evaluation takes ~1-5ms, local search iterations take similar
    # Budget: ~100s for search
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:  # Leave 5s margin
            break
            
        # Random start
        assignment = [rng.choice(top_sets) for _ in range(5)]
        
        current_score = evaluate(assignment)
        
        if current_score > best_score:
            best_score = current_score
            best_assignment = list(assignment)
        
        # Local search: try changing one set at a time
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(5):
                if time.time() > deadline - 5.0:
                    break
                for new_set in top_sets:
                    if new_set == assignment[i]:
                        continue
                    old_set = assignment[i]
                    assignment[i] = new_set
                    new_score = evaluate(assignment)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_assignment = list(assignment)
                    else:
                        assignment[i] = old_set
        
    if best_assignment is None:
        # Fallback: return Grötzsch + isolated vertices
        # But we need 16 vertices.
        # Just return the core edges.
        return core_edges
        
    # Reconstruct edges from best_assignment
    final_edges = list(core_edges)
    for i, v_new in enumerate(new_vertices):
        for v_core in best_assignment[i]:
            final_edges.append([v_new, v_core])
            
    # Add new-new edges
    possible_new_edges = []
    for i in range(5):
        for j in range(i+1, 5):
            set_i = set(best_assignment[i])
            set_j = set(best_assignment[j])
            if set_i.isdisjoint(set_j):
                possible_new_edges.append((new_vertices[i], new_vertices[j]))
                
    # Find max TF subset again
    max_new_edges = 0
    best_sub = []
    m = len(possible_new_edges)
    for mask in range(1 << m):
        adj_subset = [0]*5
        for k in range(m):
            if mask & (1 << k):
                u, v = possible_new_edges[k]
                lu = u - core_n
                lv = v - core_n
                adj_subset[lu] |= (1 << lv)
                adj_subset[lv] |= (1 << lu)
        
        tf = True
        for u in range(5):
            for v in range(u+1, 5):
                if adj_subset[u] & (1 << v):
                    if adj_subset[u] & adj_subset[v]:
                        tf = False
                        break
            if not tf:
                break
        
        if tf:
            cnt = bin(mask).count("1")
            if cnt > max_new_edges:
                max_new_edges = cnt
                best_sub = []
                for k in range(m):
                    if mask & (1 << k):
                        best_sub.append(possible_new_edges[k])
                        
    final_edges.extend(best_sub)
    
    # Final polish: try to add any remaining edges that maintain triangle-free and 4-chromatic
    # Build adjacency from final_edges
    adj_final = [0] * n
    for u, v in final_edges:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)
    
    # Try adding each missing edge
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_final[u] & (1 << v)):
                # Check if adding (u,v) maintains triangle-free
                # No common neighbor
                if not (adj_final[u] & adj_final[v]):
                    # Try adding this edge
                    test_edges = final_edges + [[u, v]]
                    # Check 3-colorability
                    if not _is_3_colorable(test_edges, n):
                        final_edges.append([u, v])
                        adj_final[u] |= (1 << v)
                        adj_final[v] |= (1 << u)
    
    return final_edges
