"""
Search for large C5-free graphs with independence number < 9.
Uses a greedy edge-removal heuristic starting from a dense graph,
then tries to add vertices/edges while maintaining constraints.
"""
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 115.0  # seconds, leaving 5s safety margin for the 120s limit


def has_c5(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) contains a 5-cycle."""
    # For each pair of vertices, check if they can be part of a 5-cycle
    # A 5-cycle exists if there are vertices v0,v1,v2,v3,v4 such that
    # all consecutive pairs (and v4-v0) are edges.
    # Efficient check: for each vertex, look at its neighbors and check
    # for paths of length 4 between two neighbors that don't go through them.
    
    # Better approach: enumerate all 5-tuples is too slow.
    # Use: a 5-cycle exists iff there exist two vertices u,v that share
    # at least two common neighbors, and those common neighbors are not adjacent
    # to each other... no that's for triangles.
    
    # Actually, let's use a different approach:
    # A C5 exists if there's a path of length 4 between two adjacent vertices
    # where the internal vertices are all distinct and not the endpoints.
    
    # Most efficient: for each vertex v, consider its neighbors N(v).
    # A 5-cycle through v: v - a - b - c - d - v where a,d in N(v), a!=d
    # and a-b, b-c, c-d are edges, and b,c not in N(v) and b!=c.
    # Also b != d and c != a (to avoid shorter cycles or repeated vertices).
    
    # Let's try: for each vertex v, for each pair (a, d) of neighbors of v with a < d:
    #   check if there exists b, c such that a-b, b-c, c-d are edges,
    #   b != c, b != v, c != v, b != a, b != d, c != a, c != d.
    
    # This is O(n * deg^2 * n^2) in worst case. For n=35, deg~17, that's manageable.
    
    for v in range(n):
        neighbors = [u for u in range(n) if adj[v][u]]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                d = neighbors[j]
                # Find common "middle" paths of length 2 from a to d
                # a-b-c-d: b adjacent to a, c adjacent to d, b adjacent to c
                # b, c not in {v, a, d}, b != c
                for b in range(n):
                    if b == v or b == a or b == d:
                        continue
                    if not adj[a][b]:
                        continue
                    for c in range(n):
                        if c == v or c == a or c == d or c == b:
                            continue
                        if not adj[d][c]:
                            continue
                        if not adj[b][c]:
                            continue
                        return True
    return False


def has_independent_set(adj, n, k=9):
    """Check if there's an independent set of size k. Use greedy + backtracking."""
    # Greedy: repeatedly pick vertex with minimum degree in remaining graph
    # Then verify with a simple search if needed.
    
    # First try a greedy heuristic
    remaining = set(range(n))
    chosen = []
    while remaining and len(chosen) < k:
        # Pick vertex with fewest neighbors in remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in remaining if adj[v][u])
            if deg < best_deg:
                best_deg = deg
                best_v = v
        chosen.append(best_v)
        remaining.discard(best_v)
        # Remove all neighbors of best_v from remaining
        remaining -= {u for u in remaining if adj[best_v][u]}
    
    if len(chosen) >= k:
        return True
    
    # If greedy didn't find one, do a bounded backtracking search
    # This is expensive, so limit it
    def backtrack(remaining, chosen_set):
        if len(chosen_set) >= k:
            return True
        if not remaining:
            return False
        # Upper bound: can we reach k?
        if len(chosen_set) + len(remaining) < k:
            return False
        # Pick vertex with minimum degree
        v = min(remaining, key=lambda x: sum(1 for u in remaining if adj[x][u]))
        # Try including v
        new_remaining = remaining - {v} - {u for u in remaining if adj[v][u]}
        if backtrack(new_remaining, chosen_set | {v}):
            return True
        # Try excluding v
        if backtrack(remaining - {v}, chosen_set):
            return True
        return False
    
    # Only do backtracking if n is small enough
    if n <= 30:
        return backtrack(set(range(n)), set())
    return False


def count_c5_paths(adj, n):
    """Count number of (not necessarily induced) 5-cycles for optimization."""
    count = 0
    for v in range(n):
        neighbors = [u for u in range(n) if adj[v][u]]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                d = neighbors[j]
                for b in range(n):
                    if b == v or b == a or b == d:
                        continue
                    if not adj[a][b]:
                        continue
                    for c in range(n):
                        if c == v or c == a or c == d or c == b:
                            continue
                        if not adj[d][c]:
                            continue
                        if adj[b][c]:
                            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Strategy: Start with a complete graph on n vertices, remove edges to eliminate C5s
    # while keeping independence number < 9.
    # 
    # Key insight: C5-free graphs have limited structure. 
    # Known constructions: 
    # - Disjoint union of cliques (current: 7xK4 = 28 vertices, alpha=7)
    # - We want to push to 32+ vertices.
    #
    # Approach: Use a greedy/randomized search.
    # 1. Start with a candidate graph (e.g., union of cliques with varying sizes)
    # 2. Try to add vertices and connect them optimally
    # 3. Remove edges to break C5s while maintaining alpha < 9
    
    best_n = 0
    best_edges = []
    
    # Try different base constructions
    # Construction 1: Union of cliques
    # K_a + K_b + ... where sum = n, alpha = number of components
    # We need alpha < 9, so at most 8 components.
    # To maximize n with 8 components: make them as large as possible.
    # But large cliques create more C5 risk when we add cross-edges.
    
    # Actually, disjoint union of cliques is always C5-free (no cycles at all between components).
    # So 8 copies of K4 = 32 vertices, alpha = 8. This is valid!
    # Wait, let me verify: 8 * K4 has 32 vertices, no cycles (only triangles within each K4),
    # independence number = 8 (pick one from each K4).
    # 8 < 9, so no independent set of size 9. Valid!
    
    # Can we do 33? 8 components, need sum of sizes = 33, max component size...
    # If we use 7xK4 + 1xK1: n=29, alpha=8. 
    # 8xK4 = 32. To get 33, we'd need 9 components (alpha=9, invalid) or
    # one component of size 5: 7xK4 + K5 = 33, alpha=8. But K5 is fine (no C5 in a clique).
    # Wait, K5 has no 5-cycle? A 5-cycle requires 5 distinct vertices with edges forming a cycle.
    # In K5, there ARE 5-cycles! K5 contains C5 as a subgraph.
    # So we can't use K5 or larger cliques.
    
    # Max clique size without C5: K4 (K4 has triangles but no 4-cycles or 5-cycles).
    # Actually K4: vertices 0,1,2,3. Is there a 5-cycle? No, only 4 vertices.
    # K5: vertices 0,1,2,3,4. Cycle 0-1-2-3-4-0 is a 5-cycle. Yes, K5 contains C5.
    
    # So max clique is K4. With 8 components of K4: n=32, alpha=8. Valid.
    # Can we do better? What about non-clique components?
    
    # A component that is C5-free and has independence number 1 is a clique (max K4).
    # A component with independence number 2 could be larger.
    # E.g., K4 minus an edge: 4 vertices, alpha=2, no C5.
    # Or a larger C5-free graph with small alpha.
    
    # Let's try: 8 components, each with alpha <= 1 (cliques, max size 4) gives 32.
    # Or mix: some components with alpha=2 but larger size.
    # E.g., a C5-free graph on 5 vertices with alpha=2: 
    #   K4 + isolated vertex? No, that has alpha=2 (the isolated vertex + one from K4 that's non-adjacent... wait in K4 all are adjacent).
    #   K4 with an extra vertex connected to 3 of the 4: 5 vertices. Alpha: the isolated-ish vertex + the one K4 vertex not connected to it = 2. C5? The 5 vertices: 4-clique plus one connected to 3. Any 5-cycle? Need all 5 in a cycle. The extra vertex has degree 3, so it can be in a cycle of length at most 4 (via its 3 neighbors in K4). Actually a 5-cycle would need to use all 5 vertices. Path through the extra vertex: it connects to 3 K4 vertices. A 5-cycle using all 5: v_extra - a - b - c - d - v_extra? But v_extra only connects to 3 of {a,b,c,d}. If it connects to a,b,c, then a 5-cycle would need v_extra-a-?-?-c-v_extra, but that's only using 5 vertices if the middle two are distinct from a,c,v_extra. In K4, a,b,c,d form a clique, so a-b, b-c, etc. all exist. 5-cycle: v_extra-a-b-d-c-v_extra? Check: v_extra-a (yes), a-b (yes, in K4), b-d (yes, in K4), d-c (yes, in K4), c-v_extra (yes). That's a 5-cycle! So this graph has a C5.
    
    # So we need to be careful. Let's stick with the safe construction: 8xK4 = 32 vertices.
    
    # Now, can we add a 33rd vertex? We'd need to connect it to some existing vertices.
    # The new vertex must not create a C5 and must not create an independent set of size 9.
    # Currently alpha=8. Adding a vertex: if it's isolated, alpha becomes 9 (bad).
    # So it must connect to at least one vertex in each "independent set choice".
    # Actually, we need that every set of 9 vertices includes at least one edge.
    # Equivalently, the new vertex must "hit" all maximum independent sets of the old graph.
    # In 8xK4, max independent sets are exactly one vertex from each K4 (8 choices per component, 4^8 total).
    # The new vertex v must be adjacent to at least one vertex in every such independent set.
    # That means: for every choice (v1,...,v8) where vi is in component i, v must be adjacent to some vi.
    # Equivalently, the non-neighbors of v must not contain a full independent set (one from each component).
    # If v is non-adjacent to all vertices in component i, then we can pick any vertex from component i.
    # So we need: the set of components where v is non-adjacent to ALL vertices must have size < 8.
    # I.e., v must be adjacent to at least one vertex in at least 7 of the 8 components.
    # But also, v must not create a C5.
    
    # Let's try: add vertex 32, connect it to one vertex in each of 8 components.
    # Say connect to vertex 0 of each K4.
    # Check C5: Could there be a 5-cycle involving vertex 32?
    # 32 is connected to {0, 4, 8, 12, 16, 20, 24, 28} (one from each K4).
    # A 5-cycle through 32: 32 - a - b - c - d - 32 where a,d are neighbors of 32.
    # a and d are in different K4s (say component i and component j).
    # b must be adjacent to a: b is in component i (since a's only neighbors are in its K4, plus 32).
    # c must be adjacent to d: c is in component j.
    # b-c must be an edge: but b is in component i and c is in component j, different components, no edge.
    # So no 5-cycle! 
    # 
    # Wait, b could also be 32? No, b is distinct from 32 in a simple cycle.
    # Could b be in a third component? No, b must be adjacent to a, and a only has neighbors in its own K4 and 32.
    # So b must be in component i (or b=32, excluded).
    # Similarly c must be in component j.
    # b-c edge only if same component, but i != j. So no 5-cycle.
    
    # What about a 5-cycle NOT through 32? The original 8xK4 has no cycles at all, so no.
    
    # So 9xK4... wait, I added one vertex connected to 8 others. That's not 9xK4.
    # Let me reconsider. We have 8 components of K4 (32 vertices). Add vertex 32.
    # Connect vertex 32 to one vertex in each component.
    # Now n=33. Check alpha: is there an independent set of size 9?
    # An independent set can include vertex 32 and 8 others, one from each component, 
    # but those 8 others must not be adjacent to 32.
    # Vertex 32 is adjacent to vertex 0 of each component. So we must pick non-zero vertices.
    # From each component, we can pick any of vertices 1,2,3 (3 choices).
    # So {32, v1, v2, ..., v8} where vi is in {1,2,3} of component i is an independent set of size 9!
    # That's bad.
    
    # To prevent this, vertex 32 must be adjacent to ALL vertices in at least one component.
    # If 32 is adjacent to all 4 vertices of component 0, then any independent set containing 32
    # cannot contain any vertex from component 0, so it can have at most 8 vertices total (32 + 7 from other components).
    # But wait, we need independent set of size 9. With 32 and no vertices from component 0,
    # we can pick at most 1 from each of the other 7 components = 7 more, total 8. Not 9.
    # Without 32, max independent set is 8 (one from each component).
    # So alpha = 8. 
    #
    # But now check C5: 32 is connected to all 4 vertices of component 0.
    # 5-cycle through 32: 32 - a - b - c - d - 32.
    # a, d are neighbors of 32, so in component 0.
    # b adjacent to a: b in component 0.
    # c adjacent to d: c in component 0.
    # b-c edge: both in component 0, yes (it's a K4).
    # So 32-a-b-c-d-32 is a 5-cycle if a,b,c,d are 4 distinct vertices in component 0.
    # Component 0 has exactly 4 vertices, so a,b,c,d are all 4 of them.
    # 32-a-b-c-d-32: this is a 5-cycle! Bad.
    
    # So connecting to all 4 in one component creates a C5.
    # What if 32 connects to 3 vertices in component 0 and 1 vertex in each other component?
    # Then independent set: 32 + one from each of components 1-7 (7 vertices) + one from component 0 not adjacent to 32 (1 choice). Total = 9. Bad.
    
    # Hmm. So adding a single vertex to 8xK4 seems hard.
    
    # What about a completely different construction?
    # Let's try: 7xK4 + one larger C5-free graph with alpha=1... but max clique is K4.
    # Or: use a different base.
    
    # Alternative: Start with 32 vertices (8xK4) and see if we can restructure.
    # Or try: 9 components where one has alpha=1 (K4, 4 vertices) and others are smaller,
    # but 9 components means alpha >= 9 if each has alpha >= 1. Bad.
    
    # Let me just go with the safe 32-vertex construction for now.
    # 8 disjoint K4s.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = edges
    
    # Now try to improve: can we get to 33, 34, 35?
    # Let's try a randomized search: start from 8xK4, try to add a vertex with random connections,
    # check constraints, iterate.
    
    random.seed(42)
    
    def try_add_vertex(adj, n, max_iter=200):
        """Try to add a vertex to the graph. Return new adj, n if successful."""
        for _ in range(max_iter):
            new_n = n + 1
            # Randomly choose which components to connect to and which vertices
            # Component structure: vertices 4*b to 4*b+3 are component b
            num_components = n // 4  # This is only valid for the 8xK4 structure
            if num_components * 4 != n:
                return None, None  # Structure changed
            
            # Choose for each component: connect to 0, 1, 2, or 3 vertices
            # Must connect to at least 3 vertices in total to avoid independent set issues
            conn_pattern = []
            for b in range(num_components):
                # Number of connections to this component
                k = random.choice([0, 1, 1, 2, 2, 3])
                if k > 0:
                    verts = random.sample(range(4), k)
                else:
                    verts = []
                conn_pattern.append(verts)
            
            # Check: is there an independent set of size 9?
            # The new vertex is connected to vertices in conn_pattern.
            # Non-neighbors of new vertex: all vertices in components where conn_pattern is empty,
            # plus non-connected vertices in other components.
            # An independent set of size 9 can include the new vertex + 8 others (one per component,
            # avoiding neighbors of new vertex).
            # For component b, available vertices = 4 - len(conn_pattern[b])
            # We need to pick one from each component (or skip some).
            # Max independent set including new vertex: 1 + sum over components of (1 if available > 0 else 0)
            # But we can only pick at most 1 per component.
            # So max size = 1 + (number of components with available > 0)
            num_avail = sum(1 for b in range(num_components) if len(conn_pattern[b]) < 4)
            if 1 + num_avail >= 9:
                continue  # Independent set of size 9 exists
            
            # Check C5
            new_adj = [row[:] for row in adj]
            for row in new_adj:
                row.append(False)
            new_adj.append([False] * (n + 1))
            
            for b in range(num_components):
                for v in conn_pattern[b]:
                    idx = 4 * b + v
                    new_adj[n][idx] = True
                    new_adj[idx][n] = True
            
            if has_c5(new_adj, new_n):
                continue
            
            # Also need to check: did we accidentally create a C5 not involving the new vertex?
            # No, the old graph had no C5, and we only added edges from the new vertex.
            # A C5 not involving the new vertex would be in the old graph, which has none.
            # A C5 involving the new vertex: checked above.
            
            # Check independent set more carefully
            if has_independent_set(new_adj, new_n, 9):
                continue
            
            return new_adj, new_n
        
        return None, None
    
    # Build adjacency for 8xK4
    adj = [[False] * n for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(4):
                if i != j:
                    adj[base + i][base + j] = True
    
    # Try to add vertices
    current_n = n
    current_adj = adj
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        new_adj, new_n = try_add_vertex(current_adj, current_n, max_iter=500)
        if new_n is not None:
            current_adj = new_adj
            current_n = new_n
            # Update best
            if current_n > best_n:
                best_n = current_n
                # Rebuild edges
                best_edges = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if current_adj[i][j]:
                            best_edges.append([i, j])
    
    # Final verification
    if best_n > 0:
        # Verify no C5
        adj_check = [[False] * best_n for _ in range(best_n)]
        for e in best_edges:
            adj_check[e[0]][e[1]] = True
            adj_check[e[1]][e[0]] = True
        
        if has_c5(adj_check, best_n):
            # Fallback to safe construction
            best_n = 32
            best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
        
        if has_independent_set(adj_check, best_n, 9):
            best_n = 32
            best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    return {'n': best_n, 'edges': best_edges}
