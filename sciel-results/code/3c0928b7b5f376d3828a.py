"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }
    
    # Try to find a valid graph with n > 32
    # We'll try n=33, 34, 35 in order, trying to find any valid graph
    
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle. Returns True if found."""
        # For each pair of vertices, check if they have two common neighbors
        # that are not adjacent to each other and form a path of length 2
        # Actually, a 5-cycle exists iff there exist vertices v0,v1,v2,v3,v4 
        # such that v0-v1-v2-v3-v4-v0 are all edges.
        # 
        # A more efficient check: A graph has a C5 if and only if there exist
        # three vertices a, b, c such that a-b and b-c are edges, a and c are
        # NOT adjacent, and a and c have a common neighbor d that is not b.
        # Wait, that's not quite right either.
        #
        # Let me think again. A 5-cycle v0-v1-v2-v3-v4-v0 means:
        # v0~v1, v1~v2, v2~v3, v3~v4, v4~v0
        # and no other edges among these 5 vertices (for an induced C5),
        # but for a subgraph C5, we just need these 5 edges to exist.
        #
        # To check for any C5 subgraph:
        # For each ordered triple (a, b, c) where a~b and b~c and a not~ c:
        #   For each d ~ c, d != b, d != a:
        #     If d ~ a, then we have a-b-c-d-a which is a 4-cycle, not 5.
        #     We need a 5th vertex e such that e~d and e~a, e not in {b,c}.
        #
        # This is getting complex. Let me use a different approach.
        #
        # For n up to 35, we can check all 5-tuples but that's O(n^5) which is 35^5 ≈ 52M.
        # That might be too slow if done naively in Python.
        #
        # Better: use the fact that C5 is a specific structure. 
        # A graph contains C5 as a subgraph iff there exist distinct vertices
        # v0,v1,v2,v3,v4 with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0).
        #
        # We can iterate over pairs (v0, v1) that are edges, then for each such pair,
        # look at common neighbors of v0 and v1... no wait, they don't need to share neighbors.
        #
        # Let me think of it as: pick an edge (u,v). Then we need a path of length 3 from u to v
        # through 3 other vertices, i.e., u-a-b-c-v where all edges exist and a,b,c distinct from u,v.
        # Wait no, the cycle is u-v-...-u. So if we fix edge (u,v), we need a path of length 4
        # from u to v not using edge (u,v) directly. So u-a-b-c-v where u~a, a~b, b~c, c~v,
        # and a,b,c distinct from u,v and from each other.
        
        # For efficiency with n<=35:
        # For each edge (u,v):
        #   For each a in N(u) \ {v}:
        #     For each b in N(a) \ {u, v}:
        #       For each c in N(b) \ {u, a, v}:
        #         if c in N(v) and c != u:
        #           return True
        
        # This is roughly sum over edges of |N(u)|*|N(a)|*|N(b)| which could be large.
        # For dense graphs this is bad, but our graphs should be somewhat sparse to avoid C5.
        
        # Alternative: use bitsets for adjacency.
        adj_set = [set(adj[i]) for i in range(n)]
        
        for u in range(n):
            for v in adj_set[u]:
                if v <= u:
                    continue
                # Find all paths of length 3 from u to v (i.e., u-a-b-v where u~a, a~b, b~v)
                # Wait, we need u-a-b-c-v, that's length 4.
                # Let me reconsider. Cycle is u-v-x-y-z-u. So from u to v, we need path u-x-y-z-v of length 3 edges? No.
                # u-v-x-y-z-u: edges are (u,v), (v,x), (x,y), (y,z), (z,u).
                # So if we fix edge (u,v), we need a path from v to u of length 4 through intermediate vertices.
                # v-x-y-z-u where v~x, x~y, y~z, z~u.
                
                # So: for edge (u,v), check if there exist x,y,z distinct from u,v and each other
                # such that v~x, x~y, y~z, z~u.
                
                for x in adj_set[v]:
                    if x == u:
                        continue
                    for y in adj_set[x]:
                        if y == u or y == v:
                            continue
                        for z in adj_set[y]:
                            if z == u or z == v or z == x:
                                continue
                            if u in adj_set[z]:
                                return True
        return False
    
    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9."""
        # Use greedy/backtracking with bitsets
        # For n<=35, we can use a recursive approach with pruning
        
        adj_set = [set(adj[i]) for i in range(n)]
        
        def can_extend(current_set, remaining_candidates, size_needed):
            """Try to extend current_set with vertices from remaining_candidates to get independent set of size size_needed."""
            if size_needed == 0:
                return True
            if len(remaining_candidates) < size_needed:
                return False
            
            # Pick a vertex from remaining_candidates
            # Heuristic: pick the one with fewest connections to remaining_candidates
            best_v = None
            best_count = float('inf')
            for v in remaining_candidates:
                # Count how many in remaining_candidates are NOT adjacent to v
                non_adj_count = sum(1 for u in remaining_candidates if u != v and u not in adj_set[v])
                if non_adj_count < best_count:
                    best_count = non_adj_count
                    best_v = v
                if best_count == 0:
                    break
            
            if best_v is None:
                return False
            
            v = best_v
            # Option 1: include v
            new_remaining = {u for u in remaining_candidates if u != v and u not in adj_set[v]}
            if can_extend(current_set | {v}, new_remaining, size_needed - 1):
                return True
            
            # Option 2: exclude v
            new_remaining2 = {u for u in remaining_candidates if u != v}
            if can_extend(current_set, new_remaining2, size_needed):
                return True
            
            return False
        
        return can_extend(set(), set(range(n)), 9)
    
    def is_valid(n, adj):
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True
    
    def adj_to_edges(adj, n):
        """Convert adjacency matrix to edge list."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Try to construct graphs with n = 33, 34, 35
    # Strategy: start from 8xK4 (n=32) and try to add vertices by modifying the graph
    # Or: random search with local improvement
    
    best_result = known_32
    
    # Try random constructions for n=33,34,35
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Start with 8xK4 structure and add extra vertices
        # For n=33: 8xK4 + 1 extra vertex
        # The extra vertex needs to be connected such that:
        # 1. No C5 is created
        # 2. No independent set of 9 is created
        
        # Let's try random edge additions/modifications
        found = False
        
        # Approach: start with 8xK4, add vertex 32 (for n=33)
        # Connect it to some vertices. To avoid C5, we need to be careful.
        # To avoid independent set of 9, the vertex should be connected to enough vertices
        # that no 9-set avoids it.
        
        # Actually, let's try a different approach: random graph search
        # Generate random graphs with appropriate density and check validity
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Generate a random graph
            # Density: for C5-free graphs, Turan's theorem gives max edges
            # For C5-free, max edges is roughly O(n^(3/2))
            # For n=33, that's about 33^(1.5) ≈ 189 edges
            # But we also need alpha < 9
            
            # Let's try various densities
            p = random.uniform(0.2, 0.5)
            
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Quick check: count edges and check basic properties
            edge_count = sum(sum(row) for row in adj) // 2
            
            # Check C5-free (this is the expensive check)
            if has_c5(adj, target_n):
                continue
            
            # Check independent set
            if has_independent_set_9(adj, target_n):
                continue
            
            # Valid!
            edges = adj_to_edges(adj, target_n)
            best_result = {'n': target_n, 'edges': edges}
            found = True
            break
        
        if found:
            break
    
    return best_result

# EVOLVE-BLOCK-END
