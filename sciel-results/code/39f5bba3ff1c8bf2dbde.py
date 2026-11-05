"""Improved K4-free graph construction for n=18."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _is_k4_free(adj_sets):
    n = len(adj_sets)
    for i in range(n):
        for j in range(i+1, n):
            if j in adj_sets[i]:
                common = adj_sets[i] & adj_sets[j]
                for k in common:
                    if k <= j:
                        continue
                    # Check if there exists l > k such that l is adjacent to i, j, k
                    # i.e., l in adj_sets[i] & adj_sets[j] & adj_sets[k]
                    triple_common = adj_sets[i] & adj_sets[j] & adj_sets[k]
                    for l in triple_common:
                        if l > k:
                            return False
    return True

def _max_triangle_free_size(adj_sets, timeout=5.0):
    """
    Find the maximum size of a triangle-free induced subgraph.
    Triangle-free means no 3 vertices form a triangle (K3).
    Uses branch and bound.
    """
    n = len(adj_sets)
    start_time = time.time()
    
    # Precompute: for each pair (u,v), the set of common neighbors
    # common[u][v] = adj_sets[u] & adj_sets[v]
    common = {}
    for u in range(n):
        for v in range(u+1, n):
            if v in adj_sets[u]:
                common[(u,v)] = adj_sets[u] & adj_sets[v]
            else:
                common[(u,v)] = set()
    
    # We want to find the largest subset S of vertices such that no triangle is contained in S.
    # A triangle in S is a set {a,b,c} where a,b,c in S and all three pairs are edges.
    
    # Branch and bound approach:
    # State: (candidate_set, excluded_set)
    # We build the set incrementally.
    # Upper bound: |candidate_set|
    
    best_size = 0
    best_set = set()
    
    # Use a recursive branch and bound
    # Order vertices by degree (descending) to find large triangles early? 
    # Actually, we want to find a large triangle-free set.
    # Let's use a different approach: iterate over all subsets? No, 2^18 is too big.
    
    # Branch and bound on the set S.
    # Start with all vertices as candidates.
    # Pick a vertex v.
    # Option 1: Include v in S. Then we must exclude any pair (u,w) in S such that u,w,v form a triangle.
    #   Specifically, if we include v, then for any edge (u,w) in S, if u,w,v form a triangle, we can't have both u and w in S.
    #   This is getting complicated.
    
    # Alternative: Use the complement. We want to find the largest set with no K3.
    # This is equivalent to finding the independence number of the "triangle graph"? No.
    
    # Let's use a standard branch and bound for maximum independent set in a 3-uniform hypergraph?
    # The hyperedges are the triangles. We want the largest set of vertices containing no hyperedge.
    # This is the Maximum Independent Set problem in a 3-uniform hypergraph.
    
    # List all triangles in the graph.
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if j in adj_sets[i]:
                common_ij = adj_sets[i] & adj_sets[j]
                for k in common_ij:
                    if k > j:
                        triangles.append((i, j, k))
    
    # Now we have a list of triangles. We want to find the largest subset of vertices that contains no triangle.
    # This is the maximum independent set in the hypergraph (V, triangles).
    
    # Branch and bound:
    # State: (available_vertices, current_set)
    # Upper bound: len(available_vertices) + len(current_set)
    
    # To speed up, we can use a greedy heuristic to get a good initial lower bound.
    # Greedy: repeatedly pick a vertex that is in the fewest remaining triangles.
    
    def greedy_independent_set(vertices):
        """Greedy algorithm for maximum independent set in hypergraph."""
        vertices = set(vertices)
        remaining_triangles = [set(t) for t in triangles]
        result = set()
        
        while vertices:
            # Find vertex in fewest remaining triangles
            min_count = float('inf')
            chosen = None
            for v in vertices:
                count = sum(1 for t in remaining_triangles if v in t)
                if count < min_count:
                    min_count = count
                    chosen = v
            
            if chosen is None:
                break
                
            result.add(chosen)
            vertices.remove(chosen)
            
            # Remove all triangles containing chosen
            remaining_triangles = [t for t in remaining_triangles if chosen not in t]
            
            if not remaining_triangles:
                # No more triangles, add all remaining vertices
                result.update(vertices)
                break
                
        return result
    
    # Get initial lower bound
    initial_set = greedy_independent_set(range(n))
    best_size = len(initial_set)
    best_set = initial_set
    
    # Branch and bound
    # Use a stack for iterative DFS to avoid recursion depth issues
    # State: (available_vertices, current_set)
    # We order vertices to branch on.
    
    # For efficiency, we represent available_vertices as a frozenset or sorted tuple.
    # Since n=18, we can use a bitmask.
    
    # Let's use bitmask representation.
    # vertices is a bitmask of available vertices.
    # current_set is a bitmask of chosen vertices.
    
    def greedy_bitmask(avail_mask):
        """Greedy on bitmask."""
        result = 0
        remaining = [t for t in triangles]  # List of (i,j,k)
        # Convert triangles to bitmasks
        tri_masks = [1<<i | 1<<j | 1<<k for i,j,k in triangles]
        
        while avail_mask:
            # Find vertex in fewest remaining triangles
            min_count = float('inf')
            chosen = -1
            for v in range(n):
                if avail_mask & (1<<v):
                    count = sum(1 for tm in tri_masks if tm & (1<<v))
                    if count < min_count:
                        min_count = count
                        chosen = v
            
            if chosen == -1:
                break
                
            result |= (1<<chosen)
            avail_mask &= ~(1<<chosen)
            
            # Remove triangles containing chosen
            tri_masks = [tm for tm in tri_masks if not (tm & (1<<chosen))]
            
            if not tri_masks:
                result |= avail_mask
                break
                
        return result
    
    # Re-initialize with bitmask
    initial_mask = greedy_bitmask((1<<n)-1)
    initial_count = bin(initial_mask).count('1')
    if initial_count > best_size:
        best_size = initial_count
        best_set = set(i for i in range(n) if initial_mask & (1<<i))
    
    # Branch and bound
    # Stack entries: (avail_mask, current_mask)
    # Upper bound: popcount(avail_mask) + popcount(current_mask)
    
    stack = [((1<<n)-1, 0)]
    
    while stack:
        avail_mask, current_mask = stack.pop()
        
        # Upper bound
        ub = bin(avail_mask).count('1') + bin(current_mask).count('1')
        if ub <= best_size:
            continue
        
        # Check time
        if time.time() - start_time > timeout:
            break
        
        if avail_mask == 0:
            continue
        
        # Pick a vertex to branch on
        # Choose a vertex that is in many triangles among available ones?
        # Or just pick the lowest set bit.
        # Let's pick the lowest set bit for simplicity, but maybe a heuristic is better.
        
        # Heuristic: pick a vertex that is in the most triangles within avail_mask
        # This might help prune sooner if we exclude it.
        
        # For now, let's just pick the lowest set bit.
        v = (avail_mask & -avail_mask).bit_length() - 1
        
        # Option 1: Include v
        new_avail = avail_mask & ~(1<<v)
        new_current = current_mask | (1<<v)
        
        # If including v creates a triangle with two other vertices in current_mask,
        # then we cannot include v.
        # Check if there exist u,w in current_mask such that (u,v,w) is a triangle.
        # This means u and w are in current_mask, and (u,w) is an edge, and (u,v) and (w,v) are edges.
        # Since we are building a triangle-free set, if we add v, we must ensure no triangle is formed.
        # A triangle is formed if there are u,w in current_mask such that u-w, u-v, w-v are all edges.
        
        # Let's check this condition.
        can_include = True
        # Iterate over pairs in current_mask? That's expensive.
        # Instead, check for each u in current_mask adjacent to v, if there is w in current_mask adjacent to both u and v.
        # i.e., w in adj_sets[u] & adj_sets[v] & current_mask
        for u in range(n):
            if current_mask & (1<<u) and u in adj_sets[v]:
                common_uv = adj_sets[u] & adj_sets[v]
                if common_uv & current_mask:
                    can_include = False
                    break
        
        if can_include:
            stack.append((new_avail, new_current))
        
        # Option 2: Exclude v
        stack.append((new_avail, current_mask))
        
    return best_size

def _construct_greedy_k4_free():
    """Construct a K4-free graph using greedy edge addition."""
    n = 18
    
    # We want to find a K4-free graph that minimizes the max triangle-free set.
    # Start with a random graph and use local search?
    # Or start with a known structure.
    
    # Let's try to construct a graph based on a known extremal example or random search.
    
    # Random search with local improvement
    random.seed(42)
    
    best_graph = None
    best_score = float('inf')
    
    # Time limit for this function
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s for evaluation and safety
    
    # Try different random graphs
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random graph
        adj = [[False]*n for _ in range(n)]
        # Start with a sparse graph and add edges
        # Or start with a dense graph and remove edges to make it K4-free
        
        # Let's start with a random graph with p=0.5
        adj_sets = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
                    adj_sets[i].add(j)
                    adj_sets[j].add(i)
        
        # Remove edges to make it K4-free
        # Find K4s and remove an edge from each
        changed = True
        while changed:
            changed = False
            # Find a K4
            for i in range(n):
                for j in range(i+1, n):
                    if j in adj_sets[i]:
                        common_ij = adj_sets[i] & adj_sets[j]
                        for k in common_ij:
                            if k > j:
                                common_ijk = adj_sets[i] & adj_sets[j] & adj_sets[k]
                                for l in common_ijk:
                                    if l > k:
                                        # Found K4: i,j,k,l
                                        # Remove edge (i,j)
                                        adj[i][j] = False
                                        adj[j][i] = False
                                        adj_sets[i].discard(j)
                                        adj_sets[j].discard(i)
                                        changed = True
                                        break
                                if changed:
                                    break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        # Now adj is K4-free
        # Evaluate
        score = _max_triangle_free_size(adj_sets, timeout=2.0)
        
        if score < best_score:
            best_score = score
            best_graph = [s.copy() for s in adj_sets]
        
        # Local search: try to add edges
        # If adding an edge doesn't create a K4, it might reduce the triangle-free set
        # But it might also increase it.
        # Let's try to add edges that are in many triangles?
        
        # For now, just random restarts
    
    if best_graph is None:
        # Fallback: Turán graph T(18,3)
        best_graph = [set() for _ in range(n)]
        parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        best_graph[a].add(b)
                        best_graph[b].add(a)
    
    edges = []
    for i in range(n):
        for j in best_graph[i]:
            if i < j:
                edges.append([i, j])
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a better graph
    result = _construct_greedy_k4_free()
    
    # Verify K4-free
    n = 18
    adj_sets = [set() for _ in range(n)]
    for u, v in result:
        if u != v:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    if not _is_k4_free(adj_sets):
        # Fallback to Turán
        result = [[a, b] for i in range(3) for j in range(i+1, 3)
                  for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen, but just in case
        result = [[a, b] for i in range(3) for j in range(i+1, 3)
                  for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    return result

# EVOLVE-BLOCK-END
