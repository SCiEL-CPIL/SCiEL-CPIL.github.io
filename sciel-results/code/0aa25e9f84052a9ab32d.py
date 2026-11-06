import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        """Return list of triangle bitmasks."""
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def evaluate(adj):
        """
        Find the size of the largest triangle-free induced subgraph.
        A k-subset is triangle-free iff it contains no triangle.
        We check subsets from largest to smallest (k from n down to 1).
        For each k, enumerate all C(n,k) subsets and check if any is triangle-free.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Precompute: for each subset, check if it contains any triangle
        # Use a faster approach: for each vertex, precompute its triangle membership
        # Then use DP or direct enumeration
        
        # Direct enumeration from k = n down
        # For k = n: check if whole graph is triangle-free
        if not triangles:
            return n
        
        # Enumerate subsets of size k from n down to 1
        # Use combinatorial generation with pruning
        
        # Precompute for each vertex which triangles it belongs to
        # tri_count[v] = number of triangles containing v
        # tri_mask[v] = bitmask of triangles containing v (if num_tris <= 64)
        
        num_tris = len(triangles)
        
        # For small num_tris, use bitmask approach
        # For each subset, compute union of triangle masks of its vertices
        # A subset is triangle-free iff no triangle is fully contained
        
        # Better: precompute for each pair (u,v) whether they form an edge with some w
        # Actually, let's just enumerate subsets and check directly
        
        # Key insight: enumerate subsets of size k, check if triangle-free
        # A subset S is triangle-free iff for every triangle t, t & S != t
        
        # For efficiency, precompute triangle membership per vertex
        # vertex_in_tri[v] = bitmask of triangles that contain v
        
        if num_tris <= 20:  # Use bitmask if manageable
            vertex_in_tri = [0] * n
            for i, tri in enumerate(triangles):
                bits = tri
                while bits:
                    lsb = bits & -bits
                    v = lsb.bit_length() - 1
                    vertex_in_tri[v] |= (1 << i)
                    bits ^= lsb
            
            all_tris_mask = (1 << num_tris) - 1
            
            # Enumerate subsets of size k from n down to 1
            # For each subset, check if it's triangle-free
            # A subset S is triangle-free iff for every triangle t, not all 3 vertices of t are in S
            
            # Use iterative deepening: try k = n, n-1, ...
            # For each k, generate all C(n,k) subsets
            
            # Precompute: for each vertex, its triangle mask
            # For a subset S, the triangles fully contained in S are those where all 3 vertices are in S
            
            # Alternative: for each triangle, the subsets that contain it are those that include all 3 vertices
            # So S is triangle-free iff S does NOT contain any triangle as a subset
            
            # Let's enumerate subsets of size k and check
            # For k close to n, most subsets will contain triangles, so we can prune early
            
            # Use a recursive approach with pruning
            # State: (start_vertex, remaining_to_pick, current_mask)
            # current_mask: which vertices are in the subset so far
            
            # For each triangle, check if all its vertices are in current_mask
            # If yes, this subset (and all extensions) will contain a triangle
            
            # Better: precompute for each vertex, the set of triangles it completes
            # When we add vertex v to subset S, new triangles formed are those where v is the max vertex
            
            # Let's use a different approach: 
            # For each subset S of size k, S is triangle-free iff 
            # for every triangle (a,b,c), not (a in S and b in S and c in S)
            
            # Enumerate subsets using combinations-like generation
            # Start with k = n, check if full set is triangle-free
            # If not, try k = n-1, etc.
            
            # For each k, we need to check if ANY subset of size k is triangle-free
            # If yes, return k
            
            # To speed up: for k = n, just check the full set
            # For k = n-1, check all n subsets (remove one vertex each)
            # For k = n-2, check C(n,2) subsets, etc.
            
            # Precompute: for each vertex v, the triangles that v is part of
            # When we exclude vertex v, we eliminate all triangles containing v
            
            # Let's think about it differently:
            # S is triangle-free iff every triangle has at least one vertex NOT in S
            # Equivalently, the complement T = V \ S is a triangle transversal
            # |T| = n - k
            # So we want to find the minimum size of a triangle transversal
            # Then answer = n - min_transversal_size
            
            # Find minimum m such that there exists a set T of size m hitting all triangles
            # This is the hitting set / vertex cover problem on the hypergraph of triangles
            
            # For m from 1 to n:
            #   Check if there exists a subset of size m that hits all triangles
            
            # Use backtracking with pruning
            
            def find_min_transversal():
                # Try m = 1, 2, 3, ...
                # For each m, check if a transversal of size m exists
                
                # Precompute: for each vertex, which triangles it hits
                # vertex_hits[v] = bitmask of triangles containing v
                vertex_hits = [0] * n
                for v in range(n):
                    mask = 1 << v
                    hit = 0
                    for i, tri in enumerate(triangles):
                        if tri & mask:
                            hit |= (1 << i)
                    vertex_hits[v] = hit
                
                all_tris_mask = (1 << num_tris) - 1
                
                # For m from 1 to n:
                for m in range(1, n + 1):
                    found = False
                    
                    # Backtracking: pick m vertices that cover all triangles
                    # Pruning: if remaining vertices can't cover remaining triangles, stop
                    
                    # Sort vertices by number of triangles they hit (descending) for better pruning
                    # But we need to maintain the "start" constraint for combination generation
                    # Alternative: just try all combinations
                    
                    def backtrack(start, chosen_count, covered):
                        nonlocal found
                        if found:
                            return
                        if covered == all_tris_mask:
                            found = True
                            return
                        if chosen_count == m:
                            return
                        
                        # Pruning: if we can't cover all remaining triangles with remaining picks
                        remaining_tris = all_tris_mask & ~covered
                        # Count how many vertices from start to n-1 can help
                        # Simple pruning: if remaining_tris has bits that no remaining vertex can cover, fail
                        # For now, just proceed
                    
                        for v in range(start, n):
                            new_covered = covered | vertex_hits[v]
                            if new_covered == covered:
                                # v doesn't add any new coverage, skip
                                continue
                            backtrack(v + 1, chosen_count + 1, new_covered)
                            if found:
                                return
                    
                    backtrack(0, 0, 0)
                    if found:
                        return m
                
                return n  # Should not reach here
            
            min_trans = find_min_transversal()
            return n - min_trans
        else:
            # Fallback for many triangles: direct enumeration
            # This is slow but should rarely be needed
            triangles_set = set(triangles)
            
            def is_triangle_free(subset_mask):
                # Check if subset contains any triangle
                for tri in triangles_set:
                    if (tri & subset_mask) == tri:
                        return False
                return True
            
            # Enumerate subsets of size k from n down to 1
            # Use combinations
            from itertools import combinations
            for k in range(n, 0, -1):
                for combo in combinations(range(n), k):
                    mask = 0
                    for v in combo:
                        mask |= (1 << v)
                    if is_triangle_free(mask):
                        return k
            return 1

    # Initialize with K6,6,6 complete tripartite
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    # Also try a few random K4-free initializations
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate multiple initial candidates
    best_adj = None
    best_score = n + 1
    
    # Start with K6,6,6
    candidate = init_k666()
    score = evaluate(candidate)
    best_score = score
    best_adj = candidate[:]
    
    # Try a few random starts
    for r in range(5):
        if time.time() > deadline - 5:
            break
        candidate = generate_random_k4_free()
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_adj = candidate[:]
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local search: focus on adding edges (reduces triangle-free subgraphs)
    # and occasional removals to escape local minima
    t = 1.5
    cooling = 0.999
    min_t = 0.01
    
    while time.time() < deadline and t > min_t:
        # Try to add a random edge (most likely to improve)
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if current_adj[u] & (1 << v):
            continue
        if not can_add_edge(current_adj, u, v):
            continue
        
        old_u = current_adj[u]
        old_v = current_adj[v]
        add_edge(current_adj, u, v)
        new_score = evaluate(current_adj)
        
        delta = new_score - current_score
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            current_adj[u] = old_u
            current_adj[v] = old_v
        
        # Occasionally try removing an edge to explore
        if random.random() < 0.15:
            u = random.randint(0, n - 1)
            v_mask = current_adj[u]
            if v_mask:
                # Pick a random neighbor
                bits = []
                tmp = v_mask
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    bits.append(b)
                    tmp &= tmp - 1
                v = random.choice(bits)
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                delta = new_score - current_score
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
        
        t *= cooling
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
