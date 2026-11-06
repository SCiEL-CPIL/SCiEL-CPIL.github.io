import time
import random
import sys

DEADLINE = 135.0  # Safety margin before 150s kill

def _is_3_colorable_fast(adj_mask, n):
    """
    Fast 3-colorability check using backtracking with bitmasks.
    Returns True if 3-colorable.
    """
    # Order by degree descending
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    colors = [-1] * n
    # For each vertex in order, precompute which neighbors come before it
    # We'll use a recursive approach with early pruning
    
    # Count of colored neighbors for forward checking
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine forbidden colors from already-colored neighbors
        forbidden = 0
        m = adj_mask[v]
        mm = m
        while mm:
            lsb = mm & -mm
            nb = lsb.bit_length() - 1
            if pos[nb] < idx:
                c = colors[nb]
                if c >= 0:
                    forbidden |= (1 << c)
            mm ^= lsb
        
        # Try colors
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

def _is_triangle_free(adj_mask, n):
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True

def construct():
    n = 16
    start_time = time.time()
    
    best_edges = []
    best_count = 0
    
    def edge_count(adj_mask):
        return sum(bin(m).count('1') for m in adj_mask) // 2
    
    def can_add_edge(adj_mask, u, v):
        if u == v:
            return False
        if adj_mask[u] & (1 << v):
            return False
        if adj_mask[u] & adj_mask[v]:
            return False
        return True
    
    def try_greedy_completion(adj_mask, seed_val):
        """
        Given a partial graph, greedily add all possible edges in random order.
        Returns the final adj_mask.
        """
        rng = random.Random(seed_val)
        adj = adj_mask[:]
        
        # Collect all addable edges, shuffle, and add them
        # Do multiple passes to handle edges that become addable after others are added
        for _pass in range(3):
            addable = []
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add_edge(adj, u, v):
                        addable.append((u, v))
            if not addable:
                break
            rng.shuffle(addable)
            for u, v in addable:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj
    
    # Define several non-3-colorable triangle-free cores
    # Core 1: Grötzsch graph (11 vertices, 20 edges)
    # Core 2: Mycielski of C5 = Grötzsch, but we can try different embeddings
    # Core 3: Smaller cores - e.g., odd cycle C5 is 3-colorable, so we need at least 4-chromatic
    # Actually, the smallest 4-chromatic triangle-free graph is the Grötzsch (11 vertices)
    # But we can also use other 4-chromatic triangle-free graphs on fewer vertices? No, Grötzsch is minimal.
    # Wait - we can use subgraphs. A 4-chromatic graph must have chromatic number 4.
    # The Grötzsch graph is the unique 4-chromatic triangle-free graph on 11 vertices.
    # But we can embed it in different ways on 16 vertices.
    
    # Let's define the Grötzsch graph edges
    groetzsche_edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # C5
        (5,1),(5,3),  # 5 connected to 1,3 (neighbors of 0 in C5... wait let me get this right)
    ]
    # Actually, let me construct Grötzsch properly:
    # Start with C5: vertices 0-4, edges (0,1),(1,2),(2,3),(3,4),(4,0)
    # Mycielski: add vertices 5-9, each 5+i connected to (i+1)%5 and (i-1)%5
    # Add vertex 10 connected to 5,6,7,8,9
    # This gives 5 + 10 + 5 = 20 edges on 11 vertices
    
    def make_groetzsche():
        adj = [0] * n
        # C5
        for i in range(5):
            u, v = i, (i+1) % 5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Mycielski layer
        for i in range(5):
            u = 5 + i
            v1 = (i + 1) % 5
            v2 = (i - 1) % 5
            adj[u] |= (1 << v1)
            adj[v1] |= (1 << u)
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
        # Universal vertex
        for i in range(5):
            u = 10
            v = 5 + i
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def make_groetzsche_shifted(shift):
        """Embed Grötzsch with a permutation of vertices 0-10 mapped to specific positions."""
        adj = [0] * n
        # Map Groetzsch vertices 0-10 to specific positions
        # We'll try different mappings
        perm = list(range(11))
        # Apply some permutation to make different embeddings
        rng = random.Random(shift)
        rng.shuffle(perm)
        # Map groetzsch vertex i to position perm[i]
        # But we need to keep it within 0-15
        # Let's just use a subset of 11 positions from 0-15
        positions = list(range(16))
        rng.shuffle(positions)
        chosen_positions = positions[:11]
        # Map groetzsch vertex i to chosen_positions[perm[i]]
        # Actually simpler: just permute which 11 of 16 vertices are used
        # and permute the labeling within those 11
        
        # Let me just create the standard Groetzsch and then apply a vertex permutation
        base_adj = make_groetzsche()
        # Create permutation of all 16 vertices
        full_perm = list(range(16))
        rng.shuffle(full_perm)
        # New adj: edge (u,v) in new graph iff (perm_inv[u], perm_inv[v]) is edge in base
        perm_inv = [0] * 16
        for i, p in enumerate(full_perm):
            perm_inv[p] = i
        
        new_adj = [0] * n
        for u in range(11):
            for v in range(u+1, 11):
                if base_adj[u] & (1 << v):
                    nu = full_perm[u]
                    nv = full_perm[v]
                    new_adj[nu] |= (1 << nv)
                    new_adj[nv] |= (1 << nu)
        return new_adj
    
    # Also try: start with just a 4-chromatic core and see if we can do better
    # Another approach: use a bipartite graph plus a few extra edges to break 3-colorability
    # Actually, let's try a different strategy:
    # Start with a random triangle-free graph, check if it's 3-colorable.
    # If not, try to add more edges.
    # If yes, modify it to make it non-3-colorable while keeping it triangle-free.
    
    # Strategy: Try many random triangle-free graphs, check 3-colorability, 
    # and for non-3-colorable ones, greedily complete.
    
    # But generating random triangle-free graphs is tricky.
    # Better: start from known non-3-colorable cores and complete greedily.
    
    # Let's also try: the 4-chromatic property can come from odd cycles in the "constraint graph"
    # For 3-coloring, the constraint is that adjacent vertices get different colors.
    # A graph is not 3-colorable if and only if... it's complex.
    
    # Let's focus on the Greetzsch-based approach with many permutations.
    
    # Phase 1: Try many permutations of Grötzsch embedding + greedy completion
    n_trials = 0
    max_trials = 500
    
    for trial in range(max_trials):
        if time.time() - start_time > DEADLINE:
            break
        
        adj = make_groetzsche_shifted(trial)
        
        # Verify it's triangle-free and non-3-colorable
        if not _is_triangle_free(adj, n):
            continue
        if _is_3_colorable_fast(adj, n):
            continue
        
        # Greedy completion with random order
        completed = try_greedy_completion(adj, trial * 1000 + 42)
        
        # Verify completed is still triangle-free
        if not _is_triangle_free(completed, n):
            continue
        
        # Verify still non-3-colorable
        if _is_3_colorable_fast(completed, n):
            continue
        
        cnt = edge_count(completed)
        if cnt > best_count:
            best_count = cnt
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if completed[u] & (1 << v):
                        best_edges.append([u, v])
            n_trials = trial + 1
    
    # Phase 2: Try to improve by adding edges one at a time to the best graph
    if best_count > 0 and time.time() - start_time < DEADLINE:
        # Reconstruct best_adj from best_edges
        best_adj = [0] * n
        for u, v in best_edges:
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        
        # Try to add more edges
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add_edge(best_adj, u, v):
                        # Check if adding this edge keeps it non-3-colorable
                        best_adj[u] |= (1 << v)
                        best_adj[v] |= (1 << u)
                        if not _is_3_colorable_fast(best_adj, n):
                            improved = True
                        else:
                            best_adj[u] &= ~(1 << v)
                            best_adj[v] &= ~(1 << u)
        
        cnt = edge_count(best_adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_adj[u] & (1 << v):
                        best_edges.append([u, v])
    
    # Phase 3: Local search - try removing one edge and adding two
    if best_count > 0 and time.time() - start_time < DEADLINE:
        best_adj = [0] * n
        for u, v in best_edges:
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        
        for attempt in range(200):
            if time.time() - start_time > DEADLINE:
                break
            
            # Get all current edges
            current_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_adj[u] & (1 << v):
                        current_edges.append((u, v))
            
            if not current_edges:
                break
            
            # Pick a random edge to remove
            rem_u, rem_v = random.choice(current_edges)
            best_adj[rem_u] &= ~(1 << rem_v)
            best_adj[rem_v] &= ~(1 << rem_u)
            
            # Try to add two edges
            addable = []
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add_edge(best_adj, u, v):
                        addable.append((u, v))
            
            if len(addable) >= 2:
                # Try pairs of addable edges
                random.shuffle(addable)
                found = False
                for i in range(min(len(addable), 30)):
                    if found:
                        break
                    for j in range(i + 1, min(len(addable), 30)):
                        a1, b1 = addable[i]
                        a2, b2 = addable[j]
                        # Temporarily add both
                        best_adj[a1] |= (1 << b1)
                        best_adj[b1] |= (1 << a1)
                        # Check if second can still be added
                        if can_add_edge(best_adj, a2, b2):
                            best_adj[a2] |= (1 << b2)
                            best_adj[b2] |= (1 << a2)
                            # Check triangle-free and non-3-colorable
                            if _is_triangle_free(best_adj, n) and not _is_3_colorable_fast(best_adj, n):
                                cnt = edge_count(best_adj)
                                if cnt > best_count:
                                    best_count = cnt
                                    best_edges = []
                                    for u in range(n):
                                        for v in range(u + 1, n):
                                            if best_adj[u] & (1 << v):
                                                best_edges.append([u, v])
                                found = True
                                break
                            else:
                                best_adj[a2] &= ~(1 << b2)
                                best_adj[b2] &= ~(1 << a2)
                        else:
                            pass
                        # Undo first edge
                        best_adj[a1] &= ~(1 << b1)
                        best_adj[b1] &= ~(1 << a1)
                
                if found:
                    continue
            
            # If we didn't find a good pair, restore the removed edge
            best_adj[rem_u] |= (1 << rem_v)
            best_adj[rem_v] |= (1 << rem_u)
    
    # Final verification
    if not best_edges:
        # Fallback: return Grötzsch graph
        adj = make_groetzsche()
        best_edges = []
        for u in range(11):
            for v in range(u + 1, 11):
                if adj[u] & (1 << v):
                    best_edges.append([u, v])
    
    # Verify the final answer
    final_adj = [0] * n
    for u, v in best_edges:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    if not _is_triangle_free(final_adj, n):
        # Fallback
        adj = make_groetzsche()
        best_edges = []
        for u in range(11):
            for v in range(u + 1, 11):
                if adj[u] & (1 << v):
                    best_edges.append([u, v])
    elif _is_3_colorable_fast(final_adj, n):
        # Fallback
        adj = make_groetzsche()
        best_edges = []
        for u in range(11):
            for v in range(u + 1, 11):
                if adj[u] & (1 << v):
                    best_edges.append([u, v])
    
    return best_edges
