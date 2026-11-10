import time
import itertools
import random

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    def time_ok():
        return time.time() < deadline

    # --- K4-free check ---
    def is_k4_free(adj, n=N):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b]
                cb = common
                while cb:
                    lb = cb & (-cb)
                    c = lb.bit_length() - 1
                    if adj[c] & common:
                        return False
                    cb ^= lb
        return True

    def k4_free_after_add(adj, u, v):
        common = adj[u] & adj[v]
        cb = common
        while cb:
            lb = cb & (-cb)
            c = lb.bit_length() - 1
            if adj[c] & common:
                return False
            cb ^= lb
        return True

    # --- Triangle-free check ---
    def is_tf(adj, mask):
        vb = mask
        while vb:
            lb = vb & (-vb)
            u = lb.bit_length() - 1
            vb ^= lb
            nu = adj[u] & mask
            wb = nu
            while wb:
                lb2 = wb & (-wb)
                w = lb2.bit_length() - 1
                wb ^= lb2
                if adj[w] & nu:
                    return False
        return True

    # --- Max TF size via branch and bound (exact) ---
    def max_tf_size(adj, timeout=1.0):
        st = time.time()
        n = N
        # Upper bound: independence number of the complement restricted to triangles
        # Use branch and bound: try to find TF set of size k
        # Key insight: a set is TF iff it contains no triangle
        # We search for the largest subset with no triangle
        # Branch: include/exclude vertices, prune if remaining + current < best
        
        best = [1]
        vertices = list(range(n))
        
        # Precompute: for each vertex, which other vertices form triangles with it
        # tri_partners[u] = set of v such that there exists w with (u,v,w) triangle
        # Actually for B&B: when we include u, we must exclude all v that form a triangle with u AND some already-included vertex
        # Simpler: when building set S, if we add u, for every pair (a,b) in S with a-b edge, 
        # if u-a and u-b edges exist, we have a triangle -> exclude u
        
        def bnb(idx, current_mask, current_size):
            if time.time() - st > timeout:
                return
            if current_size > best[0]:
                best[0] = current_size
            if idx >= n:
                return
            # Pruning: even if we take all remaining, can't beat best
            if current_size + (n - idx) <= best[0]:
                return
            
            u = vertices[idx]
            
            # Option 1: exclude u
            bnb(idx + 1, current_mask, current_size)
            
            # Option 2: include u (check no triangle formed)
            # Check: for all pairs (a,b) in current set, if adj[a]&adj[b] and adj[u]&adj[a] and adj[u]&adj[b]
            # Faster: for each a in current set that is adjacent to u, check if any b in current set 
            # that is adjacent to both a and u
            nu_in_set = adj[u] & current_mask
            # For each a in nu_in_set, check if adj[a] & nu_in_set has any bit (other than a itself)
            ok = True
            nb = nu_in_set
            while nb:
                lb = nb & (-nb)
                a = lb.bit_length() - 1
                nb ^= lb
                # Check if any other vertex in nu_in_set is adjacent to a
                if adj[a] & (nu_in_set & ~(1 << a)):
                    ok = False
                    break
            if ok:
                bnb(idx + 1, current_mask | (1 << u), current_size + 1)
        
        bnb(0, 0, 0)
        return best[0]

    # --- Build initial K6,6,6 ---
    def build_k666():
        adj = [0] * N
        for a in range(6):
            for b in range(6, 12):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
            for b in range(12, 18):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        for a in range(6, 12):
            for b in range(12, 18):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj

    # --- Get all intra-part edges ---
    intra_edges = []
    for part_start in [0, 6, 12]:
        for i in range(6):
            for j in range(i + 1, 6):
                intra_edges.append((part_start + i, part_start + j))
    
    # Total 3 * C(6,2) = 3 * 15 = 45 intra-part edges

    # --- Branch and bound over intra-part edge selections ---
    # State: which intra-part edges are added
    # We want to maximize the number of added edges (to break TF sets) while keeping K4-free
    # and minimizing max TF size
    
    # Strategy: Use B&B to find a good set of intra-part edges
    # The key constraint is K4-freeness. Adding intra-part edges can create K4s.
    
    # Let's use a smarter approach: 
    # 1. Start with K6,6,6
    # 2. Use local search with the B&B evaluator to find improvements
    # 3. Additionally, try structured patterns (e.g., add edges to form cliques within parts)
    
    best_adj = build_k666()
    best_score = max_tf_size(best_adj, timeout=2.0)
    if best_score is None:
        best_score = 12
    
    # Phase 1: Try adding edges in structured patterns
    # For each part, try different edge configurations
    # A part of 6 vertices: we can add up to 15 edges
    # But we need to maintain K4-free globally
    
    # Key insight: In K6,6,6, adding an edge (u,v) within a part creates a K4 
    # iff there exists a vertex w in another part adjacent to both u and v 
    # that also has a common neighbor... wait, in K6,6,6 all cross-part edges exist.
    # So adding edge (u,v) in part 1: any w in part 2 or part 3 is adjacent to both u and v.
    # For K4: need 4 mutually adjacent vertices. {u,v,w,x} where u,v in part1, w in part2, x in part3.
    # u-v edge added, u-w (cross), v-w (cross), w-x (cross), u-x (cross), v-x (cross). 
    # So {u,v,w,x} is a K4! 
    # Therefore: in K6,6,6, you CANNOT add ANY intra-part edge without creating a K4!
    
    # Wait, that means K6,6,6 is already maximal in terms of intra-part edges?
    # Let me verify: K6,6,6 has parts A,B,C of size 6. Edges: all A-B, A-C, B-C.
    # Add edge (a1,a2) in A. Take b1 in B, c1 in C. 
    # {a1,a2,b1,c1}: a1-a2 (new), a1-b1, a2-b1, b1-c1, a1-c1, a2-c1. All 6 edges present -> K4!
    # So indeed, no intra-part edge can be added to K6,6,6 while keeping K4-free.
    
    # This means we need a DIFFERENT base graph, not K6,6,6!
    # We need a K4-free graph that is NOT complete tripartite, so we can add edges.
    
    # Alternative: Start from a different K4-free graph structure.
    # The Ramsey number R(3,3,3) considerations... 
    # For K4-free: Turan graph T_3(18) = K6,6,6 has 54 edges.
    # We can REMOVE some cross-part edges to allow adding intra-part edges.
    
    # New strategy: 
    # 1. Start with K6,6,6
    # 2. Remove some cross-part edges (creating "gaps")
    # 3. Add intra-part edges in the gaps to reduce TF size
    # 4. Use B&B/local search to optimize
    
    # Actually, let's think about what minimizes max TF size.
    # A TF induced subgraph in K6,6,6: can have at most 2 vertices from each part (since 3 from same part form triangle if intra-edges exist, but in K6,6,6 no intra-edges, so any subset is TF as long as no 3 from same part... wait)
    # In K6,6,6: a subset S is TF iff for each part, S contains at most 2 vertices from that part.
    # Because: if S has 3 vertices from part A, they have no edges between them (no intra-part edges), 
    # but they each connect to all of B and C. The induced subgraph on S: vertices in A are independent, 
    # vertices in B are independent, vertices in C are independent, all cross edges present.
    # A triangle would need 3 mutually adjacent vertices. In K6,6,6, any triangle must have vertices 
    # from 3 different parts (one from each). So S is TF iff S has no "transversal triangle", 
    # i.e., no a in A∩S, b in B∩S, c in C∩S with all three cross edges present.
    # But in K6,6,6 ALL cross edges are present! So if S has at least one vertex from each of A, B, C, 
    # it contains a triangle (a,b,c). 
    # Therefore: S is TF in K6,6,6 iff S is contained in the union of at most 2 parts.
    # Max TF size = 12 (all of 2 parts).
    
    # To reduce below 12, we need to ensure that no 12-vertex subset is TF.
    # This means for every pair of parts (say A,B), the 12 vertices A∪B must contain a triangle.
    # In K6,6,6, A∪B has all A-B edges, no A-A or B-B edges. No triangles in A∪B!
    # So we need to add edges within parts or modify the structure.
    
    # If we add an edge within part A, say (a1,a2), then in the set A∪B:
    # a1-a2 edge exists, and a1-b, a2-b edges exist for all b in B.
    # Triangle: a1, a2, and... no, a1-a2-b1: a1-b1 yes, a2-b1 yes, a1-a2 yes -> triangle!
    # So adding even ONE intra-part edge in A makes A∪B contain triangles.
    # But then max TF might be: A∪C (12 vertices, no intra-A or intra-C edges, all A-C cross) -> still TF!
    # Unless we also add edges in C.
    
    # So to break ALL 12-vertex TF sets, we need:
    # - A∪B contains a triangle: need intra-A edge OR intra-B edge
    # - A∪C contains a triangle: need intra-A edge OR intra-C edge  
    # - B∪C contains a triangle: need intra-B edge OR intra-C edge
    # - Also need to check other 12-vertex subsets that span all 3 parts
    
    # Minimum: add at least one intra-edge in each of 2 parts (to cover all 3 pairs).
    # E.g., add one edge in A and one edge in B. Then:
    # - A∪B: has triangles (from A-edge or B-edge)
    # - A∪C: has triangles (from A-edge)
    # - B∪C: has triangles (from B-edge)
    # But what about mixed 12-vertex subsets? E.g., 5 from A, 4 from B, 3 from C.
    # This could still be TF if no triangles formed.
    
    # The B&B approach: systematically try edge additions and evaluate.
    
    # Let's use a different base: a "blow-up" of a smaller K4-free graph.
    # Or: use the fact that we can remove cross-edges and add intra-edges.
    
    # Practical approach: local search with fast evaluation
    # 1. Start with K6,6,6
    # 2. In each step: try (remove one cross-edge, add one intra-edge) or (add one intra-edge after removing needed cross-edges)
    # 3. Evaluate with B&B max_tf_size
    
    # For efficiency, let's precompute which intra-edge additions require which cross-edge removals
    
    # Actually, the cleanest approach given time constraints:
    # Use simulated annealing / local search with the exact B&B evaluator
    
    adj = build_k666()
    score = max_tf_size(adj, timeout=2.0)
    if score is None:
        score = 12
    
    # The issue: we can't add intra-edges to K6,6,6 without removing cross-edges first
    # So moves are: (remove cross-edge u-v) + (add intra-edge a-b) as a combined move
    
    # Precompute: for each intra-edge (a,b) in part P, which cross-edges must be removed?
    # Adding (a,b) in part P creates K4 with any {a,b,w,x} where w in other part1, x in other part2,
    # and all 6 edges present. We need to break at least one edge in each such K4.
    # Simplest: remove all cross-edges incident to a or b to other parts? That's too many.
    # Alternative: for each (a,b) intra-edge, we need to ensure no w,x in other parts have all 4 cross-edges.
    # This is complex. Let's just do local search with combined moves.
    
    # Define move space:
    # Move type 1: add intra-edge (requires checking K4-free after addition, which fails for K6,6,6)
    # Move type 2: remove cross-edge (u,v) where u in part i, v in part j
    # Move type 3: combined: remove cross-edge(s) + add intra-edge
    
    # For simplicity and speed, let's do:
    # 1. Try removing one cross-edge and adding one intra-edge
    # 2. Evaluate
    # 3. Keep if improved
    
    cross_edges = []
    for a in range(6):
        for b in range(6, 12):
            cross_edges.append((a, b))
        for b in range(12, 18):
            cross_edges.append((a, b))
    for a in range(6, 12):
        for b in range(12, 18):
            cross_edges.append((a, b))
    
    random.seed(42)
    
    # Phase 1: Greedy combined moves
    improved = True
    while improved and time_ok():
        improved = False
        random.shuffle(cross_edges)
        random.shuffle(intra_edges)
        
        for ce in cross_edges:
            if not time_ok():
                break
            cu, cv = ce
            # Skip if already removed
            if not (adj[cu] >> cv) & 1:
                continue
            
            for ie in intra_edges:
                if not time_ok():
                    break
                iu, iv = ie
                # Skip if already added
                if (adj[iu] >> iv) & 1:
                    continue
                # Skip if same part as cross edge endpoints (doesn't help)
                # Actually, any intra-edge might help
                
                # Try: remove cross-edge, add intra-edge
                adj[cu] &= ~(1 << cv)
                adj[cv] &= ~(1 << cu)
                adj[iu] |= (1 << iv)
                adj[iv] |= (1 << iu)
                
                if is_k4_free(adj):
                    new_score = max_tf_size(adj, timeout=1.0)
                    if new_score is not None and new_score < score:
                        score = new_score
                        improved = True
                        # Continue from this state
                        break
                    else:
                        # Revert
                        adj[cu] |= (1 << cv)
                        adj[cv] |= (1 << cu)
                        adj[iu] &= ~(1 << iv)
                        adj[iv] &= ~(1 << iu)
                else:
                    # Revert
                    adj[cu] |= (1 << cv)
                    adj[cv] |= (1 << cu)
                    adj[iu] &= ~(1 << iv)
                    adj[iv] &= ~(1 << iu)
            
            if improved:
                break
    
    best_adj = adj[:]
    best_score = score
    
    # Phase 2: Multi-start with random perturbations
    for restart in range(50):
        if not time_ok():
            break
        
        adj = build_k666()
        # Random perturbation: remove some cross-edges, add some intra-edges
        num_remove = random.randint(1, 5)
        num_add = random.randint(1, 3)
        
        shuffled_cross = cross_edges[:]
        random.shuffle(shuffled_cross)
        for ce in shuffled_cross[:num_remove]:
            cu, cv = ce
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
        
        shuffled_intra = intra_edges[:]
        random.shuffle(shuffled_intra)
        for ie in shuffled_intra[:num_add]:
            iu, iv = ie
            adj[iu] |= (1 << iv)
            adj[iv] |= (1 << iu)
            if not is_k4_free(adj):
                adj[iu] &= ~(1 << iv)
                adj[iv] &= ~(1 << iu)
        
        if not is_k4_free(adj):
            continue
        
        score = max_tf_size(adj, timeout=1.5)
        if score is None:
            continue
        
        # Local search from this state
        local_improved = True
        while local_improved and time_ok():
            local_improved = False
            random.shuffle(cross_edges)
            random.shuffle(intra_edges)
            
            for ce in cross_edges:
                if not time_ok():
                    break
                cu, cv = ce
                if not (adj[cu] >> cv) & 1:
                    continue
                
                for ie in intra_edges:
                    if not time_ok():
                        break
                    iu, iv = ie
                    if (adj[iu] >> iv) & 1:
                        continue
                    
                    adj[cu] &= ~(1 << cv)
                    adj[cv] &= ~(1 << cu)
                    adj[iu] |= (1 << iv)
                    adj[iv] |= (1 << iu)
                    
                    if is_k4_free(adj):
                        new_score = max_tf_size(adj, timeout=0.8)
                        if new_score is not None and new_score < score:
                            score = new_score
                            local_improved = True
                            break
                        else:
                            adj[cu] |= (1 << cv)
                            adj[cv] |= (1 << cu)
                            adj[iu] &= ~(1 << iv)
                            adj[iv] &= ~(1 << iu)
                    else:
                        adj[cu] |= (1 << cv)
                        adj[cv] |= (1 << cu)
                        adj[iu] &= ~(1 << iv)
                        adj[iv] &= ~(1 << iu)
                
                if local_improved:
                    break
            
            # Also try single intra-edge additions (if K4-free allows)
            if not local_improved and time_ok():
                for ie in intra_edges:
                    if not time_ok():
                        break
                    iu, iv = ie
                    if (adj[iu] >> iv) & 1:
                        continue
                    adj[iu] |= (1 << iv)
                    adj[iv] |= (1 << iu)
                    if is_k4_free(adj):
                        new_score = max_tf_size(adj, timeout=0.8)
                        if new_score is not None and new_score < score:
                            score = new_score
                            local_improved = True
                        else:
                            adj[iu] &= ~(1 << iv)
                            adj[iv] &= ~(1 << iu)
                    else:
                        adj[iu] &= ~(1 << iv)
                        adj[iv] &= ~(1 << iu)
                    if local_improved:
                        break
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    # Phase 3: Final polish - try all single moves
    if time_ok():
        adj = best_adj[:]
        score = best_score
        improved = True
        while improved and time_ok():
            improved = False
            # Try adding intra-edges
            for ie in intra_edges:
                if not time_ok():
                    break
                iu, iv = ie
                if (adj[iu] >> iv) & 1:
                    continue
                adj[iu] |= (1 << iv)
                adj[iv] |= (1 << iu)
                if is_k4_free(adj):
                    new_score = max_tf_size(adj, timeout=0.5)
                    if new_score is not None and new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[iu] &= ~(1 << iv)
                        adj[iv] &= ~(1 << iu)
                else:
                    adj[iu] &= ~(1 << iv)
                    adj[iv] &= ~(1 << iu)
            
            # Try removing cross-edges (unlikely to help but check)
            if not improved:
                for ce in cross_edges:
                    if not time_ok():
                        break
                    cu, cv = ce
                    if not (adj[cu] >> cv) & 1:
                        continue
                    adj[cu] &= ~(1 << cv)
                    adj[cv] &= ~(1 << cu)
                    new_score = max_tf_size(adj, timeout=0.3)
                    if new_score is not None and new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[cu] |= (1 << cv)
                        adj[cv] |= (1 << cu)
                    if improved:
                        break
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
