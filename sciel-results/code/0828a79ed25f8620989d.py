import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_result = None
    best_n = 0

    # Strategy 1: Optimized Local Search with Delta Violations for n=26
    # This is the primary engine. It uses fast updates to allow many more steps.
    if time.time() < deadline - 20:
        r26 = fast_local_search(26, time_limit=40.0)
        if r26 is not None:
            best_result = r26
            best_n = 26
            
            # Strategy 2: Try to extend to 27 using the same fast engine
            if time.time() < deadline - 15:
                r27 = fast_local_search(27, time_limit=25.0, initial_colors=None)
                if r27 is not None:
                    return r27
            
            # If direct 27 search failed, try extending the specific 26 solution
            # by adding a vertex and running a focused search on the new edges only
            # But fast_local_search on 27 from scratch is often better if it converges.
            # Let's try a second attempt at 27 with different random seeds if time permits
            if time.time() < deadline - 10:
                r27b = fast_local_search(27, time_limit=15.0, initial_colors=None)
                if r27b is not None:
                    return r27b

    # Strategy 3: If 26 failed, try again with different parameters or seeds
    if best_n < 26 and time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            r26 = fast_local_search(26, time_limit=5.0)
            if r26 is not None:
                best_result = r26
                best_n = 26
                break

    # Fallback: Try smaller n if 26 failed completely (unlikely with fast search)
    if best_result is None:
        for n in [25, 24, 23, 22, 21, 20]:
            if time.time() > deadline - 5:
                break
            r = fast_local_search(n, time_limit=5.0)
            if r is not None:
                return r

    # Absolute fallback: Hardcoded valid small coloring
    # For n=11, a simple cyclic coloring often works or we can just return a known valid one
    # Actually, let's just generate a random one and hope, or use a deterministic small one.
    # The prompt says 1 <= n <= 29. A valid coloring for n=10 is easy to find.
    # Let's provide a guaranteed valid n=11 coloring if all else fails.
    # Colors for K_11 without C4 in any color class.
    # A known construction:
    # But simpler: just return a random one and hope it's valid? No, must be valid.
    # Let's use a simple greedy construction for n=11 as absolute fallback.
    return greedy_construct(11)

def greedy_construct(n):
    """
    Greedy vertex-by-vertex construction.
    This is slow but guaranteed to find a solution for small n if one exists easily.
    For n=11, it should be instant.
    """
    colors = [0] * (n * (n - 1) // 2)
    adj = [[set() for _ in range(n)] for _ in range(5)]
    
    # We need to assign colors to edges (i, j) in lexicographic order.
    # For each edge (i, j), we pick a color c such that adding it doesn't create a C4.
    # A C4 is formed if there exist u, v such that (u, i) and (u, j) are color c, 
    # and (v, i) and (v, j) are color c? No.
    # C4: u-v-w-x-u. 
    # If we add edge (i, j) with color c, it forms a C4 if there is a vertex k 
    # such that (i, k) is c and (j, k) is c, AND there is ANOTHER vertex l such that 
    # (i, l) is c and (j, l) is c.
    # Wait, the condition is: "No two vertices with two common neighbours in one colour".
    # So, if we add edge (i, j) with color c, we must ensure that 
    # |N_c(i) ∩ N_c(j)| < 2.
    # Note: N_c(i) includes j if edge (i,j) is c. But we are adding it.
    # Before adding (i,j), N_c(i) and N_c(j) do not include each other.
    # After adding (i,j) with color c:
    # Common neighbors of i and j in color c will be (N_c(i) ∩ N_c(j)) plus potentially others?
    # No, the common neighbors of i and j are vertices k such that (i,k) is c and (j,k) is c.
    # Adding (i,j) makes j a neighbor of i and i a neighbor of j.
    # Does it create a C4?
    # A C4 involving the new edge (i,j) would be i-j-k-l-i.
    # This requires (j,k)=c, (k,l)=c, (l,i)=c.
    # This is equivalent to k being a common neighbor of i and j in color c, 
    # AND l being a common neighbor of i and j in color c?
    # No.
    # Let's stick to the definition: "No colour class may contain a four-cycle".
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and edges (a,b), (b,c), (c,d), (d,a) all in color c.
    # If we add edge (i,j) with color c, it can only be part of a new C4 if there exists
    # a path of length 3 between i and j in color c: i-k-l-j.
    # This means k is a neighbor of i in c, l is a neighbor of j in c, and (k,l) is c.
    # Alternatively, the "common neighbor" definition:
    # Two vertices u, v have two common neighbors w, x in color c if (u,w),(u,x),(v,w),(v,x) are all c.
    # This forms the C4 u-w-v-x-u.
    # So, adding edge (i,j) with color c is valid if and only if
    # the number of vertices k such that (i,k) is c and (j,k) is c is less than 2.
    # Wait, if (i,j) is c, then i and j are neighbors.
    # The condition "no two vertices with two common neighbours" applies to ALL pairs.
    # If we add (i,j) with color c, we must check if it creates a violation for ANY pair.
    # The new edge (i,j) creates a potential C4 for the pair (i,j) themselves?
    # No, (i,j) are neighbors. Do they have 2 common neighbors?
    # Let S = {k | (i,k)=c and (j,k)=c}.
    # If |S| >= 2, say k1, k2, then i-k1-j-k2-i is a C4?
    # Edges: (i,k1), (k1,j), (j,k2), (k2,i). All are c. Yes.
    # So we must ensure |N_c(i) ∩ N_c(j)| < 2 BEFORE adding (i,j)?
    # No, N_c(i) currently does not contain j. N_c(j) does not contain i.
    # After adding (i,j) with color c:
    # N_c(i) becomes N_c(i) ∪ {j}.
    # N_c(j) becomes N_c(j) ∪ {i}.
    # The common neighbors of i and j in the NEW graph are N_c(i)_old ∩ N_c(j)_old.
    # Wait, does (i,j) affect the common neighbors of OTHER pairs?
    # Adding (i,j) with color c might create a C4 for a pair (u,v) where u != i,j and v != i,j?
    # Only if the C4 uses the edge (i,j).
    # A C4 using (i,j) must have (i,j) as one side.
    # The other side is a path i-k-l-j or i-l-k-j?
    # It must be i-j-k-l-i.
    # This requires (j,k)=c, (k,l)=c, (l,i)=c.
    # This is equivalent to: k is a neighbor of j in c, l is a neighbor of i in c, and (k,l)=c.
    # AND k != i, l != j.
    # This is getting complicated to check incrementally for all pairs.
    
    # However, the "common neighbor" check is local to the endpoints of the edge being added?
    # If we add (i,j) with color c, the ONLY new C4s that can be formed are those that include the edge (i,j).
    # Any C4 including (i,j) must be of the form i-j-k-l-i.
    # This implies k is a common neighbor of i and j in color c?
    # Let's check:
    # Edges: (i,j), (j,k), (k,l), (l,i).
    # If (j,k)=c and (l,i)=c, and (k,l)=c.
    # This does NOT require k to be a common neighbor of i and j.
    # It requires k to be in N_c(j) and l to be in N_c(i) and (k,l)=c.
    
    # Actually, the condition "No two vertices with two common neighbours in one colour" is equivalent to C4-free.
    # If we add edge (i,j) with color c, we introduce i to N_c(j) and j to N_c(i).
    # This might cause a pair (u,v) to have 2 common neighbors if one of the new neighbors completes a pair.
    # But it's much faster to just check the C4s involving (i,j).
    # A C4 involving (i,j) is determined by choosing k from N_c(j) \ {i} and l from N_c(i) \ {j} such that (k,l)=c.
    # If there is MORE THAN ONE such path? No, a single path i-j-k-l-i is a C4.
    # So we must ensure there is NO such k,l.
    
    # This greedy check is O(n^2) per edge, which is O(n^4) total. Too slow for n=26?
    # 26^4 is ~450k operations. That's fine.
    
    # Let's implement this greedy check properly.
    
    # We maintain adj_mask[c][u] as a bitmask of neighbors in color c.
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [0] * len(edges)
    adj_mask = [[0] * n for _ in range(5)]
    
    def is_valid_addition(i, j, c):
        # Check if adding (i,j) with color c creates a C4.
        # A C4 containing (i,j) is i-j-k-l-i.
        # We need to check if there exists k, l such that:
        # 1. (j,k) is c
        # 2. (l,i) is c
        # 3. (k,l) is c
        # 4. k != i, l != j, k != l
        # Note: Since we are adding (i,j), k cannot be i (i is not neighbor of j yet in c? 
        # Actually, we are checking validity BEFORE adding. So N_c(j) does not contain i.
        # So k is from current N_c(j).
        # l is from current N_c(i).
        
        neighbors_j = adj_mask[c][j]
        neighbors_i = adj_mask[c][i]
        
        # Iterate through k in N_c(j)
        temp_k = neighbors_j
        while temp_k:
            lsb = temp_k & -temp_k
            k = lsb.bit_length() - 1
            temp_k ^= lsb
            
            # Iterate through l in N_c(i)
            temp_l = neighbors_i
            while temp_l:
                lsb_l = temp_l & -temp_l
                l = lsb_l.bit_length() - 1
                temp_l ^= lsb_l
                
                # Check if (k,l) is color c
                if (adj_mask[c][k] >> l) & 1:
                    # Also need k != l (implied if graph is simple and no self loops, but k,l are distinct vertices)
                    # And we need to ensure it's a simple cycle.
                    # i-j-k-l-i.
                    # If k == l, it's a triangle i-j-k-i, not a C4.
                    # If k == i or l == j, it's not a simple cycle of length 4.
                    # But k is from N_c(j), so k != j. k could be i? No, i is not in N_c(j) yet.
                    # l is from N_c(i), so l != i. l could be j? No, j is not in N_c(i) yet.
                    # So k, l are distinct from i, j.
                    # If k == l, then i-j-k-i is a triangle. Not a C4.
                    if k != l:
                        return False
        return True

    for idx, (i, j) in enumerate(edges):
        found = False
        for c in range(5):
            if is_valid_addition(i, j, c):
                colors[idx] = c
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                found = True
                break
        if not found:
            return None # Failed to construct
    
    return {'n': n, 'colors': colors}

def fast_local_search(n, time_limit=40.0, initial_colors=None):
    """
    Local search with O(1) delta violation updates.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize
    if initial_colors:
        colors = initial_colors[:]
    else:
        colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)
    
    # Precompute edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Function to compute total violations from scratch (for initialization)
    def compute_total_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    current_violations = compute_total_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    no_improve = 0
    max_no_improve = 5000
    
    # Helper to get delta violations when flipping edge (i,j) from c_old to c_new
    def get_delta_violations(i, j, c_old, c_new):
        # Violations lost by removing (i,j) from c_old
        # Violations gained by adding (i,j) to c_new
        
        # A violation is a pair (u,v) with >= 2 common neighbors in color c.
        # Removing (i,j) from c_old affects:
        # 1. The pair (i,j) itself: its common neighbors in c_old decrease by 0? 
        #    No, removing edge (i,j) means i is no longer neighbor of j.
        #    The common neighbors of i and j in c_old are N_c_old(i) ∩ N_c_old(j).
        #    Before removal, j was in N_c_old(i) and i was in N_c_old(j).
        #    Wait, N_c(i) includes j if (i,j) is c.
        #    So N_c_old(i) ∩ N_c_old(j) includes... wait.
        #    Common neighbors of i and j are vertices k such that (i,k)=c and (j,k)=c.
        #    Removing (i,j) does NOT change the set of common neighbors of i and j.
        #    It only changes the fact that i and j are no longer connected.
        #    Does this remove a C4?
        #    A C4 involving (i,j) is i-j-k-l-i.
        #    If we remove (i,j), this C4 is broken.
        #    How many C4s involve (i,j)?
        #    It's the number of paths i-k-l-j in color c_old?
        #    No, it's the number of pairs (k,l) such that (j,k)=c, (l,i)=c, (k,l)=c.
        #    This is exactly the number of common neighbors? No.
        
        # Let's use the common neighbor definition directly.
        # Violation count = sum_{u<v} C(|N_c(u) ∩ N_c(v)|, 2)
        
        # When we change (i,j) from c_old to c_new:
        
        # 1. Impact on c_old:
        #    Remove (i,j) from c_old.
        #    N_c_old(i) loses j. N_c_old(j) loses i.
        #    Which pairs (u,v) have their common neighbor count changed?
        #    Only pairs where u or v is i or j.
        #    Pairs: (i, v) for v != j, and (j, v) for v != i.
        #    For pair (i, v):
        #    Old common neighbors: N_c_old(i) ∩ N_c_old(v).
        #    New common neighbors: (N_c_old(i) \ {j}) ∩ N_c_old(v).
        #    If j was in the intersection, the count drops by 1.
        #    j is in N_c_old(i) ∩ N_c_old(v) iff (i,j)=c_old and (j,v)=c_old.
        #    So for each v such that (j,v)=c_old, the common neighbor count of (i,v) drops by 1.
        #    Similarly for (j,v) where (i,v)=c_old.
        
        #    Let's calculate the change in sum of C(k,2).
        #    Let k_old = |N_c_old(i) ∩ N_c_old(v)|.
        #    If j is in the intersection, k_new = k_old - 1.
        #    Delta = C(k_new, 2) - C(k_old, 2) = C(k_old-1, 2) - C(k_old, 2) = -(k_old - 1).
        
        #    So, for each v such that (j,v)=c_old:
        #       k = |N_c_old(i) ∩ N_c_old(v)|
        #       delta -= (k - 1)
        
        #    And for each v such that (i,v)=c_old:
        #       k = |N_c_old(j) ∩ N_c_old(v)|
        #       delta -= (k - 1)
        
        #    Note: We must be careful not to double count or mess up indices.
        #    Also, the pair (i,j) itself?
        #    Common neighbors of i and j: N_c_old(i) ∩ N_c_old(j).
        #    This set does NOT include i or j.
        #    Removing (i,j) does not change this set.
        #    So the violation count for pair (i,j) remains the same?
        #    Wait. If i and j have 2 common neighbors, that's a violation.
        #    Removing the edge (i,j) does NOT remove the common neighbors.
        #    So the violation for pair (i,j) persists?
        #    Yes, the definition is "No two vertices with two common neighbours".
        #    Even if they are not connected, if they share 2 neighbors, it's a violation?
        #    "No colour class may contain a four-cycle".
        #    A 4-cycle is a subgraph. If i and j are not connected, they can't form a cycle with each other as adjacent vertices.
        #    But the condition "No two vertices with two common neighbours" is a stronger condition than "C4-free"?
        #    Actually, in a graph, if two vertices have two common neighbors, they form a C4 with those neighbors?
        #    u-w-v-x-u. Yes.
        #    So if i and j have 2 common neighbors, there IS a C4 in the color class.
        #    Removing the edge (i,j) does NOT break that C4. The C4 is i-w-j-x-i.
        #    So the violation count for pair (i,j) is UNCHANGED by removing (i,j).
        
        #    So, for c_old:
        #    Delta_old = 0
        #    For each v in N_c_old(j) \ {i}:
        #         k = |N_c_old(i) ∩ N_c_old(v)|
        #         # Note: j is in N_c_old(i). Is j in N_c_old(v)?
        #         # If (j,v)=c_old, then yes.
        #         # So j is in the intersection.
        #         # When we remove (i,j), j is removed from N_c_old(i).
        #         # So the new intersection is (Old Intersection) \ {j}.
        #         # So k_new = k_old - 1.
        #         # Delta = C(k-1, 2) - C(k, 2) = -(k-1).
        #         Delta_old -= (k - 1)
        
        #    For each v in N_c_old(i) \ {j}:
        #         k = |N_c_old(j) ∩ N_c_old(v)|
        #         Delta_old -= (k - 1)
        
        # 2. Impact on c_new:
        #    Add (i,j) to c_new.
        #    N_c_new(i) gains j. N_c_new(j) gains i.
        #    For each v in N_c_new(j) \ {i}:
        #         k = |N_c_new(i) ∩ N_c_new(v)|
        #         # j is in N_c_new(i). Is j in N_c_new(v)?
        #         # If (j,v)=c_new, then yes.
        #         # So j is in the intersection.
        #         # Adding (i,j) adds j to N_c_new(i).
        #         # So the new intersection is (Old Intersection) ∪ {j}.
        #         # k_new = k_old + 1.
        #         # Delta = C(k+1, 2) - C(k, 2) = k.
        #         Delta_new += k
        
        #    For each v in N_c_new(i) \ {j}:
        #         k = |N_c_new(j) ∩ N_c_new(v)|
        #         Delta_new += k
        
        #    Total Delta = Delta_old + Delta_new
        
        delta = 0
        
        # Process c_old
        if c_old != -1:
            # v in N_c_old(j)
            temp = adj_mask[c_old][j]
            # Exclude i
            temp &= ~(1 << i)
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # k = |N_c_old(i) ∩ N_c_old(v)|
                # Note: j is in this intersection because (j,v)=c_old and (i,j)=c_old.
                k = (adj_mask[c_old][i] & adj_mask[c_old][v]).bit_count()
                delta -= (k - 1)
            
            # v in N_c_old(i)
            temp = adj_mask[c_old][i]
            temp &= ~(1 << j)
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                k = (adj_mask[c_old][j] & adj_mask[c_old][v]).bit_count()
                delta -= (k - 1)
        
        # Process c_new
        if c_new != -1:
            # v in N_c_new(j)
            temp = adj_mask[c_new][j]
            temp &= ~(1 << i)
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                k = (adj_mask[c_new][i] & adj_mask[c_new][v]).bit_count()
                delta += k
            
            # v in N_c_new(i)
            temp = adj_mask[c_new][i]
            temp &= ~(1 << j)
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                k = (adj_mask[c_new][j] & adj_mask[c_new][v]).bit_count()
                delta += k
                
        return delta

    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            # Restart
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            current_violations = compute_total_violations()
            temperature = 1.0
            no_improve = 0
            continue
        
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try all new colors
        best_new_color = old_color
        best_delta = 0
        
        for new_color in range(5):
            if new_color == old_color:
                continue
            
            d = get_delta_violations(i, j, old_color, new_color)
            if d < best_delta:
                best_delta = d
                best_new_color = new_color
        
        if best_new_color != old_color:
            # Apply change
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            colors[e_idx] = best_new_color
            
            current_violations += best_delta
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                no_improve = 0
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
            else:
                no_improve += 1
                
            # Simulated Annealing step
            if best_delta > 0:
                if random.random() < math.exp(-best_delta / max(temperature, 0.01)):
                    # Accept
                    pass
                else:
                    # Revert
                    adj_mask[best_new_color][i] &= ~(1 << j)
                    adj_mask[best_new_color][j] &= ~(1 << i)
                    adj_mask[old_color][i] |= (1 << j)
                    adj_mask[old_color][j] |= (1 << i)
                    colors[e_idx] = old_color
                    current_violations -= best_delta
                    no_improve += 1
        else:
            no_improve += 1
            
        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05

    return None

