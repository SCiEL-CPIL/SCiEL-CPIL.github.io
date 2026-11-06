import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def count_edges_in_set(adj, mask, n):
        """Count number of edges in the induced subgraph of vertices in mask."""
        count = 0
        # Iterate over bits in mask
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            m ^= lsb
            # Count neighbors of i that are in mask
            count += (adj[i] & mask).bit_count()
        return count // 2

    def count_triangles_in_set(adj, mask, n):
        """Count number of triangles in the induced subgraph of vertices in mask."""
        # A triangle is a set of 3 vertices all connected.
        # We can count by iterating over edges in the induced subgraph and checking common neighbors?
        # Or simpler: For each vertex i in mask, count triangles involving i where i is the smallest index?
        # To avoid double counting, we can just count all triangles and divide by 3? No, that's complex with masks.
        # Let's use the standard O(N^3/64) or similar.
        # Since mask size is small (intersection of neighborhoods), we can extract the subgraph.
        # However, for speed, let's use a property:
        # Number of triangles in G[S] = 1/3 * sum_{i in S} (number of edges in G[N(i) ∩ S])? No.
        # Let's just do a fast loop.
        verts = []
        m = mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        
        # For each pair i, j in verts, if edge exists, count common neighbors k in verts with k > j?
        # This is O(|S|^3). If |S| is ~10-20, this is fast.
        count = 0
        k = len(verts)
        for i in range(k):
            u = verts[i]
            for j in range(i+1, k):
                v = verts[j]
                if (adj[u] >> v) & 1:
                    # Count common neighbors in verts[j+1:]
                    common = adj[u] & adj[v]
                    # Mask out bits not in verts?
                    # Actually, we just need to count how many w in verts[j+1:] are in common.
                    # We can build a mask for verts[j+1:]
                    # Or just iterate.
                    # Since k is small, iterating is fine.
                    for l in range(j+1, k):
                        w = verts[l]
                        if (common >> w) & 1:
                            count += 1
        return count

    def count_k4_in_set(adj, mask, n):
        """Count number of K4s in the induced subgraph of vertices in mask."""
        # Similar to triangles, but for K4s.
        # A K4 is a set of 4 vertices all connected.
        # Count by iterating over edges (u,v) in induced subgraph, and counting edges in N(u)∩N(v)∩S with indices > v?
        # Let's extract vertices.
        verts = []
        m = mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        
        k = len(verts)
        count = 0
        # Iterate over all subsets of 4? No, too slow if |S| is large.
        # Iterate over pairs (u, v) in S such that (u,v) is an edge.
        # Then count edges in (N(u) ∩ N(v) ∩ S) where both endpoints are > v (to avoid double counting).
        # Wait, the standard way to count cliques is:
        # For each edge (u,v) with u < v, count edges in N(u) ∩ N(v) ∩ {w | w > v}.
        # Let's implement that.
        for i in range(k):
            u = verts[i]
            for j in range(i+1, k):
                v = verts[j]
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # We need to count edges (w, z) in common such that w, z > v and w < z.
                    # Actually, the condition "w > v" is sufficient if we count all edges in the induced subgraph of common & mask_greater_than_v.
                    # Let's build mask_greater_than_v.
                    mask_gt_v = 0
                    for l in range(j+1, k):
                        w = verts[l]
                        mask_gt_v |= (1 << w)
                    
                    sub_mask = common & mask_gt_v
                    count += count_edges_in_set(adj, sub_mask, n)
        return count

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_valid(adj, n):
        # Check if no K4 and no I6
        # K4 check
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    # Check if any edge in common
                    # If there is an edge (k, l) in common, then i,j,k,l form K4.
                    # We just need to check if count_edges_in_set(adj, common, n) > 0
                    # But we only need to check if there is ANY K4.
                    # Optimization: just check if any pair in common is connected.
                    # This is O(n^2) per edge, total O(n^4).
                    # Better: use the count function but break early?
                    # For final validation, O(n^4) is fine for n=35.
                    if count_edges_in_set(adj, common, n) > 0:
                        return False
        # I6 check: K6 in complement
        # We can use a similar logic.
        # Or just use a brute force backtracking for I6?
        # Let's use the clique counting on complement.
        # Construct complement adj?
        # Or just check directly.
        # Let's write a specific I6 checker.
        # Find all independent sets of size 6.
        # This is equivalent to K6 in ~G.
        
        # Efficient check:
        # For each set of 5 vertices, check if they are independent, then see if any 6th vertex is independent from them.
        # Too slow.
        
        # Let's use the K4 counting logic on the complement graph logic.
        # But we don't have complement adj stored.
        # Let's just do a simple backtracking search for I6.
        # n is small (<=40).
        
        def find_i6():
            # Backtracking
            # Order vertices by degree?
            # Simple recursive search
            # Pruning: if remaining vertices + current set size < 6, return False.
            # If current set size == 6, return True.
            
            # To make it fast, we can use bitmasks.
            # available = mask of all vertices
            # current = mask of selected vertices
            
            # We want to find if there is an independent set of size 6.
            # This is the same as finding a clique of size 6 in the complement.
            
            # Let's implement a fast clique finder for the complement.
            # We can reuse count_k4_in_set? No, we need K6.
            
            # Let's just use a simple recursive search with bitmasks.
            # This is only called at the end or when cost is 0.
            
            # Actually, the local search maintains the count.
            # So we only call is_valid if the tracked count is 0.
            # So this function should be robust.
            
            # Let's just do a brute force check for I6 using combinations if n is small,
            # or a backtracking.
            
            # Backtracking for Independent Set of size 6.
            # Sort vertices by degree in G (lower degree is better for independent set? No, higher degree in ~G is better for clique).
            # Let's just do it.
            
            # We'll use a simple DFS.
            # To speed up, we can check if the maximum possible size of independent set in the remaining graph is < 6.
            # Greedy bound?
            
            # For now, let's just check all combinations of 6 if n is small, else use a smarter method.
            # But n can be 35. C(35, 6) is ~1.6 million. That's fast in Python if optimized?
            # 1.6M * 6 checks = ~10M ops. Might be too slow if called many times.
            # But we only call it when cost is 0.
            
            # Let's implement a backtracking with bitmasks.
            # available_mask
            # current_size
            
            # We can use the fact that if we have a candidate set, we only consider neighbors in ~G.
            
            # Let's use a simple recursive function.
            # To avoid deep recursion overhead, we can use a stack or just recursion (depth 6 is fine).
            
            def dfs(available, size):
                if size == 6:
                    return True
                if available == 0:
                    return False
                # If popcount(available) < 6 - size, return False
                if available.bit_count() < 6 - size:
                    return False
                
                # Pick a vertex
                # To prune, pick a vertex with high degree in ~G (low degree in G)?
                # Or just pick the lowest bit.
                lsb = available & -available
                v = lsb.bit_length() - 1
                
                # Option 1: Include v
                # New available: neighbors of v in ~G that are in available and > v (to avoid permutations)
                # Actually, to avoid permutations, we only consider vertices > v.
                # Let's mask out bits <= v.
                mask_gt_v = available & ~((1 << (v + 1)) - 1)
                
                # Neighbors of v in ~G:
                # ~adj[v] & mask_gt_v
                # Note: ~adj[v] includes bits outside n, so mask with (1<<n)-1
                comp_neigh = (~adj[v]) & mask_gt_v & ((1 << n) - 1)
                
                if dfs(comp_neigh, size + 1):
                    return True
                
                # Option 2: Exclude v
                # New available: available without v, and without vertices <= v?
                # No, we just remove v.
                # But to maintain the "only consider > v" property for the next step,
                # we should remove all vertices <= v?
                # Standard Bron-Kerbosch style:
                # P = available
                # Choose v from P.
                # Recurse on P ∩ N(v)
                # Remove v from P.
                # Recurse on P \ {v}
                
                # My previous logic:
                # If I include v, I can only pick from N(v) ∩ {w | w > v}.
                # If I exclude v, I can still pick from {w | w > v} minus v?
                # No, if I exclude v, I can pick any w > v.
                # So:
                
                # Wait, the standard algorithm for finding a clique of size k:
                # expand(R, P, X)
                # If |R| == k: return True
                # For each v in P:
                #   expand(R ∪ {v}, P ∩ N(v), X ∩ N(v))
                #   P = P \ {v}
                #   X = X ∪ {v}
                
                # We don't need X if we just want to find ONE.
                # And we can prune if |P| + |R| < k.
                
                # Let's rewrite dfs to follow this.
                # But we need to be careful with the "v" choice.
                # We can just pick the first available vertex.
                
                # Let's try a different approach for speed:
                # Since we only call this when we think cost is 0, it's rare.
                # Let's just use itertools.combinations for n <= 30.
                # For n > 30, use the backtracking.
                
                # Actually, the local search count should be accurate.
                # If count_i6 == 0, there is no I6.
                # So we might not even need this check if the delta calculation is correct.
                # But to be safe, let's do a quick check.
                
                # Let's just return True for now and rely on the count?
                # No, the task requires "checked exactly".
                # So we must verify.
                
                # Let's implement the backtracking properly.
                return False
            
            # We need a better dfs.
            # Let's define it outside or use a closure.
            
            # Actually, I can just check if count_i6 is 0 in the main loop.
            # If it is 0, I will run this verification.
            
            # Let's implement a simple verification.
            # If n is small, use combinations.
            if n <= 25:
                from itertools import combinations
                for combo in combinations(range(n), 6):
                    is_ind = True
                    for i in range(6):
                        for j in range(i+1, 6):
                            u, v = combo[i], combo[j]
                            if (adj[u] >> v) & 1:
                                is_ind = False
                                break
                        if not is_ind:
                            break
                    if is_ind:
                        return False
                return True
            else:
                # Use backtracking
                # We'll use a simple recursive function with bitmasks.
                # To avoid recursion depth issues (though depth 6 is fine), 
                # and to be fast, we'll use the property that we only pick increasing vertices.
                
                def check_i6_backtrack(available, size):
                    if size == 6:
                        return True
                    if available == 0:
                        return False
                    if available.bit_count() < 6 - size:
                        return False
                    
                    lsb = available & -available
                    v = lsb.bit_length() - 1
                    
                    # Mask for vertices > v
                    mask_gt_v = available & ~((1 << (v + 1)) - 1)
                    
                    # 1. Include v
                    # Next candidates must be neighbors of v in ~G and > v
                    comp_neigh = (~adj[v]) & mask_gt_v & ((1 << n) - 1)
                    if check_i6_backtrack(comp_neigh, size + 1):
                        return True
                    
                    # 2. Exclude v
                    # Next candidates are just mask_gt_v (all vertices > v, excluding v which is not in mask_gt_v anyway)
                    # Wait, if we exclude v, we can still pick any vertex > v.
                    # So we call with mask_gt_v.
                    if check_i6_backtrack(mask_gt_v, size):
                        return True
                    
                    return False
                
                if check_i6_backtrack((1 << n) - 1, 0):
                    return False
                return True

    best_n = 0
    best_result = None
    
    # Try to find a valid graph for n=35, 30, 25
    for target_n in [35, 30, 25, 20]:
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        # For K4-free, p should be low.
        # For no I6, p should be high enough.
        # Turan graph T_3(n) is K4-free.
        # Let's start with a random graph with p=0.3
        
        p = 0.3
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Initial counts
        current_k4 = 0
        current_i6 = 0
        
        # Calculate initial counts
        # K4
        for i in range(target_n):
            for j in range(i+1, target_n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    current_k4 += count_edges_in_set(adj, common, target_n)
        # I6 (K6 in complement)
        # We can calculate this by counting K6 in the complement.
        # Or just use the same logic:
        # For each pair (i,j) non-edge, count edges in common non-neighbors?
        # No, K6 in ~G.
        # Let's just use a slower method for initial count since it's done once.
        # Or use the incremental logic to build it up?
        # Let's just use a function that counts K6 in a graph defined by adj_mask.
        
        def count_k6_in_complement(adj, n):
            # Count K6 in ~G
            # This is equivalent to counting independent sets of size 6 in G.
            # We can use the same "count edges in set" logic but for K6.
            # This is expensive.
            # Let's just do a brute force check for I6 initially?
            # Or use the backtracking?
            # Let's use a simple O(n^6) loop? No.
            
            # Let's use the property:
            # Count K6 in ~G.
            # We can iterate over subsets of 6? C(35, 6) is 1.6M.
            # For each subset, check if it is an independent set.
            # This is fast enough for initialization.
            
            count = 0
            # Use combinations
            from itertools import combinations
            for combo in combinations(range(n), 6):
                is_ind = True
                for i in range(6):
                    for j in range(i+1, 6):
                        u, v = combo[i], combo[j]
                        if (adj[u] >> v) & 1:
                            is_ind = False
                            break
                    if not is_ind:
                        break
                if is_ind:
                    count += 1
            return count

        current_i6 = count_k6_in_complement(adj, target_n)
        current_cost = current_k4 + current_i6
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        
        while T > min_T and time.time() < deadline - 5:
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            if (adj[u] >> v) & 1:
                # Remove edge
                # Delta K4: - (number of K4s containing u,v)
                # K4s containing u,v: {u,v} + {w,z} where w,z in N(u)∩N(v) and (w,z) is edge.
                # So count = count_edges_in_set(adj, N(u)∩N(v), n)
                common = adj[u] & adj[v]
                delta_k4 = -count_edges_in_set(adj, common, target_n)
                
                # Delta I6: + (number of I6s containing u,v)
                # I6s containing u,v: {u,v} + {w,x,y,z} where w,x,y,z in N^c(u)∩N^c(v) and form I4 in G (K4 in ~G).
                # So count = count_k4_in_set(complement_adj, N^c(u)∩N^c(v), n)
                # We need count_k4_in_set for the complement graph.
                # Let's implement a helper that takes adj and mask and counts K4 in the induced subgraph of the COMPLEMENT of adj restricted to mask?
                # No, let's just count K4 in ~G induced by mask.
                
                # To do this, we can create a temporary adj for the complement?
                # Or modify count_k4_in_set to take a "is_edge" function?
                # Let's write a specific function: count_k4_in_complement_induced(adj, mask, n)
                
                def count_k4_in_comp_induced(adj, mask, n):
                    # Count K4s in ~G induced by mask
                    # Equivalent to counting independent sets of size 4 in G induced by mask.
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        verts.append(lsb.bit_length() - 1)
                        m ^= lsb
                    
                    k = len(verts)
                    count = 0
                    # Iterate over pairs (u, v) in verts.
                    # If (u, v) is NOT an edge in G, then they are connected in ~G.
                    # Then we need to count edges in ~G in N~(u) ∩ N~(v) ∩ mask_gt_v.
                    # N~(u) in G is ~adj[u].
                    
                    for i in range(k):
                        u = verts[i]
                        for j in range(i+1, k):
                            v = verts[j]
                            if not ((adj[u] >> v) & 1):
                                # u, v connected in ~G
                                # Common neighbors in ~G:
                                # ~adj[u] & ~adj[v] & mask
                                common_comp = (~adj[u]) & (~adj[v]) & mask
                                # We need to count edges in ~G in the induced subgraph of common_comp & mask_gt_v
                                # This is count_edges_in_comp_induced?
                                # No, we need to count K2s (edges) in ~G.
                                # Which is count of pairs (w, z) in common_comp & mask_gt_v such that (w, z) is NOT an edge in G.
                                
                                mask_gt_v = 0
                                for l in range(j+1, k):
                                    w = verts[l]
                                    mask_gt_v |= (1 << w)
                                    
                                sub_mask = common_comp & mask_gt_v
                                
                                # Count non-edges in G in sub_mask
                                # Total pairs - edges in G
                                num_verts = sub_mask.bit_count()
                                total_pairs = num_verts * (num_verts - 1) // 2
                                edges_in_g = count_edges_in_set(adj, sub_mask, n)
                                non_edges_in_g = total_pairs - edges_in_g
                                count += non_edges_in_g
                    return count

                non_common = (~adj[u]) & (~adj[v]) & ((1 << target_n) - 1)
                # Remove u and v from non_common?
                # N^c(u) does not contain u.
                # But we need to be careful.
                # The set of vertices w such that w is not adjacent to u and not adjacent to v.
                # This set does not include u or v.
                # So non_common is correct.
                
                delta_i6 = count_k4_in_comp_induced(adj, non_common, target_n)
                
                # Apply flip
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
            else:
                # Add edge
                # Delta K4: + (number of K4s containing u,v)
                # K4s containing u,v: {u,v} + {w,z} where w,z in N(u)∩N(v) and (w,z) is edge.
                # Note: After adding the edge, the count is the same as before?
                # No, the edge (u,v) is part of the K4.
                # The triangles in N(u)∩N(v) are the same.
                # So delta_k4 = + count_edges_in_set(adj, N(u)∩N(v), n)
                common = adj[u] & adj[v]
                delta_k4 = count_edges_in_set(adj, common, target_n)
                
                # Delta I6: - (number of I6s containing u,v)
                # I6s containing u,v: {u,v} + {w,x,y,z} in N^c(u)∩N^c(v) forming I4 in G.
                # Adding edge (u,v) destroys these I6s.
                # So delta_i6 = - count_k4_in_comp_induced(adj, N^c(u)∩N^c(v), n)
                
                non_common = (~adj[u]) & (~adj[v]) & ((1 << target_n) - 1)
                delta_i6 = -count_k4_in_comp_induced(adj, non_common, target_n)
                
                # Apply flip
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_cost = current_cost + delta_k4 + delta_i6
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost == 0:
                    # Verify
                    if is_valid(adj, target_n):
                        return {'n': target_n, 'edges': to_edges(adj, target_n)}
                    else:
                        # If invalid, reset counts?
                        # Just continue search.
                        pass
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
            steps += 1
            if steps % 1000 == 0:
                 if time.time() > deadline - 5:
                     break

    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': to_edges(adj, 17)}
