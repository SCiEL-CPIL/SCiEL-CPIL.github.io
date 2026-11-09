import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def greedy_construct(n, seed=0):
        """
        Greedy randomized construction: assign colors to edges in a specific order,
        choosing the color that minimizes creation of new "forbidden pairs" 
        (pairs with exactly one common neighbor in that color).
        Rejects any color that would immediately create a C4.
        """
        rng = random.Random(seed)
        
        # Track for each color, for each pair of vertices (i,j), how many common neighbors they have
        # common_count[c][i][j] = number of vertices k such that (i,k) and (j,k) are both color c
        # We only care about pairs that could form a C4: if common_count >= 2, we have a C4.
        # We want to avoid creating pairs with common_count = 2.
        
        # For efficiency, track for each color:
        # - adj[c][v] = set of neighbors of v in color c
        # - For counting common neighbors, we can use the adjacency sets
        
        adj = [set() for _ in range(5)]  # adj[c] is not quite right, let's do adj[c][v]
        adj = [[set() for _ in range(n)] for _ in range(5)]
        
        # colors[i*n + j] for i < j, but let's use a dict or 2D array
        # Actually, let's store colors as a list in lexicographic order
        # index = i*n + j - j*(j+1)//2 ... let's just use a 2D list
        color_matrix = [[-1]*n for _ in range(n)]
        
        # Generate all edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # We'll process edges in a randomized order, but with some structure
        # Let's try processing in a specific order: maybe by distance or random
        
        # To be fast, let's precompute for each edge (i,j) and color c, 
        # how many common neighbors i and j would have in color c after adding this edge.
        # Actually, that's expensive. Let's use a simpler heuristic.
        
        # Alternative: for each edge, try each color, and count how many 
        # "new" pairs would be created with exactly 1 common neighbor.
        # A new pair (a,b) is created if a is a neighbor of i in color c and b is a neighbor of j in color c
        # (or vice versa), and (a,b) was not already counted.
        
        # This is getting complex. Let's use a simpler approach:
        # For each edge (i,j), for each color c, compute the number of vertices k 
        # such that (i,k) is color c and (j,k) is color c. If this is >= 2, 
        # then adding (i,j) in color c would create a C4 (since i and j would have 
        # at least 2 common neighbors in color c, plus the edge itself... wait, no.
        # The C4 is formed by i, j, and two common neighbors k1, k2.
        # So if before adding (i,j) in color c, i and j already have >= 2 common neighbors in color c,
        # then adding (i,j) in color c creates a C4.
        # Also, we need to check if adding (i,j) in color c creates new pairs that 
        # will be problematic later.
        
        # Let's just check: for color c, count common neighbors of i and j in color c.
        # If count >= 2, reject.
        # For the cost, count how many new pairs (a,b) would be created where a is neighbor of i 
        # in color c and b is neighbor of j in color c (and a != b, a != i, a != j, b != i, b != j).
        # Each such pair (a,b) would now have i and j as potential common neighbors... 
        # wait, no. If we add (i,j) in color c, then for any a in N_c(i) and b in N_c(j),
        # the pair (a,b) now has one more common neighbor? No, that's not right.
        
        # Let me think again. When we add edge (i,j) in color c:
        # - i and j become neighbors in color c.
        # - For any vertex k, if (i,k) and (j,k) are both color c, then i and j have k as a common neighbor.
        #   If there are >= 2 such k's, we have a C4.
        # - Additionally, for any a in N_c(i) and b in N_c(j) (a != b), the pair (a,b) 
        #   now has... hmm, this doesn't directly create a C4 for (a,b).
        
        # Actually, the C4 constraint is: no two vertices have >= 2 common neighbors in the same color.
        # So when we add (i,j) in color c, we need to check:
        # 1. i and j: do they have >= 2 common neighbors in color c? (excluding each other, 
        #    since the edge (i,j) itself doesn't count as a common neighbor)
        #    Wait, common neighbors of i and j are vertices k such that (i,k) and (j,k) are both color c.
        #    The edge (i,j) is between i and j, so it doesn't make i a common neighbor of itself.
        #    So we just count k != i, j such that (i,k) and (j,k) are both color c.
        #    If this count >= 2, we have a C4.
        # 2. For any other pair (a,b), does adding (i,j) create a new common neighbor for them?
        #    A new common neighbor for (a,b) in color c would be i (if (a,i) and (b,i) are both color c) 
        #    or j (if (a,j) and (b,j) are both color c).
        #    But we're adding (i,j), not (a,i) or (a,j). So adding (i,j) doesn't directly 
        #    create new common neighbors for other pairs.
        
        # Wait, that's not right either. Let me reconsider.
        # When we add edge (i,j) in color c, the only new adjacency is between i and j.
        # For a pair (a,b) to have a new common neighbor, we would need a new edge incident to a or b.
        # But we're adding edge (i,j), which is only incident to i and j.
        # So the only pair whose common neighbor count changes is (i,j) itself.
        
        # Therefore, the check is simply: when adding (i,j) in color c, 
        # count the number of vertices k such that (i,k) and (j,k) are both color c.
        # If this count >= 2, reject color c for edge (i,j).
        
        # For the cost function, we want to minimize the risk of future C4s.
        # One heuristic: minimize the number of vertices k such that (i,k) and (j,k) are both color c.
        # Because if this count is 1, then i and j have exactly 1 common neighbor in color c, 
        # and if we later add another edge that creates another common neighbor, we'll have a C4.
        
        # So the cost for assigning color c to edge (i,j) is:
        # - infinity if common_count >= 2 (would create C4)
        # - common_count if common_count == 1 (risky)
        # - 0 if common_count == 0 (safe)
        
        # But we also need to consider that adding (i,j) might affect the common neighbor 
        # counts of other pairs? No, as argued above, it doesn't.
        
        # Wait, I think I'm missing something. Let me re-examine.
        # The C4 is a cycle of length 4: v1-v2-v3-v4-v1, all edges in the same color.
        # This is equivalent to: there exist two vertices v1, v3 with two common neighbors v2, v4 
        # such that all four edges (v1,v2), (v2,v3), (v3,v4), (v4,v1) are in the same color.
        # So for a pair (v1, v3), if they have two common neighbors v2, v4 in color c, 
        # and all four edges exist in color c, then we have a C4.
        
        # When we add edge (i,j) in color c, we need to check if this completes a C4.
        # A C4 involving edge (i,j) would be: i-j-k-i' where (i,j), (j,k), (k,i') are color c, 
        # and (i',i) is also color c. But that's a 4-cycle i-j-k-i'-i.
        # Alternatively, i-j-i'-k where (i,j), (j,i'), (i',k), (k,i) are color c.
        
        # The standard way to check: for each pair of vertices (u,v), count common neighbors in color c.
        # If any pair has >= 2 common neighbors in color c, and all four edges exist, we have a C4.
        # But actually, if (u,v) have two common neighbors w1, w2 in color c, that means 
        # (u,w1), (v,w1), (u,w2), (v,w2) are all color c. This forms a C4: u-w1-v-w2-u.
        
        # So the constraint is: for every pair (u,v), the number of common neighbors in color c 
        # (vertices w such that (u,w) and (v,w) are both color c) must be <= 1.
        
        # When we add edge (i,j) in color c:
        # - The pair (i,j) now has some common neighbors. We need to check if this count is <= 1.
        # - Does adding (i,j) affect the common neighbor count of any other pair?
        #   For a pair (u,v) where u != i,j and v != i,j: no, because (i,j) is not incident to u or v.
        #   For a pair (i,v) where v != j: the common neighbors of i and v in color c are 
        #   vertices w such that (i,w) and (v,w) are both color c. Adding (i,j) doesn't change 
        #   whether (i,w) or (v,w) is color c for w != j. But wait, j is now a neighbor of i in color c.
        #   So if (v,j) is also color c, then j becomes a common neighbor of i and v.
        #   So the common neighbor count of (i,v) increases by 1 if (v,j) is color c.
        #   Similarly, the common neighbor count of (j,v) increases by 1 if (i,v) is color c.
        
        # So adding (i,j) in color c can increase the common neighbor count of pairs (i,v) and (j,v) 
        # for all v != i,j.
        
        # This makes the cost function more complex. Let me define:
        # - For color c, let N_c(x) be the set of neighbors of x in color c.
        # - When adding (i,j) in color c:
        #   - i and j are added to each other's neighbor sets.
        #   - For each v in N_c(i) (excluding j): v now has i as a neighbor, and if v is also in N_c(j), 
        #     then i and j have v as a common neighbor. Wait, I'm getting confused.
        
        # Let me just implement the check directly:
        # For each color c, maintain adj_c[v] = set of neighbors of v in color c.
        # When considering adding (i,j) in color c:
        #   1. Check if len(adj_c[i] & adj_c[j]) >= 2. If yes, reject.
        #   2. For the cost, we want to estimate the risk. One simple cost: 
        #      len(adj_c[i] & adj_c[j]) (the number of common neighbors, which should be 0 or 1).
        #   3. Additionally, for each v in adj_c[i] (v != j), if v is also in adj_c[j], 
        #      that's already counted. But we also need to consider that adding (i,j) might 
        #      create new common neighbors for other pairs.
        
        # Actually, let's just use a simple heuristic: minimize len(adj_c[i] & adj_c[j]).
        # If it's >= 2, reject. Otherwise, cost = len(adj_c[i] & adj_c[j]).
        
        # Let's implement this.
        
        for edge_idx, (i, j) in enumerate(edges):
            # Check time
            if time.time() > deadline - 5:
                return None
            
            best_color = -1
            best_cost = float('inf')
            
            # Try colors in random order
            color_order = [0, 1, 2, 3, 4]
            rng.shuffle(color_order)
            
            for c in color_order:
                # Count common neighbors of i and j in color c
                common = len(adj[c][i] & adj[c][j])
                
                if common >= 2:
                    continue  # Would create C4
                
                # Cost: number of common neighbors (0 or 1)
                # We prefer 0 over 1
                cost = common
                
                # Additional cost: how many new "at-risk" pairs would be created?
                # When we add (i,j) in color c, for each v in adj[c][i] (v != j),
                # the pair (v, j) now has i as a potential common neighbor? No.
                # Let me think...
                
                # Actually, let's just use cost = common for now.
                
                if cost < best_cost:
                    best_cost = cost
                    best_color = c
            
            if best_color == -1:
                return None  # No valid color
            
            # Assign color
            color_matrix[i][j] = best_color
            color_matrix[j][i] = best_color
            adj[best_color][i].add(j)
            adj[best_color][j].add(i)
        
        # Verify
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    common = len(adj[c][i] & adj[c][j])
                    if common >= 2:
                        return None
        
        # Build colors list in lexicographic order
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(color_matrix[i][j])
        
        return {'n': n, 'colors': colors}
    
    # Try greedy construction for n=26, 25, 24, ...
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = greedy_construct(n, seed=seed)
            if result is not None:
                return result
    
    # Fallback: the known n=11 construction
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
