import random
import time

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    Uses bitset-based adjacency and Simulated Annealing.
    """
    n = 16
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def is_triangle_free(adj):
        for i in range(n):
            # If i is connected to j and k, and j is connected to k, triangle.
            # Check for each i: neighbors of i form an independent set.
            # i.e., for any j, k in N(i), j and k are not connected.
            # Equivalent: (adj[i] & adj[j]) == 0 for all j in N(i)?
            # No, adj[j] contains i.
            # Let's check: for each i, for each j in N(i), N(i) \ {j} must be disjoint from N(j) \ {i}?
            # Simpler: For each i, let N_i = adj[i].
            # For each j in N_i, we need N_i & adj[j] == 0?
            # If k is in N_i and in adj[j], then i-j, j-k, i-k is a triangle.
            # So yes, N_i & adj[j] must be empty for all j in N_i.
            # Wait, adj[j] contains i. So N_i & adj[j] will contain i?
            # No, N_i is set of neighbors of i. i is not in N_i.
            # So we check: for each i, for each j in N_i:
            #   if (N_i & adj[j]) != 0: return False
            # Note: adj[j] contains i. N_i does not contain i.
            # So the intersection checks if any neighbor of i is also a neighbor of j.
            # This is correct.
            N_i = adj[i]
            while N_i:
                # Get lowest set bit
                lsb = N_i & -N_i
                j = lsb.bit_length() - 1
                N_i ^= lsb
                if (adj[i] & adj[j]):
                    return False
        return True

    def is_3_colorable(adj):
        # Backtracking with bitsets
        color = [-1] * n
        # Order vertices by degree (popcount) descending
        order = sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine forbidden colors
            forbidden = 0
            N_v = adj[v]
            while N_v:
                lsb = N_v & -N_v
                u = lsb.bit_length() - 1
                N_v ^= lsb
                if color[u] != -1:
                    forbidden |= (1 << color[u])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)

    # Initialize with a random maximal triangle-free graph
    # Strategy: Start with empty, add random edges if triangle-free.
    # To get a good start, we can try to build a bipartite graph first (max edges)
    # then perturb it.
    
    best_adj = [0] * n
    best_score = -1
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    # Helper to get a random maximal triangle-free graph
    def random_maximal_tf():
        adj = [0] * n
        # Try to add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            # Check if adding (u,v) creates a triangle
            # Common neighbors of u and v must be empty
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Simulated Annealing
    # State: adj
    # Move: Flip an edge (add or remove)
    
    current_adj = random_maximal_tf()
    current_score = count_edges(current_adj)
    
    # Check if current is valid (not 3-colorable)
    # If it is 3-colorable, we need to modify it.
    # For SA, we define "valid" as triangle-free AND not 3-colorable.
    # If triangle-free fails, we must revert.
    # If 3-colorable, we can still keep it in the search space but penalize?
    # No, the task requires the *final* result to be not 3-colorable.
    # We can search in the space of all triangle-free graphs, and only accept as "best" if not 3-colorable.
    # But we need to guide the search towards non-3-colorable graphs.
    
    # Better: Search space is triangle-free graphs.
    # Objective: Maximize edges.
    # Constraint: Not 3-colorable.
    
    # If we just maximize edges, we'll hit K_8,8 (64 edges) which is 3-colorable.
    # We need to find the *largest* triangle-free graph that is *not* 3-colorable.
    
    # Heuristic: Non-3-colorable graphs are "sparse" compared to K_8,8?
    # Actually, the gap is small.
    
    # Let's run SA on the space of triangle-free graphs.
    # Accept moves that maintain triangle-free.
    # Score: Edge count.
    # If the best solution found is 3-colorable, we need to find a nearby non-3-colorable one.
    
    # To ensure we find a non-3-colorable one, we can add a penalty for being 3-colorable?
    # Or, we can run the search, and if the best is 3-colorable, we do a local search
    # specifically to break 3-colorability while maintaining triangle-free and high edge count.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit and T > T_min:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        is_present = (current_adj[u] >> v) & 1
        
        # Check triangle-free constraint for the flip
        if is_present:
            # Remove edge
            # Always valid to remove
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = current_score - 1
        else:
            # Add edge
            if (current_adj[u] & current_adj[v]) != 0:
                continue # Would create triangle
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_score = current_score + 1
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_adj = new_adj
            current_score = new_score
            
            # Check if this is a valid candidate (not 3-colorable)
            # Doing this every step is expensive.
            # We can check every 100 steps or when score improves.
            if current_score > best_score:
                # Check 3-colorability only if we might have a new best
                # But if it's 3-colorable, it's not a valid best.
                # We need to know if it's valid to update best.
                # To save time, we can check 3-colorability only when score > best_score.
                if not is_3_colorable(current_adj):
                    best_adj = current_adj[:]
                    best_score = current_score
        else:
            # Reject move
            pass
            
        T *= cooling
        
        # Occasionally, try to "fix" if we are stuck in 3-colorable region
        # If best_score is low, we might be stuck.
        
    # Final check: if best_adj is empty or 3-colorable, we need to force a non-3-colorable solution.
    # The search above might have found a 3-colorable graph with high edges (e.g., 64) and rejected it as "best"
    # because it's 3-colorable. But it might not have found a non-3-colorable one with high edges.
    
    # To be safe, let's do a targeted search for non-3-colorable graphs.
    # Start from a known non-3-colorable graph (e.g. Grötzsch + extensions) and try to improve.
    
    # If best_score is still -1 or very low, use the fallback from the original code's logic but faster.
    
    # Let's ensure we have at least one valid solution.
    # Generate Grötzsch graph
    def make_grotzsch_adj():
        adj = [0] * 11
        # Cycle 0-1-2-3-4-0
        for i in range(5):
            u = i
            v = (i + 1) % 5
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Mycielski
        for u in range(5):
            v = (u + 1) % 5
            adj[u] |= (1 << (5 + v))
            adj[5 + v] |= (1 << u)
            adj[v] |= (1 << (5 + u))
            adj[5 + u] |= (1 << v)
        for i in range(5):
            adj[10] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
        
        # Extend to 16
        adj16 = [0] * 16
        for i in range(11):
            adj16[i] = adj[i]
        # Add vertices 11-15
        # Connect them to maximize edges while keeping triangle-free and non-3-colorable
        # Simple greedy
        for new_v in range(11, 16):
            # Find neighbors that are independent
            # Try to connect to as many as possible
            # This is a Max Independent Set problem on the subgraph of existing neighbors?
            # No, we need the set of neighbors of new_v to be an independent set in the existing graph.
            # Let S be the set of existing vertices.
            # We want to pick a subset N of S such that N is independent in G[S].
            # And we want to maximize |N|.
            # This is MIS on G[S].
            # G[S] is triangle-free, but not necessarily bipartite.
            # However, S is small (11 vertices).
            # We can solve MIS on 11 vertices quickly.
            
            # Build graph on existing vertices 0..10
            # Actually, we just need to check if a subset is independent.
            # Let's use a simple backtracking for MIS on the subgraph induced by 0..10.
            # But we want to connect new_v to an independent set.
            # So we find the Max Independent Set of the graph induced by 0..10.
            
            # Wait, the neighbors of new_v must be independent in the *current* graph.
            # The current graph on 0..10 is the Grötzsch graph.
            # The MIS of Grötzsch graph?
            # Grötzsch is 4-chromatic, triangle-free.
            # Its independence number is...?
            
            # Let's just try all subsets of 0..10 that are independent.
            # There are 2^11 = 2048 subsets.
            best_subset = 0
            best_size = 0
            for mask in range(1 << 11):
                # Check if mask is independent
                valid = True
                # For each i in mask, check if it has any neighbor in mask
                for i in range(11):
                    if mask & (1 << i):
                        # Check neighbors of i in mask
                        if adj16[i] & mask:
                            # i is connected to someone in mask?
                            # adj16[i] contains i? No.
                            # So if adj16[i] & mask != 0, then i has a neighbor in mask.
                            # But we need to be careful: if i is in mask, and j is in mask, and i-j is edge.
                            # adj16[i] & mask will be non-zero if any neighbor of i is in mask.
                            # This is correct.
                            valid = False
                            break
                if valid:
                    size = bin(mask).count('1')
                    if size > best_size:
                        best_size = size
                        best_subset = mask
            
            # Add edges from new_v to best_subset
            for i in range(11):
                if best_subset & (1 << i):
                    adj16[new_v] |= (1 << i)
                    adj16[i] |= (1 << new_v)
            
            # Check if still non-3-colorable
            if is_3_colorable(adj16):
                # If it became 3-colorable, we might need to remove some edges?
                # Or just accept it for now and hope SA fixes it?
                # For the seed, let's just keep the structure.
                pass
                
        return adj16

    # Run SA starting from Grötzsch extension
    grotzsch_adj = make_grotzsch_adj()
    grotzsch_score = count_edges(grotzsch_adj)
    if not is_3_colorable(grotzsch_adj):
        if grotzsch_score > best_score:
            best_adj = grotzsch_adj
            best_score = grotzsch_score
    
    # If best_adj is still 3-colorable or empty, we have a problem.
    # But Grötzsch is not 3-colorable. The extension might make it 3-colorable?
    # If we add vertices, we might make it 3-colorable.
    # If so, we should try to remove edges to make it non-3-colorable again?
    # No, we want to MAXIMIZE edges.
    # If the extension made it 3-colorable, it's invalid.
    # We need to find a different extension or modify the Grötzsch part.
    
    # The SA loop above should have found a good non-3-colorable graph if it exists.
    # If best_score is -1, it means no non-3-colorable graph was found in SA.
    # This is unlikely.
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
