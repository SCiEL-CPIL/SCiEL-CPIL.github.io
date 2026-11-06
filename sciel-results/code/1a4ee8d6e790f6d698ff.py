import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 20
    
    # 1. Structured Initialization using Projective Plane of Order 4
    # The projective plane of order 4 has q+1 = 5 points on each line and 5 lines through each point.
    # Total points = q^2 + q + 1 = 16 + 4 + 1 = 21.
    # We can take the first 20 points.
    # The dual graph (incidence graph) is bipartite and 5-regular.
    # However, we want a simple graph on 20 vertices.
    # A better structured graph for K2,3-free is the complement of a sparse graph?
    # No, let's look at the properties.
    # K2,3-free means no two vertices share 3 neighbors.
    # In the incidence graph of PG(2,4), vertices are points and lines.
    # If we just take the 20 points, they don't have edges in the incidence graph itself (it's bipartite).
    # We need to construct a graph ON the 20 vertices.
    
    # Alternative: Use a known dense K2,3-free graph.
    # The Petersen graph is K2,3-free? No, it's small.
    # Let's try a random regular graph initialization with high degree, then clean.
    # But the directive asks for a STRUCTURAL change.
    # "Different problem framing".
    
    # Framing: Instead of searching for edges, search for NON-edges to REMOVE from the complete graph?
    # Or, use a deterministic greedy algorithm that is much faster than SA.
    
    # Let's try a specific deterministic construction:
    # Vertices 0..19.
    # Connect i to i+1, i+2, i+3 mod 20?
    # Let's check K2,3 constraint for circulant graphs.
    # If we use a circulant graph C20(S), two vertices u, v have common neighbors depending on the difference d = v-u.
    # Common neighbors of 0 and d are x such that x in S and x-d in S.
    # So |S ∩ (S+d)| must be <= 2 for all d != 0.
    
    # Let's try to find a good set S for C20.
    # Max degree?
    # If S = {1, 2, 3}, then for d=1: S ∩ (S+1) = {1,2,3} ∩ {2,3,4} = {2,3} (size 2). OK.
    # For d=2: {1,2,3} ∩ {3,4,5} = {3} (size 1). OK.
    # For d=3: {1,2,3} ∩ {4,5,6} = {} (size 0). OK.
    # For d=4: {1,2,3} ∩ {5,6,7} = {} (size 0). OK.
    # Due to symmetry, we only need to check d up to 10.
    # S = {1, 2, 3} gives degree 6. Total edges = 20*6/2 = 60.
    # Can we do better?
    # Try S = {1, 2, 3, 4}.
    # d=1: {1,2,3,4} ∩ {2,3,4,5} = {2,3,4} (size 3). VIOLATION.
    # So max consecutive integers is 3.
    
    # What if S is not consecutive?
    # S = {1, 2, 4, 8}?
    # d=1: {1,2,4,8} ∩ {2,3,5,9} = {2} (1).
    # d=2: {1,2,4,8} ∩ {3,4,6,10} = {4} (1).
    # d=3: {1,2,4,8} ∩ {4,5,7,11} = {4} (1).
    # d=4: {1,2,4,8} ∩ {5,6,8,12} = {8} (1).
    # d=5: {1,2,4,8} ∩ {6,7,9,13} = {} (0).
    # d=6: {1,2,4,8} ∩ {7,8,10,14} = {8} (1).
    # d=7: {1,2,4,8} ∩ {8,9,11,15} = {8} (1).
    # d=8: {1,2,4,8} ∩ {9,10,12,16} = {} (0).
    # d=9: {1,2,4,8} ∩ {10,11,13,17} = {} (0).
    # d=10: {1,2,4,8} ∩ {11,12,14,18} = {} (0).
    # All <= 2. Degree 8. Edges = 20*8/2 = 80.
    
    # Can we add more?
    # Try S = {1, 2, 4, 8, 12}?
    # d=1: {1,2,4,8,12} ∩ {2,3,5,9,13} = {2} (1).
    # d=2: {1,2,4,8,12} ∩ {3,4,6,10,14} = {4} (1).
    # d=3: {1,2,4,8,12} ∩ {4,5,7,11,15} = {4} (1).
    # d=4: {1,2,4,8,12} ∩ {5,6,8,12,16} = {8,12} (2).
    # d=5: {1,2,4,8,12} ∩ {6,7,9,13,17} = {} (0).
    # d=6: {1,2,4,8,12} ∩ {7,8,10,14,18} = {8} (1).
    # d=7: {1,2,4,8,12} ∩ {8,9,11,15,19} = {8} (1).
    # d=8: {1,2,4,8,12} ∩ {9,10,12,16,20} = {12} (1).
    # d=9: {1,2,4,8,12} ∩ {10,11,13,17,21} = {} (0).
    # d=10: {1,2,4,8,12} ∩ {11,12,14,18,22} = {12} (1).
    # All <= 2. Degree 10. Edges = 20*10/2 = 100.
    
    # This is a HUGE jump from the typical SA results (which are often around 70-80).
    # Let's verify this graph is valid.
    # S = {1, 2, 4, 8, 12}
    # Vertices 0..19.
    # Edge (u, v) exists if (v-u) mod 20 in S or (u-v) mod 20 in S.
    # Since S contains -12=8, -8=12, -4=16 (not in S), -2=18 (not in S), -1=19 (not in S).
    # Wait, S must be symmetric for an undirected graph?
    # If I define adjacency by (v-u) mod 20 in S, then edge (u,v) exists if v-u in S.
    # Edge (v,u) exists if u-v in S => -(v-u) in S.
    # So S must be symmetric: if x in S, then 20-x in S.
    # My S = {1, 2, 4, 8, 12}.
    # 20-1 = 19 (not in S).
    # 20-2 = 18 (not in S).
    # 20-4 = 16 (not in S).
    # 20-8 = 12 (in S).
    # 20-12 = 8 (in S).
    # So this S defines a DIRECTED graph or a graph where edges are not symmetric unless we include both.
    # For an undirected simple graph, we need to define the set of differences D such that
    # (u,v) is an edge if (v-u) mod 20 in D.
    # For the graph to be simple and undirected, we usually pick D from {1, ..., 10}.
    # If d in D, then (u, u+d) is an edge. This covers all pairs with difference d.
    # The condition for K2,3-free is |D ∩ (D+d)| <= 2 for all d in {1, ..., 10}.
    # Note: The intersection check needs to be done on the set of differences.
    
    # Let's re-evaluate S = {1, 2, 4, 8} with the correct check.
    # D = {1, 2, 4, 8}.
    # d=1: D ∩ (D+1 mod 20) = {1,2,4,8} ∩ {2,3,5,9} = {2}. Size 1.
    # d=2: D ∩ (D+2 mod 20) = {1,2,4,8} ∩ {3,4,6,10} = {4}. Size 1.
    # d=3: D ∩ (D+3 mod 20) = {1,2,4,8} ∩ {4,5,7,11} = {4}. Size 1.
    # d=4: D ∩ (D+4 mod 20) = {1,2,4,8} ∩ {5,6,8,12} = {8}. Size 1.
    # d=5: D ∩ (D+5 mod 20) = {1,2,4,8} ∩ {6,7,9,13} = {}. Size 0.
    # d=6: D ∩ (D+6 mod 20) = {1,2,4,8} ∩ {7,8,10,14} = {8}. Size 1.
    # d=7: D ∩ (D+7 mod 20) = {1,2,4,8} ∩ {8,9,11,15} = {8}. Size 1.
    # d=8: D ∩ (D+8 mod 20) = {1,2,4,8} ∩ {9,10,12,16} = {}. Size 0.
    # d=9: D ∩ (D+9 mod 20) = {1,2,4,8} ∩ {10,11,13,17} = {}. Size 0.
    # d=10: D ∩ (D+10 mod 20) = {1,2,4,8} ∩ {11,12,14,18} = {}. Size 0.
    # Valid! Degree 8. Edges 80.
    
    # Can we add 12 to D?
    # D = {1, 2, 4, 8, 12}. But 12 > 10.
    # In a circulant graph C20(D), D is a subset of {1, ..., 10}.
    # If we want difference 12, it is equivalent to difference 8 (since 20-12=8).
    # So D={1,2,4,8} already includes the "8" connection.
    # What if we try D = {1, 2, 4, 8, 5}?
    # d=1: {1,2,4,5,8} ∩ {2,3,5,6,9} = {2,5}. Size 2.
    # d=2: {1,2,4,5,8} ∩ {3,4,6,7,10} = {4}. Size 1.
    # d=3: {1,2,4,5,8} ∩ {4,5,7,8,11} = {4,5,8}. Size 3. VIOLATION.
    
    # Try D = {1, 2, 4, 8, 6}?
    # d=1: {1,2,4,6,8} ∩ {2,3,5,7,9} = {2}. Size 1.
    # d=2: {1,2,4,6,8} ∩ {3,4,6,8,10} = {4,6,8}. Size 3. VIOLATION.
    
    # Try D = {1, 2, 4, 8, 9}?
    # d=1: {1,2,4,8,9} ∩ {2,3,5,9,10} = {2,9}. Size 2.
    # d=2: {1,2,4,8,9} ∩ {3,4,6,10,11} = {4}. Size 1.
    # d=3: {1,2,4,8,9} ∩ {4,5,7,11,12} = {4}. Size 1.
    # d=4: {1,2,4,8,9} ∩ {5,6,8,12,13} = {8}. Size 1.
    # d=5: {1,2,4,8,9} ∩ {6,7,9,13,14} = {9}. Size 1.
    # d=6: {1,2,4,8,9} ∩ {7,8,10,14,15} = {8}. Size 1.
    # d=7: {1,2,4,8,9} ∩ {8,9,11,15,16} = {8,9}. Size 2.
    # d=8: {1,2,4,8,9} ∩ {9,10,12,16,17} = {9}. Size 1.
    # d=9: {1,2,4,8,9} ∩ {10,11,13,17,18} = {}. Size 0.
    # d=10: {1,2,4,8,9} ∩ {11,12,14,18,19} = {}. Size 0.
    # Valid! Degree 9. Edges = 20*9/2 = 90.
    
    # Can we add more?
    # Try adding 3 to D={1,2,4,8,9}?
    # D={1,2,3,4,8,9}.
    # d=1: {1,2,3,4,8,9} ∩ {2,3,4,5,9,10} = {2,3,4,9}. Size 4. VIOLATION.
    
    # Try adding 5 to D={1,2,4,8,9}?
    # D={1,2,4,5,8,9}.
    # d=1: {1,2,4,5,8,9} ∩ {2,3,5,6,9,10} = {2,5,9}. Size 3. VIOLATION.
    
    # Try adding 6 to D={1,2,4,8,9}?
    # D={1,2,4,6,8,9}.
    # d=1: {1,2,4,6,8,9} ∩ {2,3,5,7,9,10} = {2,9}. Size 2.
    # d=2: {1,2,4,6,8,9} ∩ {3,4,6,8,10,11} = {4,6,8}. Size 3. VIOLATION.
    
    # Try adding 7 to D={1,2,4,8,9}?
    # D={1,2,4,7,8,9}.
    # d=1: {1,2,4,7,8,9} ∩ {2,3,5,8,9,10} = {2,8,9}. Size 3. VIOLATION.
    
    # Try adding 10 to D={1,2,4,8,9}?
    # D={1,2,4,8,9,10}.
    # d=1: {1,2,4,8,9,10} ∩ {2,3,5,9,10,11} = {2,9,10}. Size 3. VIOLATION.
    
    # So 90 edges is a strong deterministic start.
    # Let's see if we can get 91+ via local search from this start.
    # The previous SA started from random (approx 70 edges) and climbed to ~80-90.
    # Starting at 90 should allow it to find 91-95 quickly.
    
    # Let's implement a fast greedy adder on top of this circulant graph.
    
    def build_circulant(D):
        adj = [0] * n
        for u in range(n):
            for d in D:
                v = (u + d) % n
                if v > u:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                elif v < u:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                # if v == u, self loop, ignore
        return adj
    
    D = [1, 2, 4, 8, 9]
    adj = build_circulant(D)
    
    # Verify validity of initial graph
    def is_valid(adj):
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] & adj[v]).bit_count() > 2:
                    return False
        return True
    
    # It should be valid.
    
    # Now, perform a deterministic greedy addition of missing edges.
    # Sort missing edges by "potential" (e.g., number of conflicts they would cause if added? No, we want to add valid ones).
    # We want to add edges that are valid in the current graph.
    # To maximize future potential, maybe add edges that connect high-degree vertices?
    # Or just iterate through all missing edges and add if valid.
    # Order matters. Let's try a few orders.
    
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not ((adj[u] >> v) & 1):
                missing_edges.append((u, v))
    
    # Strategy 1: Add in order of increasing (deg(u) + deg(v))
    # Strategy 2: Add in order of decreasing (deg(u) + deg(v))
    # Strategy 3: Random order
    
    def try_add_edges(adj, edge_list):
        added = 0
        for u, v in edge_list:
            if ((adj[u] >> v) & 1):
                continue
            # Check if adding (u,v) is valid
            # 1. Common neighbors of u and v must be <= 2
            if (adj[u] & adj[v]).bit_count() > 2:
                continue
            # 2. For all w != u, v, adding (u,v) increases common neighbors of (u,w) and (v,w) by 1 if the other is a neighbor.
            # Specifically:
            # new_common(u, w) = old_common(u, w) + (1 if v in N(w) else 0)
            # new_common(v, w) = old_common(v, w) + (1 if u in N(w) else 0)
            # We need new_common <= 2.
            
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check pair (u, w)
                # v is a neighbor of u? Yes, we are adding it.
                # Is v a neighbor of w?
                if (adj[w] >> v) & 1:
                    if (adj[u] & adj[w]).bit_count() > 1: # Since +1 will make it >2 if it was >=2
                        # Wait. If current common is 2, adding 1 makes 3. Invalid.
                        # If current common is 1, adding 1 makes 2. Valid.
                        # So we require current common <= 1.
                        valid = False
                        break
                # Check pair (v, w)
                # u is a neighbor of v? Yes.
                # Is u a neighbor of w?
                if (adj[w] >> u) & 1:
                    if (adj[v] & adj[w]).bit_count() > 1:
                        valid = False
                        break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return adj, added
    
    # Let's try multiple passes with different orderings to see if we can get stuck-free.
    # Since the graph is dense, the order might matter less, but let's try.
    
    # Pass 1: Sorted by index
    adj1, _ = try_add_edges(adj[:], missing_edges)
    
    # Pass 2: Randomized
    adj2 = adj[:]
    random.shuffle(missing_edges)
    adj2, _ = try_add_edges(adj2, missing_edges)
    
    # Pass 3: Sorted by degree sum (descending)
    deg = [a.bit_count() for a in adj]
    sorted_missing = sorted(missing_edges, key=lambda x: deg[x[0]] + deg[x[1]], reverse=True)
    adj3, _ = try_add_edges(adj[:], sorted_missing)
    
    # Pick the best
    best_adj = max([adj1, adj2, adj3], key=lambda a: sum(x.bit_count() for x in a)//2)
    
    # Now, do a quick local search (swap) to break ties or find +1.
    # But the greedy adder is very fast.
    # Let's do a few rounds of "try to add any valid edge" until none found.
    # The above try_add_edges does this in one pass. If an edge was invalid, it might become valid after other additions?
    # No, adding edges only INCREASES common neighbor counts. So if an edge is invalid now, it will never become valid later.
    # So one pass is sufficient for greedy addition.
    
    # However, REMOVING an edge could make another edge valid.
    # So we can do:
    # 1. Greedy Add
    # 2. Try Swaps: Remove one edge, then Greedy Add. If count increases, keep.
    
    current_adj = best_adj
    current_count = sum(a.bit_count() for a in current_adj) // 2
    
    existing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (current_adj[u] >> v) & 1:
                existing_edges.append((u, v))
    
    # Try a limited number of swaps
    swaps_to_try = 500
    for i in range(swaps_to_try):
        if time.time() > deadline - 5:
            break
            
        # Pick a random edge to remove
        ru, rv = random.choice(existing_edges)
        
        # Save state
        temp_adj = current_adj[:]
        
        # Remove edge
        temp_adj[ru] &= ~(1 << rv)
        temp_adj[rv] &= ~(1 << ru)
        
        # Try to add edges
        # We need to find valid edges in temp_adj.
        # To be fast, we can just iterate through all missing edges in temp_adj.
        # Or, we can be smarter: only try edges that were previously blocked by the removed edge?
        # That's complex. Just do a fast greedy pass.
        
        # Optimization: Only check edges that are "close" to ru, rv?
        # No, just do the full pass but it's O(E_missing * N).
        # E_missing is small (~100). N=20. 2000 ops. Very fast.
        
        new_adj, added = try_add_edges(temp_adj, missing_edges)
        
        new_count = sum(a.bit_count() for a in new_adj) // 2
        
        if new_count > current_count:
            current_adj = new_adj
            current_count = new_count
            # Rebuild existing edges list? Or just keep it and filter?
            # For simplicity, rebuild existing_edges if we improve.
            existing_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (current_adj[u] >> v) & 1:
                        existing_edges.append((u, v))
        
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (current_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")
