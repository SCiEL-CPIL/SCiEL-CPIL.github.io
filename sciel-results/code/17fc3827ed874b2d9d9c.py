import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit
    
    n = 20
    vertices = list(range(n))
    
    # Precompute pairs for fast iteration
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_valid(adj):
        """Check if the graph represented by adj bitmasks is K3,3-free."""
        # Iterate over all triples (i, j, k)
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3: 
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def generate_paley_19():
        # Vertices 0..18
        adj = [0] * 19
        # Quadratic residues mod 19
        # x^2 mod 19 for x in 1..9
        residues = set((x*x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u+1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Extend to 20 vertices: vertex 19 is isolated initially
        adj20 = adj[:] + [0]
        return adj20

    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy: 
    # 1. Start with Paley(19) + 1 node, or random dense.
    # 2. Cleanse (remove edges to be valid).
    # 3. Local Search:
    #    a. Add 1 edge if valid.
    #    b. Swap 1 edge for 1 edge (to improve structure/potential).
    #    c. Remove 1 edge, Add 2 edges (net +1).
    
    random.seed(42)
    
    num_starts = 12 # More starts with better search
    
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Initialize adj
        if s == 0:
            adj = generate_paley_19()
        else:
            # Random graph with edge probability 0.6 (denser start)
            adj = [0] * n
            for i, j in all_pairs:
                if random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Make it valid by removing edges
        valid = False
        attempts = 0
        while not valid and time.time() < deadline - 10:
            valid = True
            viol_u, viol_v, viol_w, viol_nbs = -1, -1, -1, 0
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3: continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            valid = False
                            viol_u, viol_v, viol_w = i, j, k
                            viol_nbs = cijk
                            break
                    if not valid: break
                if not valid: break
            if not valid:
                # Remove an edge (viol_u, x) where x is a common neighbor
                # To be more strategic, pick the x that is part of the most violations?
                # For now, random or first found.
                x = (viol_nbs & (-viol_nbs)).bit_length() - 1
                adj[viol_u] &= ~(1 << x)
                adj[x] &= ~(1 << viol_u)
                attempts += 1
                if attempts > 2000: break
        
        if not valid:
            continue
            
        # Phase 2: Local Search Loop
        # We will alternate between adding, swapping, and 2-for-1.
        
        # Helper to count "addable" edges (edges that can be added while maintaining validity)
        # This is expensive to compute fully. 
        # Surrogate: Just try to add.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 2a. Try adding any edge
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u = adj[u]
                old_v = adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
            
            if improved:
                continue # Restart loop as graph changed
            
            # 2b. Try 1-for-1 swaps to improve "potential"
            # Potential surrogate: Number of edges that can be added?
            # Let's just find ANY valid swap that doesn't decrease edge count (obviously same count)
            # but might allow future adds.
            # Heuristic: Try swaps that connect high-degree nodes or balance degrees?
            # Simpler: Random valid swaps.
            
            edges_in_graph = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            
            if not non_edges:
                break
                
            # To save time, don't check ALL swaps. Check a subset.
            # But systematic is better. 
            # Let's check swaps that involve the 'most constrained' nodes?
            # For now, random sample of swaps to find a valid one.
            # If we find a valid swap, accept it with some probability (simulated annealing style?)
            # Or just accept if it allows more adds? Too slow.
            # Let's just accept the first valid swap found in a random permutation.
            
            random.shuffle(edges_in_graph)
            random.shuffle(non_edges)
            
            swap_found = False
            # Limit number of swap attempts to avoid slowing down too much
            for e_remove in edges_in_graph[:50]:
                if swap_found: break
                for e_add in non_edges[:50]:
                    u1, v1 = e_remove
                    u2, v2 = e_add
                    if u1 == u2 and v1 == v2: continue
                    if u1 == v2 and v1 == u2: continue # Same edge reversed? No, simple graph.
                    
                    # Check if removing e_remove and adding e_add results in valid graph
                    # Optimization: Only check if the swap actually changes the graph
                    # (i.e., e_remove is present, e_add is absent) - already checked.
                    
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        swap_found = True
                        improved = True # Force restart to try adds again
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
            
            if not swap_found:
                # 2c. Try 2-for-1 exchange (Remove 1, Add 2)
                # This is more expensive. Only try if stuck.
                # We want to find e1 to remove, and e2, e3 to add, such that graph is valid.
                # Iterate over e1 to remove.
                # For each e1, try to add e2, then e3.
                
                found_2_1 = False
                # Limit scope: Try removing edges incident to high-degree nodes?
                # Or just random sample.
                
                # To be efficient, we can try:
                # 1. Pick a random edge to remove.
                # 2. Try to add 2 edges.
                
                # Let's try a more directed approach:
                # Find a "blocking" configuration.
                # If no single add works, maybe removing one edge unlocks two.
                
                # We will try a limited number of random 2-for-1 attempts.
                for _ in range(20): # 20 attempts
                    if time.time() > deadline - 10: break
                    if not edges_in_graph: break
                    e_remove = random.choice(edges_in_graph)
                    u1, v1 = e_remove
                    
                    # Remove it
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    
                    # Now try to add 2 edges
                    added = 0
                    # Re-calculate non-edges? Or just iterate all pairs.
                    # We need to add 2 edges.
                    # Try adding first edge
                    found_first = False
                    for u2, v2 in all_pairs:
                        if adj[u2] & (1 << v2): continue
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1): continue # Don't re-add removed edge immediately? 
                        # Actually, re-adding is pointless if it was the only thing we removed.
                        
                        old_u2, old_v2 = adj[u2], adj[v2]
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        if is_valid(adj):
                            found_first = True
                            added = 1
                            break
                        else:
                            adj[u2] = old_u2
                            adj[v2] = old_v2
                    
                    if found_first:
                        # Try to add second edge
                        found_second = False
                        for u3, v3 in all_pairs:
                            if adj[u3] & (1 << v3): continue
                            # Avoid adding same edge as first
                            if (u3 == u2 and v3 == v2) or (u3 == v2 and v3 == u2): continue
                            
                            old_u3, old_v3 = adj[u3], adj[v3]
                            adj[u3] |= (1 << v3)
                            adj[v3] |= (1 << u3)
                            
                            if is_valid(adj):
                                found_second = True
                                added = 2
                                break
                            else:
                                adj[u3] = old_u3
                                adj[v3] = old_v3
                        
                        if found_second:
                            found_2_1 = True
                            improved = True
                            break
                        else:
                            # Undo second add? No, it was rolled back.
                            # We need to undo the first add if we don't accept the 2-for-1?
                            # If we don't accept, we must restore original state.
                            # But we are in a "try" loop.
                            # If we fail to add 2, we should revert the whole move (remove e1, add e2).
                            # So we need to undo the first add.
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                            # And re-add e1
                            adj[u1] |= (1 << v1)
                            adj[v1] |= (1 << u1)
                    else:
                        # Undo remove e1
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                    
                    if found_2_1:
                        break
                
                if found_2_1:
                    continue # Restart loop to try adds again
            
            # If no swap and no 2-for-1 found, we are stuck.
            # Break out of local search for this start.
            break
        
        update_best(adj)
        
    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]
        
    return best_edges

# EVOLVE-BLOCK-END
