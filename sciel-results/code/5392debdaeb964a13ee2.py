import time
import random

def construct():
    n = 15
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    # Bitmask helpers for fast adjacency
    # adj[i] is a bitmask where bit j is 1 if (i, j) is an edge
    
    def get_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]
        
    def is_valid(adj):
        # adj is list of int bitmasks
        deg = get_degrees(adj)
        # Check all edges (u, v)
        for u in range(n):
            mask_u = adj[u]
            # Iterate neighbors v of u
            m = mask_u
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                # For edge (u, v), check paths s-u-v-t
                # s is neighbor of u, s != v
                # t is neighbor of v, t != u, t != s
                # Constraint: deg[s] != deg[t]
                
                mask_s = mask_u & ~lsb # Neighbors of u excluding v
                mask_t = adj[v] & ~(1 << u) # Neighbors of v excluding u
                
                # We need to check if there exists s in mask_s and t in mask_t 
                # such that s != t and deg[s] == deg[t].
                
                # Optimization: Group neighbors by degree
                # But n is small, simple loop might be faster than dict overhead?
                # Let's try a set-based check for degrees of neighbors.
                
                # Actually, let's just iterate s and check if any t in mask_t has same degree and t!=s.
                # To be faster, precompute sets of neighbors for each degree?
                # No, let's just do the double loop but with bit operations if possible.
                
                # Simple approach for n=15:
                # For each s in mask_s:
                #   If there is any t in mask_t with deg[t] == deg[s] and t != s: Invalid.
                
                ms = mask_s
                while ms:
                    lsb_s = ms & -ms
                    s = lsb_s.bit_length() - 1
                    ms ^= lsb_s
                    
                    ds = deg[s]
                    
                    # Check if any t in mask_t has degree ds
                    # And t != s
                    mt = mask_t
                    # If s is in mask_t, we must exclude it.
                    # But s is neighbor of u. Is s neighbor of v?
                    # If (s,v) is edge, s is in adj[v].
                    # We want t != s.
                    
                    found = False
                    # Iterate t in mask_t
                    # We can check if (mask_t & (1<<s)) is 1 to handle t!=s?
                    # No, just check degrees.
                    
                    # Efficient check:
                    # Create a bitmask of all vertices with degree ds?
                    # Let's precompute degree_masks outside? No, deg changes.
                    
                    # For n=15, linear scan of bits is fine.
                    mt = mask_t
                    while mt:
                        lsb_t = mt & -mt
                        t = lsb_t.bit_length() - 1
                        mt ^= lsb_t
                        
                        if t == s:
                            continue
                        if deg[t] == ds:
                            return False
                            
        return True

    # Initialize best solution
    best_adj = None
    best_count = 0
    
    # Helper to convert bitmask to edge set for counting or other ops
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Strategy: Randomized Greedy + Local Search
    # Run multiple restarts
    
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
            
        # 1. Greedy Construction
        adj = [0] * n
        # Order vertices or edges?
        # Try adding edges in random order, or sorted by some heuristic.
        # Let's try random order of all pairs.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(pairs)
        
        # To make greedy better, we can sort by "potential" but random is robust.
        # Let's try a few different greedy heuristics.
        
        for i, j in pairs:
            # Check if adding (i, j) is valid
            # Update adj tentatively
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            if is_valid(adj):
                # Keep it
                pass
            else:
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
        curr_count = count_edges(adj)
        
        # 2. Local Search (Add/Remove/Swap)
        # We want to maximize edges.
        # Try adding missing edges.
        # Then try swaps (remove one, add two?) No, keep size constant or increase.
        # Since we want max edges, primarily try to add.
        # If stuck, try swap to escape local optimum.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding any missing edge
            added = False
            # Shuffle missing edges to avoid bias
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            rng.shuffle(missing)
            
            for i, j in missing:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if is_valid(adj):
                    added = True
                    improved = True
                    break
                else:
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    
            if added:
                continue
                
            # If no edge could be added, try a swap: remove one, add another?
            # Or remove one, add two (if possible)? 
            # Swapping (remove 1, add 1) keeps count same, might allow future adds.
            
            # Try random swaps
            edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & (1 << j))]
            if edge_list:
                for _ in range(100): # Limit swap attempts per local search phase
                    if time.time() > deadline - 5:
                        break
                    e1 = rng.choice(edge_list)
                    # Pick a missing edge to add
                    # Ensure e1 is not the one being added (trivially true if missing)
                    # Just pick random missing
                    missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                    if not missing_now:
                        break
                    e2 = rng.choice(missing_now)
                    
                    # Perform swap
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    adj[u1] ^= (1 << v1)
                    adj[v1] ^= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        # Accept swap, update edge_list? 
                        # It's expensive to update edge_list fully. 
                        # Just break and restart the "try adding" loop.
                        # The new graph has same count, but different structure.
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
                        
            # If we did a swap, we continue the while loop to try adding again.
            # If we didn't do a swap (or swap failed), and no add succeeded, we are stuck.
            # Break local search for this restart.
            if not improved:
                break
                
        curr_count = count_edges(adj)
        if curr_count > best_count:
            best_count = curr_count
            best_adj = adj[:]
            
        # Early stop if we found a very dense graph?
        # Max edges 105.
        if best_count > 40: # Arbitrary threshold, maybe too low
             # If we get high, maybe stop?
             # But let's keep searching for a bit longer if time permits.
             pass

    # Fallback if no valid graph found (should not happen, empty graph is valid)
    if best_adj is None:
        best_adj = [0] * n

    # Convert best_adj to list of lists
    result = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                result.append([int(i), int(j)])
            m ^= lsb
            
    return result
