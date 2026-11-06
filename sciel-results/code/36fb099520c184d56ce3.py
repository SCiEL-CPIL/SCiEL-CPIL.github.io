import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n = 20
    total_possible_edges = n * (n - 1) // 2
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
    
    # Bitmask adjacency: adj[i] is a bitmask of neighbors of i
    # best_adj tracks the best solution found
    best_adj = [0] * n
    best_count = 0
    
    # Helper to count edges in bitmask representation
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
        
    # Helper to check if adding edge (u, v) is valid
    # Returns True if (u, v) can be added without creating K2,3
    def is_valid_add(adj, u, v):
        # 1. Check if (u, v) itself has > 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. Check if adding (u, v) creates a K2,3 involving u or v
        # Adding (u, v) increases common neighbors for:
        # - Pair (u, w) where w in N(v): common(u, w) increases by 1 (v becomes common)
        #   Actually, v is in N(w) iff w is in N(v). 
        #   Wait, common(u, w) = |N(u) \cap N(w)|.
        #   New N(u) = N(u) U {v}.
        #   New common(u, w) = | (N(u) U {v}) \cap N(w) | = |N(u) \cap N(w)| + (1 if v in N(w) else 0).
        #   So for all w in N(v), if old_common(u, w) == 2, adding (u, v) makes it 3 -> Invalid.
        
        # - Pair (v, w) where w in N(u): similar logic.
        
        # Efficient check:
        # Iterate neighbors of v
        mask_v = adj[v]
        # We need to check all w such that w is in N(v) and w != u
        # For each such w, if (adj[u] & adj[w]).bit_count() == 2, then adding (u,v) 
        # makes common(u,w) = 3.
        
        # Also, we must check if (u, v) edge itself is the problem? 
        # No, that was check 1.
        
        # Let's optimize the loop.
        # Candidates for w in N(v):
        candidates_v = mask_v & ~(1 << u) # Exclude u
        # For each w in candidates_v:
        #   if (adj[u] & adj[w]).bit_count() >= 2: return False
        
        # Candidates for w in N(u):
        candidates_u = adj[u] & ~(1 << v)
        # For each w in candidates_u:
        #   if (adj[v] & adj[w]).bit_count() >= 2: return False
        
        # To speed this up, we can precompute or just iterate. n=20 is small.
        
        # Check neighbors of v
        w = 0
        temp_mask = candidates_v
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            if (adj[u] & adj[w]).bit_count() >= 2:
                return False
            temp_mask ^= lsb
            
        # Check neighbors of u
        temp_mask = candidates_u
        while temp_mask:
            lsb = temp_mask & -temp_mask
            w = lsb.bit_length() - 1
            if (adj[v] & adj[w]).bit_count() >= 2:
                return False
            temp_mask ^= lsb
            
        return True

    # Helper to remove edge (u, v)
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Helper to add edge (u, v)
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Initial solution: Greedy random construction
    def greedy_init():
        adj = [0] * n
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        for u, v in edges_list:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Run Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    
    # We will run multiple SA restarts if time permits
    # But first, let's tune one long SA run or multiple short ones.
    # Given n=20, the state space is large but manageable.
    
    best_adj = greedy_init()
    best_count = count_edges(best_adj)
    
    # To improve speed, we can precompute a list of missing edges?
    # Or just pick random u, v and check if edge exists.
    
    # Let's do a single long SA run with aggressive restarts if stuck
    # Or just run SA until deadline.
    
    current_adj = best_adj[:]
    current_count = best_count
    T = T_start
    
    # Track time
    # We need to be very fast in the loop.
    
    # Precompute list of all edges for random selection of existing/missing
    # Actually, picking random u, v is O(1).
    
    # To speed up is_valid_add, we can cache? No, adj changes.
    
    # Let's run SA
    # Number of iterations is limited by time.
    # Estimate: is_valid_add is O(n) bit ops. ~20 ops.
    # Python loop overhead is high.
    
    # Optimization: Use a list of currently present edges to pick removals from.
    # And a list of missing edges to pick additions from.
    # Updating these lists is O(1) if we use sets or lists with swap-delete.
    
    present_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (current_adj[i] >> j) & 1:
                present_edges.add((i, j))
                
    missing_edges = set(all_edges) - present_edges
    
    # Convert to lists for faster random access? 
    # Sets are fine for membership, but random.choice on list is faster.
    # Let's use lists and maintain them.
    
    # Re-initialize with lists for speed
    present_list = list(present_edges)
    missing_list = list(missing_edges)
    
    # Helper to update lists
    def add_to_present(u, v):
        e = (u, v) if u < v else (v, u)
        present_list.append(e)
        # Remove from missing
        # This is O(n) in list. Might be slow.
        # Alternative: Just pick random u,v and check if present.
        # If present, try remove. If missing, try add.
        # This avoids list maintenance overhead.
        
    # Let's stick to random u, v selection.
    # If (u,v) is in adj, try remove. Else try add.
    
    # To make "try add" faster, we need to know if it's missing.
    # (1 << v) & adj[u]
    
    # Let's refine the loop to be as tight as possible.
    
    while time.time() < deadline - 1:
        # Pick random move
        # 1. Add edge
        # 2. Remove edge
        # 3. Swap (Remove one, Add one) - complex, maybe skip for now
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Ensure u < v for consistency? Not strictly needed for bitmask.
        
        # Check if edge exists
        if (adj_mask := (1 << v) & current_adj[u]):
            # Edge exists. Try to remove.
            # Removal is always valid.
            # Acceptance: always accept? Or probabilistic?
            # Removing reduces count. We want to maximize count.
            # In SA, we accept worse moves with prob exp(-delta/T).
            # Delta = new_count - old_count = -1.
            # Prob = exp(1/T). If T is small, this is ~0.
            # So we rarely remove unless T is high or we are stuck.
            
            # Actually, if we just remove, we might drift to 0 edges.
            # Better: Only remove if we can immediately add a different edge?
            # Or use "Swap": Pick u, v (present), pick x, y (missing).
            # Try to remove (u,v) and add (x,y).
            
            # Let's try a "Flip" move:
            # Pick random u, v.
            # If edge exists:
            #    50% chance: Keep it.
            #    50% chance: Try to remove it. (Accept with prob exp(1/T))
            # If edge missing:
            #    Try to add it. (Accept if valid. Always improves count, so accept if valid.)
            
            if random.random() < 0.5:
                # Try remove
                # Calculate acceptance prob
                # delta = -1
                # prob = exp(1/T)
                # If T < 1, exp(1/T) > 1? No, exp(positive) > 1.
                # Wait, standard SA: accept worse with prob exp(-delta/T).
                # delta = new - old = -1.
                # exp(-(-1)/T) = exp(1/T).
                # If T=1, prob = e ~ 2.7. Cap at 1.
                # If T=0.1, prob = exp(10) ~ huge. Cap at 1.
                # So for T > ~0.7, we always accept removals?
                # This means at high T, we random walk. At low T, we rarely remove.
                # This is correct behavior for maximization.
                
                if random.random() < min(1.0, 2.71828 ** (1.0 / T)):
                    # Remove
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_count -= 1
            else:
                # Try add (if it was present, we skipped add? 
                # My logic above: if present, 50% remove, 50% ... do nothing?
                # That wastes time.
                # Better:
                # If present: Try remove.
                # If missing: Try add.
                
                # Let's restructure:
                pass # Handled below
        
        else:
            # Edge missing. Try to add.
            if is_valid_add(current_adj, u, v):
                # Valid add. Always accept (improves objective).
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
            
        # Cooling
        T *= alpha
        
        # If T is too low, restart?
        # If T < T_end:
        #    T = T_start * 0.5
        #    # Optional: randomize current_adj slightly?
        
    # Convert best_adj to edge list
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
                
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    # Validate
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
