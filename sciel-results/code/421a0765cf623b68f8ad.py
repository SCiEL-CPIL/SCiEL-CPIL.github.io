import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    
    # Precompute bad triples masks
    # bad_with[u][v] is a bitmask of all x such that {u, v, x} is a bad triple.
    # Note: u, v, x are distinct.
    bad_with = [[0] * (N + 1) for _ in range(N + 1)]
    
    # First, identify all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    # We iterate a < b < c
    # For each bad triple, update bad_with for all pairs.
    
    # To make it faster, we can just iterate all triples and check.
    # N=128, N^3/6 ~ 350k iterations. This is fast.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                g_bc = gcd(b, c)
                if g_ab == g_bc:
                    g_ca = gcd(c, a)
                    if g_ab == g_ca:
                        # Triple (a, b, c) is bad
                        # Update masks
                        # Pair (a, b) conflicts with c
                        bad_with[a][b] |= (1 << c)
                        bad_with[b][a] |= (1 << c)
                        # Pair (a, c) conflicts with b
                        bad_with[a][c] |= (1 << b)
                        bad_with[c][a] |= (1 << b)
                        # Pair (b, c) conflicts with a
                        bad_with[b][c] |= (1 << a)
                        bad_with[c][b] |= (1 << a)

    best_size = 0
    best_set = []
    
    # Bitmask for all vertices 1..N
    # Bit i represents vertex i.
    # We use bits 1..N. Bit 0 is unused or used for 0 (not in range).
    all_mask = 0
    for i in range(1, N + 1):
        all_mask |= (1 << i)
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # Helper to get list of set bits
    def bits_to_list(mask):
        res = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            # Index of lsb
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res

    # B&B function
    # selected: list of chosen vertices
    # available_mask: bitmask of vertices that can still be chosen
    def branch_and_bound(selected, available_mask):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound
        if len(selected) + popcount(available_mask) <= best_size:
            return
            
        if available_mask == 0:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
            return
            
        # Choose a vertex to branch on
        # Heuristic: Pick a vertex that is in many conflicts? 
        # Or just pick the first one?
        # Let's pick a vertex with high degree in the conflict graph induced by available?
        # Simpler: Just pick the lowest index available to keep it deterministic and fast?
        # Or pick random?
        # Let's try picking the vertex that appears in the most bad triples with other available vertices?
        # That might be expensive to compute each step.
        # Let's just iterate through available vertices and pick one.
        # To improve pruning, we should branch on vertices that "block" the most other vertices.
        # Let's calculate a score for each available vertex v:
        # score(v) = popcount(bad_with[v][u] & available_mask) for some u? No.
        # If we pick v, we remove from available all x such that there exists u in selected with {u,v,x} bad.
        # This depends on selected.
        
        # Let's just pick the first available vertex for simplicity, or use a greedy heuristic for ordering.
        # Actually, a common heuristic is to pick the vertex with the highest "degree" in the current conflict structure.
        # But let's start with a simple selection: lowest index.
        
        # Get list of available vertices
        avail_list = bits_to_list(available_mask)
        
        # Optional: Reorder avail_list to try "critical" vertices first
        # A vertex v is critical if adding it removes many options?
        # Let's just iterate.
        
        for v in avail_list:
            # Option 1: Include v
            # Calculate forbidden mask
            forbidden = 0
            for u in selected:
                forbidden |= bad_with[u][v]
                
            new_available = available_mask & ~(1 << v) & ~forbidden
            
            # Recurse
            selected.append(v)
            branch_and_bound(selected, new_available)
            selected.pop()
            
            if time.time() > deadline:
                return
                
            # Option 2: Exclude v
            new_available_ex = available_mask & ~(1 << v)
            branch_and_bound(selected, new_available_ex)
            
            if time.time() > deadline:
                return

    # Initial call
    # We can initialize with a greedy solution to set a good lower bound (best_size)
    # Greedy: Pick vertices one by one, maintaining validity.
    # This helps pruning significantly.
    
    # Simple Greedy Initialization
    curr_selected = []
    curr_available = all_mask
    while curr_available:
        # Pick a vertex, say lowest
        v_list = bits_to_list(curr_available)
        if not v_list: break
        v = v_list[0]
        # Check if we can add v? 
        # In this greedy, we just add it and update available.
        # But we must ensure no bad triple is formed.
        # Since we update available correctly, any v in curr_available is safe to add 
        # IF we assume that "available" means "can be added without violating constraints with pairs in selected".
        # Yes, that's the invariant.
        
        curr_selected.append(v)
        forbidden = 0
        for u in curr_selected[:-1]: # Exclude v itself from loop if we were careful, but v is not in selected yet in this loop logic? 
            # Wait, curr_selected now has v. 
            # We need to check pairs (u, v) for u in curr_selected without v.
            pass
        
        # Let's redo the greedy step carefully
        # Actually, the B&B function handles the logic. 
        # For initialization, let's just run a simple greedy that adds vertices if they don't conflict with any pair in selected.
        
        # Reset for proper greedy
        curr_selected = []
        curr_available = all_mask
        while curr_available:
            # Pick vertex with max potential? Or just first?
            # Let's pick the one that conflicts least?
            # To be fast, just pick first.
            v_list = bits_to_list(curr_available)
            if not v_list: break
            v = v_list[0]
            
            # Check if v can be added to curr_selected
            # v is in curr_available, which implies no pair in curr_selected conflicts with v.
            # So it is safe.
            curr_selected.append(v)
            
            # Update available
            forbidden = 0
            for u in curr_selected[:-1]:
                forbidden |= bad_with[u][v]
            curr_available = curr_available & ~(1 << v) & ~forbidden
            
    best_size = len(curr_selected)
    best_set = curr_selected[:]
    
    # Now run B&B to improve
    # We can start B&B with empty set or with the greedy set?
    # If we start with empty set, B&B will explore.
    # If we start with greedy set, we can just branch on the remaining available vertices.
    # Let's start with the greedy set to save time.
    
    # Calculate available mask for the greedy set
    # This is already `curr_available` from the greedy loop.
    # But we need to be sure `curr_available` is correct.
    # In the greedy loop, we updated it correctly.
    
    # However, B&B needs to explore excluding some vertices from the greedy set too?
    # No, if we start with `curr_selected`, we are committed to them.
    # To allow removing them, we should start from scratch (empty selected) but with a good bound.
    # Or, we can run B&B from the state of the greedy solution, but that only finds superset extensions.
    # To find a better set of same size or different composition, we should start from empty.
    # But the bound `len(selected) + popcount(available)` will prune branches that can't beat `best_size`.
    # So starting from empty is fine, and the good `best_size` will prune heavily.
    
    branch_and_bound([], all_mask)
    
    if not best_set:
        # Fallback
        return [1]
        
    return best_set
