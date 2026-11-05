import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin 10s
    
    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    best_n, best_edges = get_seed_graph()
    
    # Targets to try, descending
    # Exoo found 42. Let's try 42, 41, 40.
    targets = [42, 41, 40, 39, 38]
    
    for target_n in targets:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 5:
            break
            
        # Try to solve for target_n
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                # If we found a very good one (e.g. 42), we can stop or continue?
                # Since 42 is the known max lower bound, we can stop if we find 42.
                if n >= 42:
                    break
    return best_n, best_edges

def solve_for_n(n, time_budget):
    # Randomized local search with bitmask optimization
    # We will try multiple restarts
    max_restarts = max(1, int(time_budget / 10)) # 10s per restart
    random.seed(42 + n)
    
    # Precompute all 5-combinations
    all_combos = list(itertools.combinations(range(n), 5))
    
    for r in range(max_restarts):
        if time.time() > time_budget:
            break
            
        # Initialize random graph
        # Use numpy for fast random generation
        # We represent the graph as an adjacency matrix or list of bitmasks
        # Let's use a list of integers where adj[i] is a bitmask of neighbors of i
        
        # Random initialization
        adj = [0] * n
        # Generate random edges
        # For i < j, decide if edge exists
        # We can use a flat array of random bits
        # Number of edges = n*(n-1)/2
        num_pairs = n * (n - 1) // 2
        # 0.5 probability
        random_bits = (np.random.rand(num_pairs) < 0.5).astype(int).flatten()
        
        # Map pair index to (i, j)
        # index = i*n - (i*(i+1))//2 + (j-i-1) ?
        # Simpler: just loop and fill
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if random_bits[idx]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        # Local search loop
        while time.time() < time_budget:
            # Find one violation
            combo, type_v = find_one_violation_fast(n, adj, all_combos)
            if not combo:
                # No violation found
                # Convert adj to list of edges
                edges_list = []
                for i in range(n):
                    mask = adj[i]
                    # Only consider j > i
                    mask &= ~((1 << (i + 1)) - 1) # Clear bits <= i
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        edges_list.append((i, j))
                        mask ^= lsb
                return n, edges_list
            
            # Try to fix by flipping an edge in the combo
            # Get all pairs in combo
            pairs = list(itertools.combinations(combo, 2))
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Flip edge (u, v)
                # Current state:
                has_edge = (adj[u] >> v) & 1
                
                if has_edge:
                    # Remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Accept the flip immediately to escape local minima?
                # Or check if it reduces violations?
                # Checking reduction is expensive.
                # Let's try a "simulated annealing" like approach or just greedy.
                # Greedy: if the specific violation is fixed, accept.
                # Does the flip fix the specific combo?
                # If type_v was 'K5', and we removed an edge, it's fixed.
                # If type_v was 'IK5', and we added an edge, it's fixed.
                # But we might have created a new violation.
                # The next iteration will find it.
                
                # Heuristic: If we just flip, we might oscillate.
                # Let's check if the count of violations decreased?
                # Too slow.
                # Let's just accept the first flip that breaks the specific combo.
                # This is essentially what the original code did, but faster.
                break
            
    return None

def find_one_violation_fast(n, adj, all_combos):
    # Iterate through combinations
    # Optimization: We can check K5 and IK5 simultaneously
    # For a combo (a,b,c,d,e):
    # K5: adj[a]&b, adj[a]&c, ... all 1
    # IK5: ~adj[a]&b, ... all 1 (where ~ is within n bits)
    
    # To speed up, we can precompute nothing, but use bit operations.
    
    for combo in all_combos:
        a, b, c, d, e = combo
        
        # Check K5
        # Edges: (a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)
        
        # Using bitmasks
        # ab = (adj[a] >> b) & 1
        # ...
        
        # Faster:
        # Let's check if adj[a] has b,c,d,e
        # mask_abcde = (1<<b) | (1<<c) | (1<<d) | (1<<e)
        # if (adj[a] & mask_abcde) == mask_abcde:
        #    then a is connected to all.
        #    Now check if b,c,d,e form K4.
        
        mask_b_c_d_e = (1 << b) | (1 << c) | (1 << d) | (1 << e)
        
        if (adj[a] & mask_b_c_d_e) == mask_b_c_d_e:
            # a is connected to all others.
            # Check if b,c,d,e form K4
            mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
            if (adj[b] & mask_c_d_e) == mask_c_d_e:
                mask_d_e = (1 << d) | (1 << e)
                if (adj[c] & mask_d_e) == mask_d_e:
                    if (adj[d] >> e) & 1:
                        return combo, 'K5'
        
        # Check IK5
        # Non-edges:
        # ~adj[a] & mask_b_c_d_e == mask_b_c_d_e
        # Note: adj[a] doesn't have bit a set.
        # We need to ensure we are checking non-edges.
        
        if (adj[a] & mask_b_c_d_e) == 0:
            # a is not connected to any of b,c,d,e
            if (adj[b] & mask_c_d_e) == 0:
                if (adj[c] & mask_d_e) == 0:
                    if (adj[d] & (1 << e)) == 0:
                        return combo, 'IK5'
                        
    return None, None
