"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import sys
import time
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    Approach:
    Model the problem as finding a Maximum Independent Set (MIS) in a conflict graph.
    Nodes are integers 1..256.
    An edge exists between u and v if the pair {u, v} is invalid (i.e., subset sums of {u, v} have a divisibility conflict).
    An Independent Set in this graph is a set where every pair is valid.
    This is a necessary condition for the whole set to be valid.
    We use a Branch and Bound algorithm with bitsets and greedy coloring to find large independent sets.
    We then verify the global validity of the top candidates.
    """
    
    n = 256
    # Bitset representation: use Python integers
    # adj[i] is a bitmask where bit j is 1 if there is a conflict between i and j
    adj = [0] * (n + 1)
    
    # Precompute pair validity
    # A pair {u, v} is valid if:
    # - u, v, u+v are distinct (u!=v, u>0, v>0 implies distinct sums? u+v != u, u+v != v yes. u!=v yes.)
    # - No sum divides another. Sums: u, v, u+v.
    #   Check: u | v? u | (u+v)? v | u? v | (u+v)? (u+v) | u? (u+v) | v?
    #   Since u, v > 0, u+v > u and u+v > v. So (u+v) cannot divide u or v.
    #   We only need to check: u | v, v | u, u | (u+v), v | (u+v).
    #   u | (u+v) is equivalent to u | v.
    #   v | (u+v) is equivalent to v | u.
    #   So the condition for a pair {u, v} to be valid is:
    #   u does not divide v AND v does not divide u.
    
    # Wait, is that it?
    # Subset sums of {u, v}: u, v, u+v.
    # Divisibility checks:
    # 1. u | v
    # 2. u | (u+v)  <=> u | v
    # 3. v | u
    # 4. v | (u+v)  <=> v | u
    # 5. (u+v) | u (Impossible for u,v > 0)
    # 6. (u+v) | v (Impossible for u,v > 0)
    
    # So, a pair {u, v} is valid if and only if u % v != 0 and v % u != 0.
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if u % v == 0 or v % u == 0:
                # Conflict
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            # else: no edge (independent in conflict graph)
            
    # Function to check if a set is globally valid
    def is_globally_valid(S):
        if not S:
            return True
        S = list(S)
        n_elem = len(S)
        subset_sums = set()
        for mask in range(1, 1 << n_elem):
            s = 0
            for i in range(n_elem):
                if mask & (1 << i):
                    s += S[i]
            if s in subset_sums:
                return False # Duplicate sum (should not happen if distinct elements? No, 1+3=2+2? No, distinct elements. But 1+4=2+3? Yes. So distinctness check is needed.)
            subset_sums.add(s)
            
        sums_list = sorted(subset_sums)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if sums_list[j] % sums_list[i] == 0:
                    return False
        return True

    # Branch and Bound for Maximum Independent Set
    # Using bitsets
    # Nodes are 1..n. We use 0..n-1 for bit positions to make it easier?
    # Let's use 0-based indexing for bitsets to save 1 bit.
    # Bit i corresponds to node i+1.
    
    adj_bits = [0] * n
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            if u != v and (adj[u] & (1 << v)):
                adj_bits[u-1] |= (1 << (v-1))
                
    # Greedy coloring bound for independent set
    # We want to color the graph such that adjacent nodes have different colors.
    # The number of colors is an upper bound on the size of the independent set?
    # No, for Clique it is. For Independent Set, we look for the maximum set of non-adjacent nodes.
    # This is equivalent to finding a Clique in the complement graph.
    # Let's work on the complement graph for the B&B?
    # Or just use the standard MIS B&B.
    
    # Standard approach:
    # State: (candidates, current_set_size)
    # Upper bound: current_size + greedy_color_bound(candidates)
    
    def greedy_color_bound(candidates_mask):
        """
        Estimate the maximum independent set size in the subgraph induced by candidates_mask.
        Uses a greedy coloring heuristic on the conflict graph restricted to candidates.
        Actually, for MIS, we can use a greedy packing bound.
        A simpler bound: The size of the maximum independent set in a graph with m nodes is at most m.
        Better bound: Greedy coloring of the *complement* graph gives a clique bound, which is an independent set bound in the original?
        No.
        Let's use the Tomita algorithm for Maximum Clique on the complement graph.
        Complement graph edge (u,v) exists if u and v are NOT adjacent in conflict graph (i.e., they are compatible).
        So we are looking for a Maximum Clique in the compatibility graph.
        """
        # Extract nodes
        nodes = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            m ^= lsb
            
        if not nodes:
            return 0
            
        # Build adjacency for these nodes in the COMPATIBILITY graph
        # Compat(u, v) is True if NOT (adj_bits[u] & (1 << v))
        
        # Greedy coloring for Clique:
        # Order nodes, assign colors.
        # The number of colors is an upper bound on the clique size.
        
        # Simple greedy:
        # Sort nodes by degree in compatibility graph? Or just iterate.
        # Let's just use a simple bound: len(nodes) is too loose.
        # A better simple bound:
        # Greedily select nodes that are compatible with all previously selected? No, that finds one IS.
        
        # Let's implement a simple greedy coloring for the Clique problem (on compat graph).
        # We want to color the compat graph. Min colors = Max Clique? No, Max Clique <= Max Color.
        # So if we color the compat graph with C colors, the max clique is <= C.
        
        # To color:
        # colors = {}
        # for node in nodes:
        #    find smallest color not used by neighbors in compat graph
        
        # This is O(V^2). Acceptable for small V (up to 256, but subproblems are smaller).
        
        compat_adj = {}
        for u in nodes:
            compat_adj[u] = 0
            for v in nodes:
                if u != v and not (adj_bits[u] & (1 << v)):
                    compat_adj[u] |= (1 << v)
                    
        # Greedy coloring
        color_of = {}
        next_color = 0
        # Sort by degree descending for better coloring?
        nodes_sorted = sorted(nodes, key=lambda x: bin(compat_adj[x]).count('1'), reverse=True)
        
        for u in nodes_sorted:
            # Find smallest available color
            used_colors = set()
            mask_u = compat_adj[u]
            for v in nodes_sorted:
                if v != u and (mask_u & (1 << v)):
                    if v in color_of:
                        used_colors.add(color_of[v])
            c = 0
            while c in used_colors:
                c += 1
            color_of[u] = c
            next_color = max(next_color, c + 1)
            
        return next_color

    def branch_and_bound(candidates_mask, current_size, current_set, best_set, best_size, timeout=50):
        start_time = time.time()
        if time.time() - start_time > timeout:
            return
        
        if candidates_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = list(current_set)
            return
            
        # Bound
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size[0]:
            return
            
        # Pick a node
        # Heuristic: pick node with highest degree in compat graph?
        # Or just pick the first one.
        lsb = candidates_mask & -candidates_mask
        idx = lsb.bit_length() - 1
        node_val = idx + 1
        
        # Branch 1: Include node
        # New candidates: neighbors of idx in COMPAT graph (nodes compatible with idx)
        new_cands = candidates_mask & ~lsb
        # Filter new_cands to only those compatible with idx
        compat_mask = 0
        m = new_cands
        while m:
            l = m & -m
            i = l.bit_length() - 1
            if not (adj_bits[idx] & l): # Compatible
                compat_mask |= l
            m ^= l
            
        current_set.append(node_val)
        branch_and_bound(compat_mask, current_size + 1, current_set, best_set, best_size, timeout)
        current_set.pop()
        
        # Branch 2: Exclude node
        branch_and_bound(new_cands, current_size, current_set, best_set, best_size, timeout)

    # Initialize B&B
    # We want to find the MIS of the conflict graph.
    # Start with all nodes.
    initial_mask = (1 << n) - 1
    
    best_set = [[]]
    best_size = [0]
    
    # To speed up, we can order nodes.
    # Let's run B&B
    # Note: B&B on 256 nodes might be slow if not pruned well.
    # The coloring bound is O(V^2).
    
    # We might need to limit the search or use a better heuristic.
    # Let's try it.
    
    try:
        branch_and_bound(initial_mask, 0, [], best_set, best_size, timeout=45)
    except Exception:
        pass
        
    # The B&B finds the maximum *pairwise* valid set.
    # We need to check if it is *globally* valid.
    candidate = best_set[0]
    
    if is_globally_valid(candidate):
        return [int(x) for x in candidate]
        
    # If the best pairwise set is not globally valid, we need to find a valid one.
    # We can try to modify it or search for other large independent sets.
    # For simplicity and robustness, if the top one fails, we might return a smaller known valid set
    # or try to prune the conflicting elements.
    
    # Let's try to remove elements that cause global conflicts.
    # This is complex.
    # Alternative: The B&B might have found a set that is pairwise valid but globally invalid.
    # Given the constraints, it's likely that the maximum pairwise valid set is also globally valid for this problem structure (superincreasing-like).
    # If not, we fallback to a smaller set.
    
    # Let's try to find a valid subset of the candidate.
    # Greedy removal:
    valid_candidate = []
    for x in candidate:
        temp = valid_candidate + [x]
        if is_globally_valid(temp):
            valid_candidate = temp
        else:
            # Try to see if we can skip x
            pass
            
    if is_globally_valid(valid_candidate):
        return [int(x) for x in valid_candidate]
        
    # Fallback to a simple known valid set if all else fails
    # {31, 32, 34} is size 3.
    # Let's try to generate a superincreasing set:
    # 1, 3, 7, 15...
    # Check 1, 3: sums 1,3,4. 1|3? No. 1|4? No. 3|1? No. 3|4? No. 4|1? No. 4|3? No. Valid.
    # 1, 3, 7: sums 1,3,4,7,8,10,11.
    # 1|3? No. 1|4? No. 1|7? No... 1 divides everything!
    # So 1 cannot be in the set if any other number is present?
    # If 1 is in the set, sum=1 is a subset sum. 1 divides every other sum.
    # So 1 MUST be excluded.
    
    # Start from 2?
    # 2, 5: sums 2,5,7. 2|5? No. 2|7? No. 5|2? No. 5|7? No. 7|2? No. 7|5? No. Valid.
    # 2, 5, 11: sums 2,5,7,11,13,16,18.
    # 2|5? No. 2|7? No. 2|11? No. 2|13? No. 2|16? YES. Invalid.
    
    # Let's try 3, 5, 17?
    # 3, 5: sums 3,5,8. Valid.
    # 3, 5, 17: sums 3,5,8,17,20,22,25.
    # 5|20? Yes. Invalid.
    
    # Let's try 3, 7, 17?
    # 3, 7: sums 3,7,10. 3|7? No. 3|10? No. 7|3? No. 7|10? No. 10|3? No. 10|7? No. Valid.
    # 3, 7, 17: sums 3,7,10,17,20,24,27.
    # 3|7? No. 3|10? No. 3|17? No. 3|20? No. 3|24? YES. Invalid.
    
    # It seems hard to get large sets.
    # The B&B approach should find the largest pairwise valid set.
    # If that set is invalid globally, the greedy removal might yield a smaller valid set.
    
    # If valid_candidate is too small, we can try a random search or just return it.
    
    return [int(x) for x in valid_candidate]

# EVOLVE-BLOCK-END
