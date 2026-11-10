"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def is_square(x):
        if x < 0:
            return False
        r = math.isqrt(x)
        return r * r == x

    squares = set(k * k for k in range(1, 23))
    
    # 1. Filter out diagonal conflicts (2a is square)
    valid_nodes = []
    for a in range(1, 257):
        if not is_square(2 * a):
            valid_nodes.append(a)
            
    n = len(valid_nodes)
    node_index = {node: i for i, node in enumerate(valid_nodes)}
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                adj[i].add(j)
                adj[j].add(i)

    # Helper to check if a set of indices is an independent set
    def is_independent(indices):
        ind_set = set(indices)
        for i in ind_set:
            if adj[i] & ind_set:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Heuristic 1: Greedy with multiple random restarts and local search
    # We use a time-limited loop
    
    def local_search(current_set):
        """
        current_set: set of indices
        Returns improved set
        """
        current_set = set(current_set)
        improved = True
        while improved:
            improved = False
            # Try to add a vertex
            for i in range(n):
                if i not in current_set:
                    # Check if i conflicts with any in current_set
                    if not (adj[i] & current_set):
                        current_set.add(i)
                        improved = True
                        break # Restart loop to check for more adds
            
            if improved:
                continue
                
            # Try 1-swap: remove v from S, add x not in S
            # This keeps size same, but might enable future adds
            # Actually, for maximizing size, 1-swap is only useful if it leads to a 2-add later.
            # Let's try a simpler 2-swap: remove {v1, v2}, add {x, y}
            # This increases size by 0 if 2-for-2, but we want to increase size.
            # So we look for: remove {v}, add {x, y} where x,y don't conflict with S\{v} and not with each other.
            
            # Let's try: for each v in S, remove it, then try to add as many as possible?
            # This is expensive.
            
            # Alternative: 2-Add move
            # Find x, y not in S such that:
            # 1. x, y not adjacent
            # 2. (adj[x] | adj[y]) & S is a set of size <= 1? No, we need to remove them.
            # If we remove C = (adj[x] | adj[y]) & S, and add x, y.
            # Size change = 2 - |C|.
            # We want 2 - |C| > 0 => |C| < 2 => |C| = 0 or 1.
            # If |C|=0, it's an add.
            # If |C|=1, it's a 1-for-2 swap.
            
            found_swap = False
            # Iterate over pairs of excluded nodes? Too slow (O(n^2)).
            # Instead, iterate over excluded nodes x, and for each, find y.
            excluded = [i for i in range(n) if i not in current_set]
            
            # Optimization: Only consider x with small conflict set in S
            candidates_x = []
            for x in excluded:
                conflicts = adj[x] & current_set
                if len(conflicts) <= 1:
                    candidates_x.append((x, conflicts))
            
            # Sort by conflict size
            candidates_x.sort(key=lambda t: len(t[1]))
            
            for x, conf_x in candidates_x:
                if time.time() > deadline:
                    return current_set
                # If conf_x is empty, we would have added it in the first loop.
                # So conf_x must be size 1.
                if len(conf_x) == 1:
                    v = next(iter(conf_x))
                    # We remove v, add x.
                    # Now we can try to add another y.
                    # But this is complex.
                    
                    # Let's just do the 1-for-1 swap if it allows a subsequent add?
                    # Actually, let's just do 1-for-1 swap to break local minima.
                    # If we swap v for x, size is same.
                    # Then we loop back to "Try to add".
                    # If x allows adding y which v didn't, we gain.
                    current_set.remove(v)
                    current_set.add(x)
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap and not improved:
                # Try a more aggressive move: 2-for-1
                # Remove v, add x, y
                # We need x, y such that adj[x] & (S\{v}) is empty, adj[y] & (S\{v}) is empty, and x!=y, x not adj y.
                # And v is the only conflict for both? Or v is conflict for one, and the other has no conflict?
                # If x has no conflict with S\{v}, we can add it.
                
                # Let's try a different local search: 
                # Remove a random subset of S? No.
                
                # Let's try to find x, y in excluded such that:
                # Let C = (adj[x] | adj[y]) & S
                # If len(C) == 1, say C={v}, and x,y not adjacent, then remove v, add x,y.
                
                # To make this fast, we can iterate v in S.
                # For each v, let S' = S \ {v}.
                # Find x, y in excluded such that adj[x] & S' is empty and adj[y] & S' is empty and x not adj y.
                # This is equivalent to finding an independent set of size 2 in the graph induced by excluded nodes,
                # where nodes are filtered by not conflicting with S'.
                
                # This is getting complex. Let's stick to 1-for-1 swaps and hope the greedy order is good.
                # Or, restart with different random order.
                break

        return current_set

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random order
        order = list(range(n))
        random.shuffle(order)
        
        current_set = set()
        for i in order:
            if not (adj[i] & current_set):
                current_set.add(i)
        
        # Local search
        improved_set = local_search(current_set)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            
        # If we found a very large set, we might stop early?
        # Max possible is roughly n/2?
        # Let's continue to find if we can do better.
        
        if num_restarts > 500: # Limit restarts to avoid infinite loop if time check is slow
            break

    # Convert indices back to values
    result = [valid_nodes[i] for i in best_set]
    
    # Final verification
    def is_valid(s):
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                if is_square(sl[i] + sl[j]):
                    return False
        return True

    if is_valid(result):
        return result
    else:
        # Fallback
        return list(range(1, 257, 3))

# EVOLVE-BLOCK-END
