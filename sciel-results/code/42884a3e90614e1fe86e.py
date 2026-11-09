import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer
    
    N = 256
    values = list(range(1, N + 1))
    
    # 1. Identify valid nodes (2*v is not a perfect square)
    def is_square(n):
        r = int(math.isqrt(n))
        return r * r == n
        
    valid_values = []
    for v in values:
        if not is_square(2 * v):
            valid_values.append(v)
            
    M = len(valid_values)
    val_to_idx = {v: i for i, v in enumerate(valid_values)}
    
    # 2. Build conflict matrix / neighbor masks
    # neighbor_mask[i] is a bitmask where bit j is set if valid_values[i] + valid_values[j] is a square
    # We use a list of lists for O(1) conflict check, and bitmasks for fast set operations if needed, 
    # but for local search, adjacency lists or boolean matrix are often faster in Python loops.
    # Let's use a list of sets for neighbors for easy "is_conflict" check.
    
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    neighbors = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_values[i]
        for j in range(i + 1, M):
            vj = valid_values[j]
            if vi + vj in squares:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Helper to check if a node can be added to a set
    # current_set: set of indices
    def can_add(node, current_set):
        for n in neighbors[node]:
            if n in current_set:
                return False
        return True

    # 1. Greedy Initialization
    def greedy_init(order):
        current_set = set()
        for idx in order:
            if can_add(idx, current_set):
                current_set.add(idx)
        return current_set

    # 2. Local Search (Hill Climbing with Random Restarts)
    # Strategy: 
    # - Start with a greedy set.
    # - Try to improve by:
    #   a) Adding a node from 'available' (nodes not in set and not conflicting with set? No, 'available' means not in set). 
    #      Actually, if a node is not in the set, it might conflict. 
    #      If we remove a node from the set, we might be able to add multiple nodes.
    #   b) Simple 1-1 Swap: Remove u from set, Add v to set. Size unchanged? No, we want to increase size.
    #   c) 1-k Swap: Remove u, add k nodes. This is complex.
    #   
    # Alternative Local Search Strategy: 
    # Maintain `in_set` and `out_set`.
    # Move: Pick a node `u` in `in_set` to remove. 
    #       Find all nodes in `out_set` that become "free" (no longer conflict with `in_set` \ {u}).
    #       If count(free) > 1, we can potentially increase size.
    #       Specifically, if we remove u, we gain freedom for its neighbors in `out_set`.
    #       Let's try a simpler heuristic: 
    #       1. Try to add any node from `out_set` that doesn't conflict with `in_set`. (This is only possible if the greedy init didn't pick it due to order, but greedy picks if possible. So this only works if we shuffled the order differently in a restart).
    #       2. If stuck, perform a "kick": Remove a random node from `in_set`, then greedily re-add as many as possible from `out_set`.
    
    def local_search(initial_set, max_iterations=5000):
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        
        # Precompute complement neighbors? 
        # For each node, who does it NOT conflict with?
        # non_neighbors[i] = all nodes j != i such that i not in neighbors[j]
        
        # Let's use a "Removal-Addition" heuristic.
        # If we remove u, we can add any v in out_set such that v does not conflict with (current_set - {u}).
        # This means v must not be a neighbor of any node in current_set \ {u}.
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Check if we can just add a node (shouldn't happen if greedy was perfect, but maybe after swaps)
            # Actually, let's just do the swap logic.
            
            # Pick a random node to remove from current_set
            if not current_set:
                break
            u = random.choice(list(current_set))
            current_set.remove(u)
            
            # Now try to add as many as possible from the remaining nodes
            # To be efficient, we only look at nodes that might have been blocked by u.
            # But simply re-greedy-ing on the "available" nodes (out_set) is safer.
            
            # Let's gather candidates: nodes in out_set that are NOT neighbors of current_set.
            # Since we just removed u, some nodes that were blocked by u might now be free.
            # Also, other nodes in out_set might have been free all along but not picked due to order?
            # In a valid independent set, no two nodes in the set conflict.
            # Any node in out_set that does not conflict with ANY node in current_set can be added.
            
            added = 0
            # To make it faster, we can just iterate through out_set.
            # out_set = set(range(M)) - current_set
            # This is O(M). M=233. 5000 iterations * 233 = 1.1M ops. Fast enough.
            
            out_set = set(range(M)) - current_set
            # We want to add nodes in a deterministic or random order?
            # Random order helps escape local minima.
            candidates = list(out_set)
            random.shuffle(candidates)
            
            for v in candidates:
                if can_add(v, current_set):
                    current_set.add(v)
                    added += 1
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
            # If we added 0 nodes, we are stuck in a local minimum for this u.
            # We can try another u immediately (next iteration).
            # To avoid oscillation, we could track visited states, but with M=233, state space is huge.
            # Random restarts are better.
            
        return best_set, best_size

    best_global_set = set()
    best_global_size = 0
    
    # Run multiple restarts with different random orders
    # 1. Degree-based order (high degree first? low degree first?)
    # Low degree first is often better for Independent Set (greedy picks nodes with fewest constraints).
    degrees = [len(neighbors[i]) for i in range(M)]
    
    # Strategy 1: Greedy with Low Degree First
    order_ld = sorted(range(M), key=lambda x: degrees[x])
    s1, sz1 = local_search(greedy_init(order_ld), 2000)
    if sz1 > best_global_size:
        best_global_size = sz1
        best_global_set = s1
        
    # Strategy 2: Greedy with High Degree First
    order_hd = sorted(range(M), key=lambda x: -degrees[x])
    s2, sz2 = local_search(greedy_init(order_hd), 2000)
    if sz2 > best_global_size:
        best_global_size = sz2
        best_global_set = s2
        
    # Strategy 3: Random Restarts
    while time.time() < deadline - 5.0:
        order_r = list(range(M))
        random.shuffle(order_r)
        s_r, sz_r = local_search(greedy_init(order_r), 1000)
        if sz_r > best_global_size:
            best_global_size = sz_r
            best_global_set = s_r
            
    # Decode best set to values
    best_nodes = [valid_values[i] for i in best_global_set]
    
    if not best_nodes:
        # Fallback
        return [valid_values[0]] if valid_values else [1]
        
    return best_nodes
