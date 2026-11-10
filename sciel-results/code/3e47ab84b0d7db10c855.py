import numpy as np
import time
import sys

# Increase recursion limit just in case, though depth should be small
sys.setrecursionlimit(10000)

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build conflict graph as bitmasks
    # conflict[i] is a bitmask where bit j is 1 if (i-j)%m is in forbidden
    conflict = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict[i] = mask
        
    # 3. Initial greedy lower bound to help pruning
    # Try a few random orderings to get a good initial independent set
    best_set = []
    best_size = 0
    
    def greedy_random():
        nodes = list(range(m))
        np.random.shuffle(nodes)
        current_set = []
        for node in nodes:
            valid = True
            for s in current_set:
                if (node - s) % m in forbidden:
                    valid = False
                    break
            if valid:
                current_set.append(node)
        return current_set

    # Run a few greedy attempts to establish a baseline
    for _ in range(100):
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 4. Branch and Bound
    # We store the current path of included vertices
    current_path = []
    
    # Use a non-local variable for time check
    start_time = time.time()
    time_limit = 550 # seconds
    
    def popcount(x):
        return bin(x).count("1")
        
    def branch_and_bound(available_mask, current_size, depth):
        nonlocal best_size, best_set, start_time, time_limit
        
        # Time check
        if time.time() - start_time > time_limit:
            return False # Signal timeout
            
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_path)
            
        if available_mask == 0:
            return True
            
        # Upper bound: current_size + number of vertices in available_mask
        # This is a weak bound. A stronger bound is the size of a maximal matching or clique cover.
        # For now, let's try a slightly stronger bound: 
        # Greedily find a maximal independent set in the subgraph? No, that's expensive.
        # Let's stick to popcount for now, but prune if current_size + popcount(available_mask) <= best_size
        if current_size + popcount(available_mask) <= best_size:
            return True
            
        # Select vertex v from available_mask
        # Heuristic: Pick vertex with minimum degree in the induced subgraph
        best_v = -1
        min_deg = m + 1
        temp_mask = available_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            # Degree of v in available subgraph
            deg = popcount(conflict[v] & available_mask)
            if deg < min_deg:
                min_deg = deg
                best_v = v
            if deg == 0:
                break
                
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New available: remove v and all its neighbors
        new_available = available_mask & ~conflict[v] & ~v_bit
        current_path.append(v)
        
        if not branch_and_bound(new_available, current_size + 1, depth + 1):
            return False # Timeout
        
        current_path.pop()
        
        # Branch 2: Exclude v
        new_available = available_mask & ~v_bit
        if not branch_and_bound(new_available, current_size, depth + 1):
            return False # Timeout
            
        return True
        
    # Start B&B with all vertices available
    initial_available = (1 << m) - 1
    branch_and_bound(initial_available, 0, 0)
    
    # If best_set is empty (shouldn't happen), fallback to greedy
    if not best_set:
        best_set = greedy_random()
        
    return np.array(best_set, dtype=np.int64)
