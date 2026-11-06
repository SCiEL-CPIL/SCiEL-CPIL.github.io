"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    N = 256
    # 1. Build Conflict Graph
    # adj[i] is a bitmask of vertices j such that i and j conflict.
    # Vertices are 0..N-1 representing numbers 1..N.
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                # i and j conflict
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Maximum Independent Set using Branch and Bound with Bitsets
    # We want to find the largest subset of vertices with no edges between them.
    
    # Precompute popcount for speed? Python 3.8+ has int.bit_count()
    
    best_set_mask = 0
    best_size = 0
    
    # To avoid deep recursion issues, we might implement an iterative version or just increase limit.
    # For N=256, depth is 256. Default limit is 1000. Should be fine.
    
    def solve(remaining_mask, current_mask, current_size):
        nonlocal best_set_mask, best_size
        
        # Pruning: if current_size + number of available nodes <= best_size, stop
        if current_size + remaining_mask.bit_count() <= best_size:
            return
            
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Heuristic: Pick a vertex from remaining.
        # A common heuristic is to pick the vertex with the highest degree in the remaining subgraph.
        # Or simply the lowest index.
        # Let's try lowest index for simplicity first, or highest degree for better pruning.
        # Calculating degree in remaining subgraph for all vertices is expensive.
        # Let's pick the first available vertex.
        
        # Find the lowest set bit
        # lowest_bit = remaining_mask & -remaining_mask
        # v = lowest_bit.bit_length() - 1
        
        # Better heuristic: Pick vertex with max degree in remaining graph.
        # This might be slow to compute every step.
        # Let's stick to a simple greedy coloring bound or just simple branching.
        
        # Actually, a very effective simple MIS algorithm uses "Maximum Cardinality Search" ordering or similar.
        # But let's try a standard branch and bound with a simple node selection.
        
        # Select vertex v
        # Let's choose the vertex that appears in the most conflicts within the remaining set?
        # That requires iterating.
        
        # Let's just pick the lowest index available.
        lowest_bit = remaining_mask & -remaining_mask
        v = lowest_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New remaining: remove v and all its neighbors from remaining
        # neighbors_in_remaining = adj[v] & remaining_mask
        # new_remaining = remaining_mask & ~(1 << v) & ~adj[v]
        # Wait, if we include v, we must exclude v and all its neighbors.
        new_remaining_include = remaining_mask & ~(1 << v) & ~adj[v]
        new_mask_include = current_mask | (1 << v)
        solve(new_remaining_include, new_mask_include, current_size + 1)
        
        # Branch 2: Exclude v
        # New remaining: remove v
        new_remaining_exclude = remaining_mask & ~(1 << v)
        solve(new_remaining_exclude, current_mask, current_size)
        
    # Initial call
    # We can initialize with a greedy solution to have a good lower bound (best_size)
    # Greedy: pick vertices in order, if no conflict with already picked, add it.
    greedy_mask = 0
    for i in range(N):
        # Check if i conflicts with any in greedy_mask
        if (adj[i] & greedy_mask) == 0:
            greedy_mask |= (1 << i)
            
    best_size = greedy_mask.bit_count()
    best_set_mask = greedy_mask
    
    # Run exact solver
    # To prevent timeout, we can add a time check, but N=256 should be manageable with bitsets.
    # However, worst case is exponential.
    # Let's add a simple time guard.
    start_time = time.time()
    
    def solve_timed(remaining_mask, current_mask, current_size):
        nonlocal best_set_mask, best_size
        
        if time.time() - start_time > 50: # 50s limit to be safe
            return
            
        if current_size + remaining_mask.bit_count() <= best_size:
            return
            
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        lowest_bit = remaining_mask & -remaining_mask
        v = lowest_bit.bit_length() - 1
        
        # Include v
        new_remaining_include = remaining_mask & ~(1 << v) & ~adj[v]
        new_mask_include = current_mask | (1 << v)
        solve_timed(new_remaining_include, new_mask_include, current_size + 1)
        
        # Exclude v
        new_remaining_exclude = remaining_mask & ~(1 << v)
        solve_timed(new_remaining_exclude, current_mask, current_size)

    solve_timed((1 << N) - 1, 0, 0)
    
    # Convert mask to list of numbers (1-indexed)
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
