import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    D = 11
    w = 4

    supports = list(itertools.combinations(range(D), w))
    all_patterns = list(itertools.product([1, -1], repeat=w))
    
    # Build all candidate vectors
    candidates = []
    for support in supports:
        for pattern in all_patterns:
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, pattern):
                v[idx] = s
            candidates.append(v)
    
    candidates_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates_arr)
    
    # Compute all pairwise dot products
    dots = candidates_arr @ candidates_arr.T
    
    # Conflict: dot > 2
    # Note: dot can be -4, -2, 0, 2, 4
    # We want to keep pairs where dot <= 2
    # So conflict is dot > 2 (i.e., dot == 3 or 4)
    conflict = dots > 2
    np.fill_diagonal(conflict, False)
    
    # Convert to bitsets for fast local search
    # n_cands = 5280, so we need 5280 bits per row
    # Python integers are arbitrary precision, so we can use them as bitsets
    
    adj_bits = [0] * n_cands
    for i in range(n_cands):
        bits = 0
        # Find neighbors
        # Using numpy where is faster for extraction
        neighbors = np.where(conflict[i])[0]
        for j in neighbors:
            bits |= (1 << j)
        adj_bits[i] = bits

    best_selected_bits = 0
    best_count = 0
    
    # Greedy Min-Degree with random tie-breaking
    # Run multiple restarts
    
    max_restarts = 100
    # Adjust restarts based on time
    # Each restart is O(N^2) roughly? No, bitsets make it fast.
    # Greedy step: find min degree in remaining.
    # Degree calculation with bitsets: popcount(adj[i] & remaining_bits)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        remaining_bits = (1 << n_cands) - 1
        selected_bits = 0
        
        # Greedy Min-Degree
        # To speed up, we can just iterate and pick min degree
        # But N=5280, N^2 = 27M. Popcount is fast.
        # 100 restarts * 27M ops = 2.7B ops. Might be too slow in Python.
        
        # Optimized Greedy:
        # Instead of full N^2, we can use a heap?
        # Or just random greedy?
        # Random greedy (pick random node, remove neighbors) is O(N) per step, O(N^2) total.
        # With bitsets, removing neighbors is `remaining_bits &= ~adj_bits[node]`.
        # This is very fast.
        
        # Let's do Random Greedy with Min-Degree preference
        # Actually, just Random Greedy is often good enough for dense-ish graphs?
        # Or "First Fit" heuristic.
        
        # Let's try:
        # 1. Randomly shuffle candidates
        # 2. Iterate, if node is in remaining, add it, remove its neighbors
        
        indices = list(range(n_cands))
        random.shuffle(indices)
        
        for node in indices:
            if time.time() > deadline - 5:
                break
            if (remaining_bits >> node) & 1:
                # Add node
                selected_bits |= (1 << node)
                # Remove neighbors
                remaining_bits &= ~adj_bits[node]
                # Remove node itself
                remaining_bits &= ~(1 << node)
                
        count = bin(selected_bits).count('1')
        if count > best_count:
            best_count = count
            best_selected_bits = selected_bits
            
        # If we have time, do local search on the best solution found so far?
        # Or just do local search on the current solution?
        
        # Local Search: 1-swap
        if time.time() > deadline - 10:
            continue
            
        cur_sel = selected_bits
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Pick a random node not in cur_sel
            # We need to find a node not in cur_sel
            # Not in cur_sel: ~(cur_sel) & ((1<<n_cands)-1)
            # Picking random bit from a large int is tricky.
            # We can maintain a list of candidates?
            # For now, let's just try a few random nodes.
            
            for _ in range(50):
                v = random.randint(0, n_cands - 1)
                if (cur_sel >> v) & 1:
                    continue
                    
                conflicts = adj_bits[v] & cur_sel
                if conflicts == 0:
                    # Can add v
                    cur_sel |= (1 << v)
                    improved = True
                    break
                    
                # Try 1-swap
                # Pick a random conflict u
                # Get list of set bits in conflicts
                # Since conflicts is a bitmask, we can extract them
                # But it might be many.
                # Let's just pick one random bit from conflicts
                # To pick a random set bit:
                # conflicts & (conflicts - 1) clears lowest set bit.
                # We can randomize by rotating?
                # Simpler: just try the lowest set bit.
                if conflicts:
                    # Extract lowest set bit
                    lsb = conflicts & -conflicts
                    u = lsb.bit_length() - 1
                    
                    # Try swapping u for v
                    new_sel = cur_sel & ~(1 << u)
                    new_conflicts = adj_bits[v] & new_sel
                    if new_conflicts == 0:
                        cur_sel = new_sel | (1 << v)
                        improved = True
                        break
                        
        count = bin(cur_sel).count('1')
        if count > best_count:
            best_count = count
            best_selected_bits = cur_sel

    # Convert best_selected_bits back to array
    selected_indices = [i for i in range(n_cands) if (best_selected_bits >> i) & 1]
    
    if len(selected_indices) == 0:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result = candidates_arr[selected_indices]
    return result
