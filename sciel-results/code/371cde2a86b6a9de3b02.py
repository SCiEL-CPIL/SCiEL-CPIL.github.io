import itertools
import time
import random

def construct():
    """
    Finds a 5-uniform hypergraph with no proper 2-coloring (Property B fails)
    by solving a Set Cover problem on the space of all 2-colorings.
    Minimizes the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try different n values
    # n=9 is the seed (126 edges).
    # n=11 is known to yield 51 edges.
    # We try 10, 11, 12, 13 to see if we can do better.
    
    for n in [11, 10, 12, 13]:
        if time.time() > deadline:
            break
            
        # Generate all edges
        all_edges = list(itertools.combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Universe size is 2^n
        U_size = 1 << n
        
        # Precompute masks for each edge
        # masks[i] is a bitmask of length U_size
        # Bit j is 1 if coloring j makes edge i monochromatic
        masks = [0] * num_edges
        
        # Optimization: Precompute bit positions for vertices
        # For each edge, we need to check if bits at positions v1..v5 are all 0 or all 1
        
        for idx, edge in enumerate(all_edges):
            mask = 0
            # Iterate through all colorings
            for c in range(U_size):
                # Check if edge is monochromatic in coloring c
                # Get bits for vertices in edge
                # c is an integer, bit i represents color of vertex i
                # We need to check if c & (1<<v) is 0 or 1 for all v in edge
                # If all 0 or all 1, it's monochromatic.
                
                # Efficient check:
                # Let m be the mask of the edge vertices
                # bits = c & m
                # If bits == 0 or bits == m, then monochromatic.
                
                # Construct edge_mask
                # We can precompute edge_mask for each edge outside the loop?
                # Yes.
                
                pass # Placeholder, will refactor below
                
        # Refactored Precomputation
        edge_masks = [0] * num_edges
        for idx, edge in enumerate(all_edges):
            em = 0
            for v in edge:
                em |= (1 << v)
            edge_masks[idx] = em
            
        for idx, edge in enumerate(all_edges):
            em = edge_masks[idx]
            mask = 0
            for c in range(U_size):
                bits = c & em
                if bits == 0 or bits == em:
                    mask |= (1 << c)
            masks[idx] = mask
            
        # Greedy Set Cover
        # We will run multiple randomized greedy attempts
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            uncovered = (1 << U_size) - 1
            selected = []
            remaining = set(range(num_edges))
            
            while uncovered:
                best_idx = -1
                best_gain = -1
                
                # Randomize order if tie-breaking is needed, or just pick max
                # To allow random tie-breaking, we can shuffle candidates or just pick one of the bests randomly
                # For speed, we find the max gain first.
                
                # Optimization: Only iterate over remaining
                for idx in remaining:
                    gain = bin(uncovered & masks[idx]).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        # If we want random tie-breaking, we can continue and collect all max, or just break?
                        # Standard greedy is deterministic. Randomized greedy helps escape local minima?
                        # Actually, Set Cover greedy is not known to be sensitive to tie-breaking for optimal solution finding, 
                        # but it might find different valid solutions.
                        # Let's just stick to deterministic first, then maybe add randomness later if time permits.
                        
                if best_idx == -1:
                    break # Should not happen
                    
                selected.append(best_idx)
                remaining.remove(best_idx)
                uncovered &= ~masks[best_idx]
                
            if len(selected) < best_edge_count:
                best_edge_count = len(selected)
                best_result = [all_edges[i] for i in selected]
                # If we found 51 or less, we can stop early? 
                # The known best is 51. If we find 50, we should keep searching.
                # If we find 51, we can break out of n loop if we are confident?
                # But maybe n=12 has 50? Unlikely.
                # Let's just record the best.
                
    # Fallback to seed if something went wrong or time ran out without result
    if best_result is None:
        # Seed: Complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return {'n': n, 'edges': [list(e) for e in best_result]}

# Note: The variable 'n' in the return statement of the loop scope might be the last n tried.
# We need to track the n associated with the best result.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_result = None
    best_edge_count = float('inf')
    best_n = 0
    
    for n in [11, 10, 12, 13]:
        if time.time() > deadline:
            break
            
        all_edges = list(itertools.combinations(range(n), 5))
        num_edges = len(all_edges)
        U_size = 1 << n
        
        # Precompute edge vertex masks
        edge_vmasks = [0] * num_edges
        for idx, edge in enumerate(all_edges):
            em = 0
            for v in edge:
                em |= (1 << v)
            edge_vmasks[idx] = em
            
        # Precompute coloring masks
        masks = [0] * num_edges
        for idx in range(num_edges):
            em = edge_vmasks[idx]
            mask = 0
            for c in range(U_size):
                bits = c & em
                if bits == 0 or bits == em:
                    mask |= (1 << c)
            masks[idx] = mask
            
        # Greedy Set Cover (Deterministic)
        # We can also add a randomized version if time allows
        uncovered = (1 << U_size) - 1
        selected = []
        remaining = set(range(num_edges))
        
        while uncovered:
            best_idx = -1
            best_gain = -1
            
            for idx in remaining:
                gain = bin(uncovered & masks[idx]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            remaining.remove(best_idx)
            uncovered &= ~masks[best_idx]
            
        if len(selected) < best_edge_count:
            best_edge_count = len(selected)
            best_result = [all_edges[i] for i in selected]
            best_n = n
            
    if best_result is None:
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_result]}
