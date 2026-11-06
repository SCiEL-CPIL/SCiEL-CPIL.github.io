import random
import time
from itertools import combinations

def is_2_colorable_brute(n, edges):
    """Check if hypergraph is 2-colorable by checking all 2^n colorings."""
    # Precompute bitmask for each edge
    # Each edge is a tuple of vertices.
    # A coloring is a bitmask of n bits.
    # An edge is monochromatic if all bits are 0 or all bits are 1.
    # We want to know if there exists a coloring such that NO edge is monochromatic.
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Iterate all colorings
    for c in range(1 << n):
        valid = True
        for mask in edge_masks:
            # Check if edge is monochromatic
            # Monochromatic if (c & mask) == 0 or (c & mask) == mask
            if (c & mask) == 0 or (c & mask) == mask:
                valid = False
                break
        if valid:
            return True
    return False

def is_2_colorable_backtrack(n, edges):
    """Backtracking with unit propagation for larger n."""
    # This is a simplified version. 
    # For n <= 15, brute force is faster in Python due to overhead.
    # For n > 15, backtracking is necessary.
    # We will use brute force for n <= 15 and backtrack for n > 15.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # We want to minimize edges.
    # Start with n=9.
    # Max edges for n=9 is 126.
    # Known lower bound is 29.
    
    best_result = None
    best_e = 127  # Initialize to something high
    
    # Try n from 9 upwards
    for n in range(9, 27):
        if time.time() > deadline:
            break
            
        max_edges = len(list(combinations(range(n), 5)))
        
        # We want to find a non-2-colorable hypergraph with few edges.
        # Try E from 29 upwards.
        # But for n=9, we know 126 works.
        # Let's try to find a smaller one.
        
        # If n is large, max_edges is huge.
        # We should limit the range of E we try.
        # For n=9, try 29 to 126.
        # For n=10, try 29 to ...?
        # The heuristic E ~ 2^(n-4) suggests:
        # n=9 -> 32
        # n=10 -> 64
        # n=11 -> 128
        # So for n=10, we might need more edges?
        # Actually, we want MIN E.
        # So we should try low E first.
        
        # However, if n is large, low E is likely 2-colorable.
        # So for each n, we should try a range of E.
        
        # Let's prioritize n=9 first, as it has the smallest n and thus smallest brute-force cost.
        # And the heuristic suggests low E is possible.
        
        if n == 9:
            # Try E from 29 to 126
            for E in range(29, 127):
                if time.time() > deadline:
                    break
                # Try random samples
                for _ in range(500): # 500 attempts per E
                    if time.time() > deadline:
                        break
                    edges = random.sample(list(combinations(range(n), 5)), E)
                    # Convert to lists for return format
                    edges_list = [list(e) for e in edges]
                    if not is_2_colorable_brute(n, edges_list):
                        # Found a non-2-colorable hypergraph
                        return {'n': n, 'edges': edges_list}
        
        elif n <= 15:
            # Brute force is still feasible (2^15 = 32768)
            # Try E from 29 to ...?
            # Let's try up to 100 edges for simplicity, or until time runs out.
            # But we want MIN E.
            # So try 29, 30, ...
            # But if n=10, E=29 is likely 2-colorable.
            # We should jump E based on heuristic?
            # No, just try increasing E.
            # But 500 attempts per E might be too many if we go high.
            # Let's limit total attempts.
            
            # For n > 9, we can try fewer attempts per E.
            for E in range(29, 101): 
                if time.time() > deadline:
                    break
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    edges = random.sample(list(combinations(range(n), 5)), E)
                    edges_list = [list(e) for e in edges]
                    if not is_2_colorable_brute(n, edges_list):
                        return {'n': n, 'edges': edges_list}
        
        else:
            # n > 15, need backtracking.
            # This is complex to implement robustly in the remaining time.
            # We will rely on n <= 15.
            pass
            
    # If we didn't find anything better, return the seed solution
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
