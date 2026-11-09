import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Known valid graph: 8 disjoint K4s, n=32, alpha=8
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find n=33 graph with quick search
    # Given the difficulty, do a limited search
    if time.time() - start_time < 10:
        # Try adding a vertex to 8 K4s with careful connections
        # We showed this is impossible, so skip
        
        # Alternative: try a completely different construction
        # For now, just return the known good result
        pass
    
    return {'n': best_n, 'edges': best_edges}
