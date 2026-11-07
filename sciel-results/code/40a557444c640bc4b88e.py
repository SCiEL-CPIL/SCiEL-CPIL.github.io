"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    edges = set()
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.add(tuple(sorted((u, v))))
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.add(tuple(sorted((u, v))))
            
    # Check deadline (should never hit for this static construction)
    if time.time() > deadline:
        # Fallback: return a valid graph with higher cycle length count if we were doing search
        # But since we are static, we just return what we have.
        pass
        
    return [list(e) for e in sorted(edges)]
# EVOLVE-BLOCK-END
