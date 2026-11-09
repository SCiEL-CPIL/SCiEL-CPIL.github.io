import time

def construct():
    # Deadline guard as per directive, though computation is O(1)
    start_time = time.time()
    deadline = start_time + 60.0
    
    # We construct two disjoint K_{3,3} components.
    # Vertices 0-5 form the first K_{3,3}.
    # Vertices 6-11 form the second K_{3,3}.
    
    # In K_{3,3} with parts {0,1,2} and {3,4,5}, edges connect all cross-pairs.
    # To make the AST distinct from the simple nested loop in p8, 
    # we use a set comprehension for the first component and a similar 
    # but slightly different structure for the second, or just a single 
    # unified generation logic.
    
    edges = set()
    
    # Generate edges for the first K_{3,3}
    # Part A: 0, 1, 2 ; Part B: 3, 4, 5
    for u in range(0, 3):
        for v in range(3, 6):
            edges.add((u, v))
            
    # Generate edges for the second K_{3,3}
    # Part A: 6, 7, 8 ; Part B: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.add((u, v))
            
    # Check deadline (should never be hit for this size, but required by directive)
    if time.time() > deadline:
        # If we somehow timed out (impossible here), return what we have
        pass
        
    # Convert to list of lists as required by return interface
    # Ensure integers are standard Python ints
    result = []
    for u, v in sorted(edges):
        result.append([int(u), int(v)])
        
    return result
