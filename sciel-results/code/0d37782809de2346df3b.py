import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Identify bad vertices (self-loop: 2v is a square)
    def is_square(n):
        if n < 0:
            return False
        r = int(n**0.5)
        return r*r == n or (r+1)*(r+1) == n

    bad_vertices = set()
    for v in vertices:
        if is_square(2 * v):
            bad_vertices.add(v)
            
    valid_vertices = [v for v in vertices if v not in bad_vertices]
    
    # Precompute conflicts for valid vertices
    # conflicts[v] is a set of u such that u+v is a square
    conflicts = {v: set() for v in valid_vertices}
    squares = set()
    for i in range(2, 23): # 2^2=4 to 22^2=484
        squares.add(i*i)
        
    for v in valid_vertices:
        for s in squares:
            u = s - v
            if 1 <= u <= N and u != v and u in valid_vertices:
                if u not in conflicts[v]:
                    conflicts[v].add(u)
                # Symmetry is handled by iterating all v, but we only need one direction for check
                # Actually, for efficiency in check, we can just store neighbors.
    
    best_set = set()
    best_size = 0
    
    # Run randomized greedy multiple times
    while time.time() < deadline:
        # Shuffle valid vertices
        shuffled = valid_vertices[:]
        random.shuffle(shuffled)
        
        current_set = set()
        for v in shuffled:
            # Check if v conflicts with any in current_set
            can_add = True
            for u in current_set:
                if u in conflicts[v]:
                    can_add = False
                    break
            if can_add:
                current_set.add(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # Try local improvement on the best set found
            # This is optional but can help. 
            # Given the low degree, greedy is usually quite good.
            # Let's add a simple local search: try to add any vertex not in set.
            # Since greedy produces a maximal independent set, no single vertex can be added.
            # So we need swaps.
            # For now, let's just rely on many random restarts.
            # If we have time, we can add swaps.
            
    # If we found nothing (shouldn't happen), return a valid singleton
    if not best_set:
        # Find any valid vertex
        for v in valid_vertices:
            best_set.add(v)
            break
        if not best_set:
             # If all are bad? 1 is not bad (2 is not square). So valid_vertices is not empty.
             pass
             
    return list(best_set)
