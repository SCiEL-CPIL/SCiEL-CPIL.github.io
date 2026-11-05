import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict sets
    # conflict[i] is a set of j such that (i+1) + (j+1) is a square
    # We use 0-indexed internally for efficiency, but return 1-indexed
    conflict = [set() for _ in range(N)]
    valid_vertices = []
    
    for i in range(N):
        a = i + 1
        # Check self conflict
        if 2 * a in squares:
            continue # This vertex cannot be in the set
        
        valid_vertices.append(i)
        
        for s in squares:
            b = s - a
            if 1 <= b <= N:
                j = b - 1
                conflict[i].add(j)
                
    # If a vertex is invalid, it shouldn't be considered, but its conflicts 
    # with valid vertices are still relevant if the invalid vertex was in S? 
    # No, invalid vertices are never in S. 
    # However, if we only iterate over valid_vertices, we must ensure 
    # that we don't accidentally consider conflicts with invalid vertices 
    # as blocking. But since invalid vertices are never in S, they don't block anything.
    # We just need to ensure our greedy selection only picks from valid_vertices.
    
    # Filter conflict sets to only contain valid vertices? 
    # It's not strictly necessary if we only check against S, and S only contains valid vertices.
    
    def is_compatible(u, S_set):
        """Check if u can be added to S_set."""
        # u is 0-indexed
        for v in S_set:
            if v in conflict[u]:
                return False
        return True

    def greedy_construct(order):
        S = set()
        for u in order:
            if is_compatible(u, S):
                S.add(u)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add any compatible vertex
            for u in valid_vertices:
                if u not in S:
                    if is_compatible(u, S):
                        S.add(u)
                        improved = True
        return S

    best_S = set()
    
    # Initial strategy: Sort by degree (number of conflicts with valid vertices)
    # Calculate degree within valid vertices
    degrees = {}
    for u in valid_vertices:
        deg = 0
        for v in conflict[u]:
            if v in valid_vertices: # Only count conflicts with valid vertices
                 deg += 1
        degrees[u] = deg
        
    # Greedy by degree (ascending)
    order_by_degree = sorted(valid_vertices, key=lambda x: degrees[x])
    current_S = greedy_construct(order_by_degree)
    current_S = local_search(current_S)
    if len(current_S) > len(best_S):
        best_S = current_S
        
    # Try random restarts if time permits
    while time.time() < start_time + DEADLINE:
        # Randomize order with some bias towards low degree
        # Or just shuffle
        random_order = valid_vertices[:]
        random.shuffle(random_order)
        
        # Maybe mix in some degree-based sorting?
        # Let's just do pure random shuffle for diversity
        current_S = greedy_construct(random_order)
        current_S = local_search(current_S)
        
        if len(current_S) > len(best_S):
            best_S = current_S
            
        # Optional: Try a local search that allows swaps?
        # For now, simple addition is fast.
        
    # Convert back to 1-indexed list
    result = [i + 1 for i in best_S]
    # Ensure non-empty
    if not result:
        # Fallback: pick any valid vertex
        if valid_vertices:
            return [valid_vertices[0] + 1]
        else:
            return [1] # Should not happen as 1+1=2 not square
            
    return result
