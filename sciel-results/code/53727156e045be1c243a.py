import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
        
    # 2. Filter vertices: 2*v must not be a square
    # Map valid vertices to 0-based indices for bitset operations
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    if not valid_vertices:
        return [1]

    M = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build conflict bitsets
    # conflict_bits[i] is an integer where bit j is 1 if valid_vertices[i] + valid_vertices[j] is a square
    conflict_bits = [0] * M
    
    for i in range(M):
        vi = valid_vertices[i]
        bits = 0
        for j in range(M):
            if i != j:
                vj = valid_vertices[j]
                if vi + vj in squares:
                    bits |= (1 << j)
        conflict_bits[i] = bits

    # Helper to check if a vertex can be added to a set represented as a bitset
    # S_bits: integer representing the set of indices currently in the solution
    # i: index of the vertex to check
    # If (S_bits & conflict_bits[i]) == 0, then no conflict
    def can_add(S_bits, i):
        return (S_bits & conflict_bits[i]) == 0

    def greedy_init():
        # Sort vertices by degree (number of conflicts) ascending to pick "easy" ones first?
        # Or descending to pick "hard" ones first? 
        # Usually, picking vertices with fewer conflicts first allows more vertices to be added later.
        # Let's try both or just one.
        indices = list(range(M))
        # Degree is the number of set bits in conflict_bits[i]
        degrees = [bin(conflict_bits[i]).count('1') for i in range(M)]
        indices.sort(key=lambda x: degrees[x])
        
        S_bits = 0
        for i in indices:
            if can_add(S_bits, i):
                S_bits |= (1 << i)
        return S_bits

    def local_search(S_bits):
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any vertex
            for i in range(M):
                if not (S_bits & (1 << i)):
                    if can_add(S_bits, i):
                        S_bits |= (1 << i)
                        improved = True
            
            # 2. Try to swap: remove u, add v
            # This is expensive O(M^2). We do it if time permits or as a second pass.
            # Let's do a simpler "remove and refill" pass first.
            if not improved:
                # Try removing one vertex and refilling greedily
                best_S = S_bits
                best_size = bin(S_bits).count('1')
                
                # We only consider removing vertices that are in the set
                # To save time, we can limit the number of swaps or use a heuristic
                for u in range(M):
                    if S_bits & (1 << u):
                        # Remove u
                        new_S = S_bits & ~(1 << u)
                        
                        # Refill greedily
                        # We can try to add any vertex that was blocked by u
                        # Or just run the add-pass on new_S
                        # To be efficient, we just try to add all possible vertices
                        # This is O(M) per removal, total O(M^2)
                        
                        # Optimization: Only try to add vertices that were in conflict with u
                        # If v was not in conflict with u, and v was not in S, it would have been added in step 1.
                        # So only vertices that were in conflict with u are candidates for addition now.
                        candidates = [v for v in range(M) if (conflict_bits[u] & (1 << v)) and not (new_S & (1 << v))]
                        
                        for v in candidates:
                            if can_add(new_S, v):
                                new_S |= (1 << v)
                                # After adding one, we might be able to add more?
                                # The "add any" pass in the next iteration of the while loop will handle cascading adds.
                                # But we should check if this specific removal leads to a better state immediately.
                        
                        size = bin(new_S).count('1')
                        if size > best_size:
                            best_size = size
                            best_S = new_S
                            improved = True
                            
                if improved:
                    S_bits = best_S
                    # Loop again to catch any further improvements
                    continue
            else:
                # If we added something, loop again to check for more adds
                continue
                
        return S_bits

    # Initial solution
    best_S_bits = greedy_init()
    best_S_bits = local_search(best_S_bits)
    best_size = bin(best_S_bits).count('1')
    
    # Iterated Local Search
    while time.time() < deadline - 2.0:
        # Perturb: remove k random vertices
        k = random.randint(1, 5)
        if best_size <= k:
            break
        current_S_bits = best_S_bits
        # Randomly remove k vertices
        set_vertices = [i for i in range(M) if current_S_bits & (1 << i)]
        to_remove = random.sample(set_vertices, min(k, len(set_vertices)))
        for i in to_remove:
            current_S_bits &= ~(1 << i)
            
        # Local search on perturbed solution
        new_S_bits = local_search(current_S_bits)
        new_size = bin(new_S_bits).count('1')
        
        if new_size > best_size:
            best_size = new_size
            best_S_bits = new_S_bits

    # Convert bitset to list of values
    result = []
    for i in range(M):
        if best_S_bits & (1 << i):
            result.append(valid_vertices[i])
            
    return result
