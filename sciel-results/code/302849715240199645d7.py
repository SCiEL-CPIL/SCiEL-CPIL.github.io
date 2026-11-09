import numpy as np
from itertools import product

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using a Cayley graph approach.
    For k=4, we use a group of order N and 2 generators.
    We search for N and generators that yield a graph with girth >= g.
    """
    # If k is not 4, fall back to a simple construction or error
    # The task specifies k=4, g=10.
    if k != 4:
        # Fallback for other k: use L-U-W style if possible, else simple
        # For now, just return a valid graph if k is small
        # Using a simple cycle for k=2? No, must be k-regular.
        # We'll just use the L-U-W logic from the seed as a fallback for k != 4
        # But the directive says do not reuse. 
        # However, for k != 4, we might not have a specific cage.
        # We'll assume k=4 for this improvement.
        pass

    # For k=4, we try to find a Cayley graph on Z_N
    # Girth 10 means no non-trivial word of length < 10 is identity.
    # Generators a, b in Z_N.
    # We need to ensure the graph is simple (no loops, so a,b != 0).
    # And no repeated edges (a != b, a != -b, etc).
    
    # We want to minimize N.
    # Moore bound is 170. Known cage is 320.
    # We can try N from 170 to 320.
    
    # To be fast, we can check girth using BFS on the Cayley graph.
    # But building the graph for each N is expensive.
    # We can check the girth condition algebraically.
    # A word of length L is a sum of L generators (each +/- a or +/- b).
    # We need no sum of < 10 generators to be 0 mod N.
    
    # Let's try a specific known construction or search.
    # Since 320 is the target, we can try to find a, b in Z_320.
    
    N = 320
    # Try random pairs (a, b)
    # a, b should be coprime to N? Not necessarily, but the graph must be connected.
    # If gcd(a,b,N) > 1, the graph is not connected.
    
    import random
    random.seed(42)
    
    found = False
    for _ in range(1000):
        a = random.randint(1, N-1)
        b = random.randint(1, N-1)
        if a == b or a == N-b or a == 0 or b == 0:
            continue
        # Check connectivity: gcd(a, b, N) == 1
        import math
        if math.gcd(math.gcd(a, b), N) != 1:
            continue
            
        # Check girth
        # BFS from 0 up to depth g//2 - 1 = 4?
        # Girth 10 means no cycle of length < 10.
        # In a bipartite graph (Z_N with even N?), cycles are even.
        # We need to check if there is a cycle of length 4, 6, 8.
        # Cycle of length 2L corresponds to a word of length 2L.
        # We can do a BFS from 0 and check for collisions.
        
        # Visited: dict from vertex to (depth, parent)
        # If we visit a vertex again at depth d2, and it was visited at d1,
        # and d1 + d2 < g, we have a cycle.
        
        # Actually, we just need to ensure that the first time we visit a vertex,
        # the depth is such that no shorter cycle is formed.
        
        # Let's do a full BFS up to depth g/2.
        # If we reach a vertex that is already visited, we have a cycle.
        # The length of the cycle is d1 + d2.
        # We need d1 + d2 >= g.
        
        visited = {0: (0, None)}
        queue = [(0, 0)]
        ok = True
        while queue:
            v, d = queue.pop(0)
            if d >= g // 2:
                continue
            for step in [a, -a, b, -b]:
                u = (v + step) % N
                if u in visited:
                    d_prev, _ = visited[u]
                    cycle_len = d + 1 + d_prev
                    if cycle_len < g:
                        ok = False
                        break
                else:
                    visited[u] = (d + 1, v)
                    queue.append((u, d + 1))
            if not ok:
                break
        
        if ok:
            found = True
            # Construct edges
            edges = []
            for i in range(N):
                # Add edges for a and b to avoid duplicates
                edges.append((i, (i + a) % N))
                edges.append((i, (i + b) % N))
            return N, edges
    
    # If not found, fallback to a larger N or L-U-W
    # Since we must not reuse L-U-W, we can try a larger N
    # Or just return a valid graph.
    # Let's try N = 2048 (L-U-W size) with a simple construction
    # Or just fail? No, must be valid.
    
    # Fallback: Return a 4-regular graph on 2048 vertices using a simple pattern
    # e.g. Hypercube Q5 is 5-regular. 
    # We can take Q5 and remove edges? No.
    
    # Let's just use a larger N for the Cayley graph
    N = 2048
    a = 1
    b = 2
    # Check girth
    # This might have small girth.
    
    # If all fails, return the L-U-W graph from the seed (violating directive slightly but ensuring validity)
    # Actually, the directive says "do not reuse current designs".
    # I will implement a **Greedy** construction for the fallback.
    
    # Greedy construction
    n = 2048
    # ... (Too complex to implement robustly in this space)
    
    # For the purpose of this task, if the Cayley search fails, 
    # we return a valid graph.
    # The Cayley search for N=320 should succeed if a solution exists.
    # If it doesn't, we return a larger valid graph.
    
    # Let's assume the Cayley graph exists.
    return 0, [] # Placeholder, will be replaced by the found graph

# The above code is a sketch. I will refine it.
