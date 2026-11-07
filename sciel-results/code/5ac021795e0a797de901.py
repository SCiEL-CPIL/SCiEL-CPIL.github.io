"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    n = 15
    deadline = time.time() + 80  # safety margin before 90s kill
    
    best_edges = 0
    best_partition = None
    
    # Try all ways to partition 15 vertices into components K_{a,b} with a != b
    # Each component uses a+b vertices, contributes a*b edges
    # We need sum(a_i + b_i) = 15, a_i != b_i for all i
    
    # Generate all multisets of (a,b) pairs with a != b, a+b >= 3 (min component size for K_{a,b} with a,b>=1)
    # Actually a,b >= 1, and a != b, so min size is 1+2=3
    
    # Use recursion to find all valid partitions
    def search(remaining, current_components, current_edges):
        nonlocal best_edges, best_partition
        if time.time() > deadline:
            return
        if remaining == 0:
            if current_edges > best_edges:
                best_edges = current_edges
                best_partition = list(current_components)
            return
        if remaining < 3:
            return
        # Try all (a,b) with a != b, a+b <= remaining, a>=1, b>=1
        for a in range(1, remaining + 1):
            for b in range(1, remaining - a + 1):
                if a == b:
                    continue
                if a + b > remaining:
                    break
                # Avoid duplicates: require a <= b for ordering, but a != b so a < b
                if a >= b:
                    continue
                current_components.append((a, b))
                search(remaining - a - b, current_components, current_edges + a * b)
                current_components.pop()
    
    search(n, [], 0)
    
    if best_partition is None:
        # Fallback: star graph
        return [[0, v] for v in range(1, n)]
    
    # Build the graph from the best partition
    edges = []
    vertex = 0
    for a, b in best_partition:
        # Part A: vertices vertex to vertex+a-1
        # Part B: vertices vertex+a to vertex+a+b-1
        for i in range(a):
            for j in range(b):
                edges.append([vertex + i, vertex + a + j])
        vertex += a + b
    
    # Ensure all values are Python int and edges are lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
