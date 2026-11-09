"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return a nonempty subset of [1, 512] with lcm(a,b)<=512 for every pair.
    Maximize cardinality.
    
    Strategy:
    1. Start with all numbers 1..512.
    2. Iteratively remove the number that conflicts with the most other numbers.
    3. Use a greedy heuristic to build a large independent set in the conflict graph.
    4. Additionally, include all numbers that are divisors of 512 (powers of 2) and their multiples
       that stay within bounds, but actually the conflict graph approach is more general.
       
    Better strategy:
    - Build conflict graph where edge (a,b) exists if lcm(a,b) > 512.
    - Find maximum independent set. Since graph is dense, use greedy: 
      repeatedly pick the vertex with minimum degree in the conflict graph, 
      add it to solution, remove it and its neighbors.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    N = 512
    numbers = list(range(1, N + 1))
    
    # Precompute lcm for all pairs? That's 512*512/2 = ~131k pairs, feasible.
    # But we can be smarter: build conflict graph on the fly.
    
    # For each pair (a, b), conflict if lcm(a,b) > 512.
    # lcm(a,b) = a * b // gcd(a,b)
    
    # Let's build the conflict graph as a set of neighbors for each node.
    # Node indices: 0..511 correspond to numbers 1..512.
    
    n = N
    # adjacency list for conflict graph
    # adj[i] = set of j such that lcm(numbers[i], numbers[j]) > 512
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = math.gcd(a, b)
            lcm_val = a * b // g
            if lcm_val > N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Now find maximum independent set using greedy min-degree heuristic.
    # We'll maintain a set of remaining nodes.
    remaining = set(range(n))
    solution = set()
    
    while remaining:
        # Find node with minimum degree in the induced subgraph on remaining
        min_deg = float('inf')
        best_node = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = v
                if deg == 0:
                    break
        
        # Add best_node to solution
        solution.add(best_node)
        
        # Remove best_node and all its neighbors from remaining
        neighbors = adj[best_node] & remaining
        remaining.discard(best_node)
        for nb in neighbors:
            remaining.discard(nb)
        
        # Check time
        if time.time() > deadline:
            break
    
    # Convert back to numbers
    result = [v + 1 for v in solution]
    
    # Verify the solution is valid
    # For safety, verify a subset if needed, but let's trust the construction.
    # Actually, let's do a quick verification on the returned set to be safe.
    if len(result) > 1:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a = result[i]
                b = result[j]
                g = math.gcd(a, b)
                if a * b // g > N:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to seed if something went wrong
            return list(range(1, math.isqrt(N) + 1))
    
    return result
# EVOLVE-BLOCK-END
