import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def find_regular_subgraph(adj, n):
        """
        Find any induced subgraph of size >= 6 that is regular.
        Returns the tuple of vertices or None.
        Uses bitmasks for fast degree checks.
        """
        # We check sizes from 6 to n.
        # To find a "bad" graph quickly, smaller subsets are more likely to be regular in random graphs?
        # Actually, in random graphs, small regular subgraphs (like K6 or C6) are more probable than large ones?
        # Let's check 6 first, then 7, etc.
        
        # Precompute bitmasks for vertices?
        # adj is a list of integers (bitmasks).
        
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                # Check if this subset is regular
                # Create mask
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degrees of all vertices in subset within the induced subgraph
                # deg(v) = popcount(adj[v] & mask)
                first_v = subset[0]
                target_deg = (adj[first_v] & mask).bit_count()
                
                is_regular = True
                for v in subset[1:]:
                    if (adj[v] & mask).bit_count() != target_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return subset
        return None

    def try_local_search(n, time_limit):
        """
        Attempt to find a valid graph on n vertices using local search.
        """
        # Initialize random graph as bitmasks
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Check initial graph
        S = find_regular_subgraph(adj, n)
        if S is None:
            return adj
        
        # Local search loop
        while time.time() < deadline - time_limit:
            # Find a regular subgraph
            S = find_regular_subgraph(adj, n)
            
            if S is None:
                return adj
            
            # Pick a random edge within S to flip
            # S is a tuple of vertices
            # We need to pick two vertices u, v from S
            u, v = random.sample(S, 2)
            
            # Flip edge (u, v)
            # Check if edge exists
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Optional: If we keep flipping the same edge back and forth, we might loop.
            # A simple random walk usually escapes.
            
        return None

    def graph_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try larger graphs first
    # n=21 is the target for score > 2.0 (if seed is 20, 21/20 = 1.05, wait. 
    # Score is value/seed. If seed is 20, value 20 is 1.0.
    # The prompt says "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
    # Wait, the score is "value/seed". 
    # Current score 2.2000 implies value=22? Or seed=10?
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # If current score is 2.2 and it found n=20, then seed must be 10? 
    # Or maybe the "value" is not just n.
    # "A valid graph on n vertices certifies G(6) > n."
    # "Maximize n."
    # If the current program scores 2.2, and it returns n=20, then 20/seed = 2.2 => seed = 9.09? Unlikely.
    # Maybe the value is related to the bound?
    # Let's assume the score is proportional to n.
    # If I find n=21, score should be 21/seed.
    # If I find n=20, score is 20/seed.
    # The instruction says "1.0 equals the seed program".
    # If seed program returns n=10, score 1.0.
    # If current program returns n=20, score 2.0? But it says 2.2.
    # Maybe the current program sometimes finds n=21? Or the value is not just n.
    # Let's assume maximizing n is the primary goal.
    
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this n
        # If we fail for 21, we move to 20.
        # Give 50s to 21, 50s to 20?
        t_limit = 50.0 if n == 21 else 40.0
        
        adj = try_local_search(n, t_limit)
        
        if adj is not None:
            edges = graph_to_edges(adj, n)
            return {'n': n, 'edges': edges}
            
    # Fallback: return a valid small graph
    # Empty graph on 5 vertices
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
