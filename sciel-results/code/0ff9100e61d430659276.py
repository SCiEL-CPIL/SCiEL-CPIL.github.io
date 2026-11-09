import time
import random
import itertools

def _alpha_less_than_10(n, adj_bits):
    """
    Checks if the independence number of the graph is < 10.
    adj_bits: list of n integers, where adj_bits[i] is a bitmask of neighbors of i.
    """
    # Branch and Bound for Max Independent Set
    # We want to know if there is an independent set of size 10.
    # If we find one, return False. If we exhaust search, return True.
    
    # Greedy upper bound for pruning
    # Simple B&B:
    # State: (current_independent_set, candidates_mask)
    # candidates_mask: vertices that can be added to current_independent_set
    
    # To speed up, we can use a simple heuristic ordering.
    
    def bnb(candidates, current_size):
        if current_size >= 10:
            return True # Found independent set of size 10
        
        if candidates == 0:
            return False
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') < 10:
            return False
        
        # Pick a vertex from candidates
        # Heuristic: pick vertex with lowest degree in induced subgraph?
        # For simplicity, just pick the first bit.
        v = (candidates & -candidates).bit_length() - 1
        
        # Option 1: Include v
        # New candidates: candidates & ~adj_bits[v] & ~(1 << v)
        if bnb(candidates & ~adj_bits[v] & ~(1 << v), current_size + 1):
            return True
        
        # Option 2: Exclude v
        # New candidates: candidates & ~(1 << v)
        if bnb(candidates & ~(1 << v), current_size):
            return True
        
        return False

    # Initial candidates: all vertices
    all_mask = (1 << n) - 1
    return not bnb(all_mask, 0)

def _is_triangle_free(n, adj_bits):
    for i in range(n):
        # Check if any two neighbors of i are connected
        # For each neighbor j of i, check if j is connected to any other neighbor k of i
        # This is O(n * E) or similar. 
        # Faster: For each i, get mask of neighbors.
        # Check if (adj_bits[i] & adj_bits[j]) is non-zero for any j in adj_bits[i]?
        # Actually, triangle exists if there is an edge (j,k) where j,k in N(i).
        # So for each i, iterate neighbors j. If (adj_bits[i] & adj_bits[j]) has a bit set, triangle.
        # But we must be careful not to count (i,j) or (j,i).
        # Since graph is simple, adj_bits[i] contains j iff adj_bits[j] contains i.
        # If j is in N(i), and k is in N(i), and k is in N(j), then (i,j,k) is a triangle.
        # So for each i, for each j in N(i):
        #    if (adj_bits[i] & adj_bits[j]) != 0: return False
        # This is sufficient.
        mask_i = adj_bits[i]
        while mask_i:
            lsb = mask_i & -mask_i
            j = lsb.bit_length() - 1
            if (mask_i & adj_bits[j]):
                return False
            mask_i ^= lsb
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Target n=40
    n = 40
    best_result = None
    
    # Try to find a valid graph on n=40
    # Strategy: Randomized search for maximal triangle-free graphs
    # We will generate random graphs, enforce triangle-free, check alpha.
    
    # Optimization: Precompute bitmasks for vertices
    # We will use a list of integers for adj_bits
    
    # To improve success rate, we can bias the random generation.
    # Start with a random graph, remove edges to break triangles.
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Enforce triangle-free
        # Repeatedly find a triangle and remove an edge
        # This is slow if done naively.
        # Better: Start with empty graph, add edges randomly if no triangle created.
        # This guarantees maximal triangle-free.
        
        adj_bits = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    # Check if adding (i,j) creates a triangle
                    # Common neighbors of i and j must be empty
                    # Common neighbors: adj_bits[i] & adj_bits[j]
                    if (adj_bits[i] & adj_bits[j]) == 0:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        edges.append([i, j])
        
        # Now adj_bits represents a maximal triangle-free graph
        # Check alpha
        if _alpha_less_than_10(n, adj_bits):
            return {'n': n, 'edges': edges}
            
    # If n=40 failed, try n=39
    n = 39
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        adj_bits = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    if (adj_bits[i] & adj_bits[j]) == 0:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        edges.append([i, j])
        
        if _alpha_less_than_10(n, adj_bits):
            return {'n': n, 'edges': edges}
            
    # Fallback: Return the seed graph (n=26) or a known smaller valid graph
    # The seed graph is Andrasfai(9) on 26 vertices.
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
