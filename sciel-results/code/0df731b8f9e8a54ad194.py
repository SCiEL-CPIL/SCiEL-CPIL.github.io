import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # 1. Filter out vertices where 2*v is a perfect square
    valid_vertices = []
    for v in range(1, N + 1):
        s = 2 * v
        root = int(math.isqrt(s))
        if root * root != s:
            valid_vertices.append(v)
    
    # Map original values to indices for bitset operations
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    M = len(valid_vertices)
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitmask of neighbors of valid_vertices[i]
    adj = [0] * M
    
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    # Note: min sum is 1+2=3, max is 255+256=511.
    # Squares in range: 4, 9, ..., 484
    
    for i in range(M):
        v = valid_vertices[i]
        for j in range(i + 1, M):
            w = valid_vertices[j]
            if v + w in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 3. Branch and Bound for Maximum Independent Set
    best_set = [0] * M  # Boolean array indicating if vertex is in best MIS
    best_size = 0
    
    # We use a recursive search. To avoid recursion depth issues, we could use a stack, 
    # but M=245 is small enough for recursion if we are careful, or we can just use recursion.
    # Python recursion limit might be an issue if depth is > 1000, but M is 245.
    
    # Heuristic for ordering: 
    # We will dynamically pick the vertex with minimum degree in the remaining subgraph.
    # This requires calculating degrees on the fly, which is expensive.
    # Alternative: Static ordering by degree (ascending).
    # Let's try static ordering first for simplicity and speed.
    
    # Calculate degrees
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    # Sort vertices by degree ascending
    order = sorted(range(M), key=lambda x: degrees[x])
    
    # Create a mapping from original index to position in sorted order?
    # Actually, it's easier to just iterate through the sorted list.
    # However, the "exclude" branch keeps the same set of available vertices minus the current one.
    # The "include" branch removes neighbors.
    
    # Let's use a recursive function with bitmasks.
    # available: bitmask of vertices that are still candidates.
    # current_mask: bitmask of vertices chosen so far.
    # current_size: number of vertices chosen so far.
    
    # To improve pruning, we can use a greedy coloring bound on the complement?
    # Or simply: upper_bound = current_size + popcount(available)
    # This is weak.
    
    # A better bound for MIS:
    # If we color the graph, the independence number is at least the size of the largest color class? No.
    # For Clique, coloring gives an upper bound.
    # For MIS, we can find a maximal independent set greedily to get a lower bound for pruning? No, we want upper bound to prune.
    # Upper bound for MIS:
    # 1. Number of vertices in available.
    # 2. If we can partition available into cliques, the MIS is at most the number of cliques.
    #    Finding clique cover is hard.
    # 3. A simple upper bound: sum of 1/deg(v) + 1 for v in available? (Lovasz)
    #    Let's stick to `popcount(available)` as it's fast, and rely on the min-degree branching to be effective.
    
    def popcount(x):
        return x.bit_count()
    
    def search(available, current_size, current_mask):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set from current_mask
                for i in range(M):
                    if current_mask & (1 << i):
                        best_set[i] = 1
                    else:
                        best_set[i] = 0
            return
            
        # Pruning
        if current_size + popcount(available) <= best_size:
            return
            
        # Select vertex to branch on
        # Strategy: Pick vertex with minimum degree in the induced subgraph `available`
        # This is O(M^2) per node if done naively.
        # Let's try picking the first vertex in `available` that has minimum static degree?
        # Or just the first available vertex?
        # Min-degree in induced subgraph is much better.
        
        min_deg = M + 1
        best_v = -1
        avail_list = []
        temp_avail = available
        while temp_avail:
            # Get lowest set bit
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            avail_list.append(v)
            temp_avail ^= lsb
            
        # Calculate degree in available subgraph for each candidate
        # To optimize, we can just check the first few? No, min degree is key.
        for v in avail_list:
            # Degree of v in available is popcount(adj[v] & available)
            deg = popcount(adj[v] & available)
            if deg < min_deg:
                min_deg = deg
                best_v = v
                if deg == 0:
                    break # Can't get better than 0
        
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New available: remove best_v and its neighbors
        new_available = available & ~(1 << best_v) & ~adj[best_v]
        new_mask = current_mask | (1 << best_v)
        search(new_available, current_size + 1, new_mask)
        
        # Branch 2: Exclude best_v
        # New available: remove best_v
        new_available = available & ~(1 << best_v)
        search(new_available, current_size, current_mask)

    initial_available = (1 << M) - 1
    search(initial_available, 0, 0)
    
    # Construct result
    result = []
    for i in range(M):
        if best_set[i]:
            result.append(valid_vertices[i])
            
    # Ensure non-empty
    if not result:
        # Fallback: just pick one valid vertex
        if valid_vertices:
            result = [valid_vertices[0]]
        else:
            result = [1] # 1 is valid (2*1=2 not square)

    return result

# Test locally if needed
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     # Verify
#     def is_square(x):
#         r = int(math.isqrt(x))
#         return r*r == x
#     valid = True
#     for i in range(len(res)):
#         a = res[i]
#         if is_square(2*a):
#             valid = False
#         for j in range(i+1, len(res)):
#             b = res[j]
#             if is_square(a+b):
#                 valid = False
#     print("Valid:", valid)
