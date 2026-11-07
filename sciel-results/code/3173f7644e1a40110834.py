import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Initialize adjacency bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # We will run multiple randomized greedy attempts
    # Deadline guard
    start_time = time.time()
    deadline = 50.0 # 50 seconds to be safe
    
    best_partition = None
    best_count = float('inf')
    
    # Precompute list of vertices
    vertices = list(range(n))
    
    # Generate candidate left sides: all singletons and all pairs
    # We will evaluate these in each greedy step
    # To speed up, we can just iterate.
    
    for attempt in range(100):
        if time.time() - start_time > deadline:
            break
            
        random.seed(attempt)
        
        # uncovered_adj[i] is bitmask of neighbors j such that (i,j) is uncovered
        uncovered_adj = adj[:]
        partition = []
        
        # Helper to get list of set bits from a mask
        def get_bits(mask):
            bits = []
            while mask:
                lsb = mask & -mask
                i = lsb.bit_length() - 1
                bits.append(i)
                mask ^= lsb
            return bits
        
        while True:
            # Check if any uncovered edges remain
            # If sum of popcounts is 0, we are done
            # Note: each edge is counted twice in sum of popcounts
            remaining_edges = sum(bin(x).count('1') for x in uncovered_adj) // 2
            if remaining_edges == 0:
                break
            
            best_L = None
            best_R = None
            best_covered = 0
            best_rand = 0 # for tie breaking
            
            # Try all singletons
            for u in vertices:
                if uncovered_adj[u] == 0:
                    continue
                L = [u]
                # R is neighbors of u in uncovered_adj, excluding u (self-loop not possible here as graph is simple)
                R_mask = uncovered_adj[u]
                # Ensure disjoint sides: u not in R
                R_mask &= ~(1 << u)
                if R_mask == 0:
                    continue
                covered = bin(R_mask).count('1')
                rand_val = random.random()
                if covered > best_covered or (covered == best_covered and rand_val > best_rand):
                    best_covered = covered
                    best_L = L
                    best_R = get_bits(R_mask)
                    best_rand = rand_val
            
            # Try all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    # Intersection of neighbors in uncovered graph
                    common = uncovered_adj[i] & uncovered_adj[j]
                    # Exclude i and j from R
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    if common == 0:
                        continue
                    covered = 2 * bin(common).count('1')
                    rand_val = random.random()
                    if covered > best_covered or (covered == best_covered and rand_val > best_rand):
                        best_covered = covered
                        best_L = [i, j]
                        best_R = get_bits(common)
                        best_rand = rand_val
            
            if best_L is None:
                # Fallback: pick any uncovered edge
                # Find an edge (u,v) in uncovered
                for u in vertices:
                    mask = uncovered_adj[u]
                    if mask:
                        v = (mask & -mask).bit_length() - 1
                        best_L = [u]
                        best_R = [v]
                        break
                if best_L is None:
                    break # Should not happen
            
            # Add biclique to partition
            partition.append([best_L, best_R])
            
            # Remove covered edges from uncovered_adj
            L_mask = 0
            for u in best_L:
                L_mask |= (1 << u)
            R_mask = 0
            for v in best_R:
                R_mask |= (1 << v)
            
            for u in best_L:
                uncovered_adj[u] &= ~R_mask
            for v in best_R:
                uncovered_adj[v] &= ~L_mask
        
        if len(partition) < best_count:
            best_count = len(partition)
            best_partition = partition
            
    # Format output
    # Each biclique is [left_vertices, right_vertices]
    # Ensure types are list of int
    if best_partition is None:
        # Fallback to seed-like behavior if something went wrong
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
        
    result = []
    for L, R in best_partition:
        # Convert to lists of ints
        result.append([[int(x) for x in L], [int(x) for x in R]])
        
    return result
