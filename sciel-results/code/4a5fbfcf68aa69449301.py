import time
import sys

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe
    
    N = 256
    # Vertices are 0 to 255, representing numbers 1 to 256
    # adj[v] is a bitmask of vertices u such that (u+1, v+1) is a conflict
    
    # 1. Build Conflict Graph
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            # Check if (a+b) divides (a*b)
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Greedy Initial Solution
    # Heuristic: Pick vertex with minimum degree in current induced subgraph
    def greedy_mis():
        C = (1 << N) - 1
        S = 0 # Use bitmask for set
        while C:
            # Find vertex in C with min degree in G[C]
            min_deg = float('inf')
            best_v = -1
            c_temp = C
            while c_temp:
                # Get lowest set bit
                lsb = c_temp & -c_temp
                v = lsb.bit_length() - 1
                # Degree of v in G[C] is popcount(adj[v] & C)
                deg = (adj[v] & C).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                c_temp ^= lsb
            
            if best_v == -1:
                break
                
            S |= (1 << best_v)
            # Remove best_v and its neighbors from C
            C &= ~(adj[best_v] | (1 << best_v))
            
        return S

    initial_mask = greedy_mis()
    best_size = initial_mask.bit_count()
    best_mask = initial_mask
    
    # 3. Branch and Bound with Greedy Coloring Bound
    
    # Helper to compute a greedy coloring bound for the induced subgraph on C
    # Returns the number of colors used (upper bound on MIS size)
    def coloring_bound(C):
        if C == 0:
            return 0
        
        # Extract vertices and sort by degree in G[C] descending
        # To speed up, we can just iterate and assign colors.
        # A simple greedy coloring:
        # 1. Get list of vertices in C
        # 2. Sort by degree in G[C] descending
        # 3. Assign colors
        
        vertices = []
        c_temp = C
        while c_temp:
            lsb = c_temp & -c_temp
            v = lsb.bit_length() - 1
            vertices.append(v)
            c_temp ^= lsb
            
        # Sort by degree in G[C]
        # Degree of v in G[C] is (adj[v] & C).bit_count()
        # This is O(K * (N/word_size)) roughly.
        # For N=256, this is fast enough if done carefully.
        
        # Optimization: Instead of sorting fully, we can just pick the highest degree one first?
        # Standard Tomita algorithm uses a specific ordering.
        # Let's use a simple greedy:
        # colors = {}
        # color_classes = []
        
        # Actually, a faster bound is often just the number of colors in a greedy coloring
        # where vertices are processed in a specific order.
        
        # Let's try a simple approach:
        # Process vertices in order of decreasing degree in G[C].
        
        # To avoid O(N log N) sort every time, we can approximate.
        # But N=256 is small.
        
        # Let's refine the coloring to be very fast.
        # We will maintain an array of available colors for each vertex? No, that's complex.
        
        # Simple Greedy Coloring:
        # 1. Order vertices by degree descending.
        # 2. For each vertex, assign the smallest color not used by its colored neighbors.
        
        # Since we just need the *count* of colors (upper bound), we can be a bit lazy.
        # A valid coloring uses at most Delta+1 colors.
        # But we want a tight bound.
        
        # Let's implement a standard greedy coloring.
        
        # 1. Get vertices
        verts = []
        c_temp = C
        while c_temp:
            lsb = c_temp & -c_temp
            v = lsb.bit_length() - 1
            verts.append(v)
            c_temp ^= lsb
            
        # 2. Sort by degree in G[C] descending
        # We can calculate degrees once.
        degs = [( (adj[v] & C).bit_count(), v ) for v in verts]
        degs.sort(key=lambda x: -x[0])
        
        color = {}
        max_color = 0
        for deg, v in degs:
            # Find smallest color not used by neighbors in C
            used = set()
            neighbors = adj[v] & C
            # Iterate neighbors
            n_temp = neighbors
            while n_temp:
                lsb = n_temp & -n_temp
                u = lsb.bit_length() - 1
                if u in color:
                    used.add(color[u])
                n_temp ^= lsb
            
            # Find smallest available color
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1

    # Recursive B&B
    # To avoid recursion depth issues and overhead, we can use a stack or just recursion with sys.setrecursionlimit
    # Depth is max 256.
    
    def expand(C, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return False
        
        # Pruning with coloring bound
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return True
            
        bound = coloring_bound(C)
        
        if current_size + bound <= best_size:
            return True
        
        # Select vertex to branch on
        # Heuristic: Pick vertex with max degree in G[C]
        # Or pick the first vertex from the coloring order?
        # Let's pick max degree.
        
        max_deg = -1
        best_v = -1
        c_temp = C
        while c_temp:
            lsb = c_temp & -c_temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & C).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            c_temp ^= lsb
        
        if best_v == -1:
            return True
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_C_include = C & ~(adj[best_v] | v_bit)
        new_mask = current_mask | v_bit
        
        expand(new_C_include, current_size + 1, new_mask)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude best_v
        new_C_exclude = C & ~v_bit
        expand(new_C_exclude, current_size, current_mask)
        
        return True

    # Start B&B
    all_C = (1 << N) - 1
    
    # Run B&B
    expand(all_C, 0, 0)
    
    # Convert best_mask to list of numbers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
