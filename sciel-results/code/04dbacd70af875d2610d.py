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
        S = set()
        while C:
            # Find vertex in C with min degree in G[C]
            min_deg = float('inf')
            best_v = -1
            # Iterate through set bits in C
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
                
            S.add(best_v)
            # Remove best_v and its neighbors from C
            C &= ~(adj[best_v] | (1 << best_v))
            
        return S

    initial_set = greedy_mis()
    best_size = len(initial_set)
    best_set = initial_set
    
    # 3. Branch and Bound
    # To speed up, we can use a recursive function with bitmasks
    
    # Helper to get degree of v in C
    def get_degree(v, C):
        return (adj[v] & C).bit_count()

    # We will use a list to store the path or just update global best
    # Since N=256, recursion depth is at most 256.
    
    # Optimization: Order vertices for branching.
    # We pick a vertex, branch on include, then exclude.
    # To prune, we use the bound: current_size + popcount(C) <= best_size
    
    def expand(C, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return False # Signal timeout
        
        # Pruning
        if current_size + C.bit_count() <= best_size:
            return True
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return True
        
        # Heuristic: Select vertex v from C
        # Strategy: Pick v with min degree in G[C] to include first?
        # Or pick v with max degree to prune faster?
        # For MIS, including a low-degree vertex leaves more options.
        # Let's pick min degree vertex.
        
        min_deg = float('inf')
        best_v = -1
        c_temp = C
        while c_temp:
            lsb = c_temp & -c_temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & C).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            c_temp ^= lsb
        
        if best_v == -1:
            return True
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New C is C minus best_v and its neighbors
        new_C_include = C & ~(adj[best_v] | v_bit)
        new_set = current_set | {best_v}
        
        # Check time before deep recursion?
        if time.time() > deadline:
            return False
            
        expand(new_C_include, current_size + 1, new_set)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude best_v
        # New C is C minus best_v
        new_C_exclude = C & ~v_bit
        
        expand(new_C_exclude, current_size, current_set)
        
        return True

    # Start B&B
    # We can order the initial vertices? The recursive selection handles it.
    # However, starting with a good order helps.
    # The greedy solution is already a good lower bound.
    
    # One optimization: If the graph is very sparse, the greedy might be optimal.
    # Let's run B&B.
    
    # To avoid hitting recursion limit or overhead, we can implement iterative or just rely on Python's recursion limit (default 1000, depth 256 is fine).
    
    # Initial C is all vertices
    all_C = (1 << N) - 1
    
    # We need to be careful with the set copying. Using a set for current_set is O(N) per step.
    # Since N=256, it's fine.
    
    # However, passing a set object is expensive.
    # We can use a bitmask for the current set too?
    # But we need to return the set at the end.
    # Let's stick to set for simplicity, as depth is small.
    
    # Actually, let's optimize the B&B by using a bitmask for the current set as well,
    # and only convert to list at the end if it's a new best.
    
    best_set_mask = 0
    # Convert initial_set to mask
    for v in initial_set:
        best_set_mask |= (1 << v)
        
    def expand_mask(C, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return False
        
        if current_size + C.bit_count() <= best_size:
            return True
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set_mask
            return True
        
        # Select vertex
        min_deg = float('inf')
        best_v = -1
        c_temp = C
        while c_temp:
            lsb = c_temp & -c_temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & C).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            c_temp ^= lsb
        
        if best_v == -1:
            return True
            
        v_bit = 1 << best_v
        
        # Include
        new_C_include = C & ~(adj[best_v] | v_bit)
        new_mask = current_set_mask | v_bit
        
        expand_mask(new_C_include, current_size + 1, new_mask)
        
        if time.time() > deadline:
            return False
            
        # Exclude
        new_C_exclude = C & ~v_bit
        expand_mask(new_C_exclude, current_size, current_set_mask)
        
        return True

    expand_mask(all_C, 0, 0)
    
    # Convert best_set_mask to list of numbers
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
