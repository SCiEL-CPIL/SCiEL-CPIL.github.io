# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a maximum subset of Z/205Z such that no two elements differ by a 
    non-zero quadratic residue modulo 205.
    
    Uses a branch-and-bound algorithm with bitset representations for efficiency.
    """
    m = 205
    k = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    start_time = time.time()
    deadline = start_time + 600

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # 2. Build conflict graph as bitmasks
    # neighbors[i] is a bitmask where bit j is 1 if i and j conflict
    neighbors = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden:
                    mask |= (1 << j)
        neighbors[i] = mask

    # Helper to get the list of indices set in a bitmask
    def bits_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # 3. Branch and Bound
    # State: current_mask (chosen vertices), cand_mask (vertices compatible with current_mask)
    best_size = 0
    best_mask = 0
    global_start_time = time.time()
    
    # Initial candidate mask is all vertices
    initial_cand_mask = (1 << m) - 1
    
    # Heuristic: Greedy initialization to get a good lower bound quickly
    # Sort vertices by degree (number of conflicts) ascending? 
    # Actually, for independent set, picking low-degree vertices first in greedy is often good.
    degrees = [bin(neighbors[i]).count('1') for i in range(m)]
    sorted_vertices = sorted(range(m), key=lambda x: degrees[x])
    
    greedy_mask = 0
    greedy_cand = initial_cand_mask
    for v in sorted_vertices:
        if (greedy_cand >> v) & 1:
            greedy_mask |= (1 << v)
            greedy_cand &= ~neighbors[v]
            greedy_cand &= ~(1 << v)
            
    best_size = bin(greedy_mask).count('1')
    best_mask = greedy_mask

    # B&B Stack: (current_mask, cand_mask)
    # We use a stack for DFS
    stack = [(0, initial_cand_mask)]
    
    # To speed up, we can use a simple recursion or iterative DFS. 
    # Given the depth (max 205), recursion limit might need adjustment, but 205 is small.
    # However, iterative is safer.
    
    # We want to explore branches that lead to larger sets first.
    # Branching on a vertex v:
    # 1. Include v: new_cand = cand & ~neighbors[v] & ~(1<<v)
    # 2. Exclude v: new_cand = cand & ~(1<<v)
    
    # Ordering strategy: Pick a vertex from cand_mask.
    # A common heuristic for Max Independent Set is to pick the vertex with the 
    # highest degree in the induced subgraph of cand_mask.
    # Let's implement a function to pick the best vertex to branch on.
    
    def pick_vertex(cand_mask):
        if cand_mask == 0:
            return -1
        # Find vertex in cand_mask with max degree within cand_mask
        best_v = -1
        max_deg = -1
        # Iterate through set bits
        temp = cand_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Count neighbors of v within cand_mask
            deg = bin(neighbors[v] & cand_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp &= temp - 1 # Remove lowest set bit
        return best_v

    while stack:
        if time.time() > deadline:
            break
            
        current_mask, cand_mask = stack.pop()
        
        # Upper bound: current size + remaining candidates
        curr_size = bin(current_mask).count('1')
        cand_count = bin(cand_mask).count('1')
        
        if curr_size + cand_count <= best_size:
            continue
        
        if cand_mask == 0:
            if curr_size > best_size:
                best_size = curr_size
                best_mask = current_mask
            continue
        
        # Pick a vertex to branch on
        v = pick_vertex(cand_mask)
        if v == -1:
            continue
            
        bit_v = 1 << v
        
        # Branch 1: Include v
        # This is more promising for finding large sets, so push Exclude branch first 
        # so Include is processed next (LIFO)
        
        # Exclude v
        new_cand_excl = cand_mask & ~bit_v
        stack.append((current_mask, new_cand_excl))
        
        # Include v
        new_cand_incl = cand_mask & ~neighbors[v] & ~bit_v
        new_curr_mask = current_mask | bit_v
        stack.append((new_curr_mask, new_cand_incl))
        
        # Optimization: If we found a solution equal to the theoretical max 
        # or if we are running out of time, we can stop. 
        # Theoretical max is m, but we don't know it.
        
    # Extract result
    result_list = bits_to_list(best_mask)
    
    # Fallback if something went wrong (shouldn't happen)
    if not result_list:
        result_list = [0]
        
    return np.array(sorted(result_list), dtype=np.int64)


# EVOLVE-BLOCK-END
