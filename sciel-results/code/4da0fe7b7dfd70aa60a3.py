import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Bitmasks for neighbors
    # neighbors[v] is a bitmask where bit j is 1 if v and j conflict
    # Note: v itself is not a neighbor of v in the standard graph sense, 
    # but the constraint a+b not square for a<=b includes a=b.
    # So if 2*v is a square, v cannot be in the set.
    # We can handle this by just not allowing such vertices to be in the set.
    # Or, we can treat self-conflict as "v is not a valid node".
    
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    if not valid_nodes:
        return [1] # Fallback, though 1 is valid (2*1=2 not square)
        
    M = len(valid_nodes)
    # Map original value to index in valid_nodes
    # But we can just work with the original values 1..256 and use bit positions 1..256?
    # Bit 0 corresponds to value 1? Or bit v-1 corresponds to value v?
    # Let's use bit position v-1 for value v.
    
    neighbors = [0] * (N + 1) # 1-indexed values, bit i-1 set if i is neighbor
    
    for u in valid_nodes:
        for v in valid_nodes:
            if u < v and (u + v) in squares:
                # u conflicts with v
                neighbors[u] |= (1 << (v - 1))
                neighbors[v] |= (1 << (u - 1))
                
    # Initial candidate set: all valid nodes
    # Bit i-1 is 1 if node i is in candidates
    candidates = 0
    for v in valid_nodes:
        candidates |= (1 << (v - 1))
        
    best_size = 0
    best_mask = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy initial solution
    def greedy_mis(cands):
        sol = 0
        curr = cands
        while curr:
            # Pick a node with lowest degree in curr? Or just first one?
            # Let's pick the one with lowest degree to keep options open?
            # Actually, for a quick lower bound, any greedy works.
            # Let's pick the first available node.
            # Find lowest set bit
            v_bit = curr & -curr
            v_idx = v_bit.bit_length() - 1
            v_val = v_idx + 1
            sol |= v_bit
            # Remove v and its neighbors
            curr &= ~v_bit
            curr &= ~neighbors[v_val]
        return sol, popcount(sol)
        
    init_mask, init_size = greedy_mis(candidates)
    best_size = init_size
    best_mask = init_mask
    
    # Branch and Bound
    # To speed up, we can sort the order of vertices we consider.
    # But with bitmasks, we just iterate.
    
    # We need to be careful with recursion depth or use a stack.
    # N=256, depth can be 256. Python recursion limit is usually 1000, so it's fine.
    # But we need to check time.
    
    def search(cands, curr_size):
        nonlocal best_size, best_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if curr_size + popcount(cands) <= best_size:
            return
            
        if cands == 0:
            if curr_size > best_size:
                best_size = curr_size
                best_mask = 0 # We need to reconstruct? 
                # Actually, we should track the mask in the recursion or pass it down.
                # Passing mask down is expensive (big ints). 
                # Alternative: store the path.
                # Or, just return the size and reconstruct later?
                # Reconstructing from size is hard.
                # Let's pass the current mask down.
            return
            
        # Pick a vertex v from cands
        # Heuristic: pick v with highest degree in cands?
        # Let's just pick the first one for simplicity, or iterate to find best.
        # Finding best:
        best_v_bit = 0
        max_deg = -1
        temp = cands
        while temp:
            v_bit = temp & -temp
            v_idx = v_bit.bit_length() - 1
            v_val = v_idx + 1
            deg = popcount(neighbors[v_val] & cands)
            if deg > max_deg:
                max_deg = deg
                best_v_bit = v_bit
            temp &= temp - 1
            
        if best_v_bit == 0:
            return
            
        v_idx = best_v_bit.bit_length() - 1
        v_val = v_idx + 1
        
        # Branch 1: Include v
        # New candidates: cands without v and without neighbors of v
        new_cands_include = cands & ~best_v_bit & ~neighbors[v_val]
        search(new_cands_include, curr_size + 1)
        
        # Check time again
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_cands_exclude = cands & ~best_v_bit
        search(new_cands_exclude, curr_size)
        
    # To track the mask, we can modify search to take current_mask
    def search_with_mask(cands, curr_size, curr_mask):
        nonlocal best_size, best_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if curr_size + popcount(cands) <= best_size:
            return
            
        if cands == 0:
            if curr_size > best_size:
                best_size = curr_size
                best_mask = curr_mask
            return
            
        # Pick a vertex v from cands
        best_v_bit = 0
        max_deg = -1
        temp = cands
        while temp:
            v_bit = temp & -temp
            v_idx = v_bit.bit_length() - 1
            v_val = v_idx + 1
            deg = popcount(neighbors[v_val] & cands)
            if deg > max_deg:
                max_deg = deg
                best_v_bit = v_bit
            temp &= temp - 1
            
        if best_v_bit == 0:
            return
            
        v_idx = best_v_bit.bit_length() - 1
        v_val = v_idx + 1
        
        # Branch 1: Include v
        new_cands_include = cands & ~best_v_bit & ~neighbors[v_val]
        new_mask_include = curr_mask | best_v_bit
        search_with_mask(new_cands_include, curr_size + 1, new_mask_include)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_cands_exclude = cands & ~best_v_bit
        search_with_mask(new_cands_exclude, curr_size, curr_mask)

    search_with_mask(candidates, 0, 0)
    
    # Extract result
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
