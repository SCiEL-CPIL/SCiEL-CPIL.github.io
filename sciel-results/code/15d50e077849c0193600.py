import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to be safe within 60s
    
    N = 256
    # Vertices are 1..256, index 0..255
    
    # 1. Build Conflict Graph
    # conflict[i] is a bitmask of vertices that conflict with i
    conflict = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # Helper to get degree
    def get_degree(mask):
        return bin(mask).count('1')
        
    # 2. Greedy Initial Solution
    # Sort vertices by degree ascending (Min-Degree Heuristic)
    vertices = list(range(N))
    # We need dynamic degrees or just static? Static is faster.
    # Let's use static degree for initial greedy.
    degrees = [get_degree(conflict[i]) for i in range(N)]
    vertices.sort(key=lambda x: degrees[x])
    
    current_set_mask = 0
    current_set_list = []
    
    for v in vertices:
        # Check if v conflicts with any in current_set
        if (conflict[v] & current_set_mask) == 0:
            current_set_mask |= (1 << v)
            current_set_list.append(v)
            
    best_mask = current_set_mask
    best_size = len(current_set_list)
    
    # 3. Local Search / Polish
    # We try to add more vertices or swap.
    # State: current_set_mask
    # Pool: ~current_set_mask & ((1 << N) - 1)
    
    all_mask = (1 << N) - 1
    
    def can_add(v, set_mask):
        return (conflict[v] & set_mask) == 0
        
    def try_add_greedy(set_mask):
        # Try to add any vertex from pool that doesn't conflict
        # We want to add the one that leaves the most options? 
        # Or just pick one.
        pool_mask = all_mask & ~set_mask
        added = False
        # Find a vertex in pool that can be added
        # To be fast, we can iterate through set bits
        while pool_mask:
            # Get lowest set bit
            lsb = pool_mask & -pool_mask
            v = lsb.bit_length() - 1
            if can_add(v, set_mask):
                set_mask |= lsb
                added = True
                # Update pool? We don't need to explicitly remove v from pool if we just track set_mask
                # But we need to check other vertices against the new set_mask
                # So we restart the scan or continue?
                # If we continue, we might add more.
                # Let's just add one and restart the loop to be safe/simple
                break
            else:
                pool_mask &= pool_mask - 1 # Clear bit
        return set_mask, added

    # Main loop
    while time.time() < deadline:
        # Try to add vertices
        new_mask, added = try_add_greedy(current_set_mask)
        if added:
            current_set_mask = new_mask
            if bin(current_set_mask).count('1') > best_size:
                best_size = bin(current_set_mask).count('1')
                best_mask = current_set_mask
        else:
            # Stuck. Try a swap or random restart
            # Swap: Pick u in set, v in pool.
            # Condition: v can be added if u is removed.
            # i.e., (conflict[v] & (current_set_mask ^ (1<<u))) == 0
            # And we want this to potentially allow adding more.
            
            # Let's try a random swap
            set_list = [i for i in range(N) if (current_set_mask >> i) & 1]
            pool_list = [i for i in range(N) if not (current_set_mask >> i) & 1]
            
            if not set_list or not pool_list:
                break
                
            # Try a few random swaps
            improved = False
            for _ in range(10):
                u = random.choice(set_list)
                v = random.choice(pool_list)
                
                # Remove u, add v
                temp_mask = current_set_mask
                temp_mask &= ~(1 << u)
                temp_mask |= (1 << v)
                
                # Check if v is valid in temp_mask
                if (conflict[v] & temp_mask) != 0:
                    continue
                    
                # Try to add more from greedy
                new_mask, added = try_add_greedy(temp_mask)
                new_size = bin(new_mask).count('1')
                
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                    current_set_mask = new_mask
                    improved = True
                    break
                elif new_size > bin(current_set_mask).count('1'):
                    # Accept if it improves current state (Hill Climbing)
                    current_set_mask = new_mask
                    improved = True
                    break
            if not improved:
                # Random restart or jump
                # Maybe remove a random vertex to unstick
                if set_list:
                    u = random.choice(set_list)
                    current_set_mask &= ~(1 << u)
                
    # Convert best_mask to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return result
