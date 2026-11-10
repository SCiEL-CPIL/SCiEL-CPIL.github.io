import time
import random

def construct():
    n = 512
    start_time = time.time()
    deadline = 55.0
    
    def can_add(x, mask):
        # 1. x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2*d)) & 1) and ((mask >> (x + 3*d)) & 1):
                return False
        
        # 2. x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1) and ((mask >> (x + 2*d)) & 1):
                return False
        
        # 3. x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if ((mask >> (x - 2*d)) & 1) and ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                return False
        
        # 4. x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if ((mask >> (x - 3*d)) & 1) and ((mask >> (x - 2*d)) & 1) and ((mask >> (x - d)) & 1):
                return False
        
        return True

    best_mask = 0
    best_size = 0
    
    # Phase 1: Multiple Greedy
    # Estimate cost of one greedy run: N * avg(can_add)
    # can_add is ~N/3 * 3 checks. ~500 ops.
    # 512 * 500 = 256,000 ops.
    # 100 runs = 25M ops. ~1-2 seconds in Python.
    # We can do 50-100 runs.
    
    runs = 0
    while time.time() < deadline - 10.0:
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        current_mask = 0
        current_size = 0
        
        for x in elements:
            if can_add(x, current_mask):
                current_mask |= (1 << x)
                current_size += 1
                
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            runs += 1
            
        # If we found a very good set, we can stop greedy earlier?
        # Max possible is unknown, but likely < 200.
        # If best_size > 150, we are doing well.
        
    # Phase 2: Local Search on best_mask
    # Use random swaps
    # We will work on a copy
    current_mask = best_mask
    current_size = best_size
    
    # To speed up swaps, we can precompute list of elements in S and not in S
    # But S changes.
    
    while time.time() < deadline - 1.0:
        # 1. Try to add any element
        # This is O(N * cost(can_add))
        # 512 * 500 = 256k ops. Fast.
        added = False
        for x in range(1, n + 1):
            if not (current_mask & (1 << x)):
                if can_add(x, current_mask):
                    current_mask |= (1 << x)
                    current_size += 1
                    added = True
                    break # Restart
            
        if added:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            continue
            
        # 2. Try random swaps
        # Pick random u in S, random v not in S
        # Check if v fits in S \ {u}
        
        # Get list of elements in S
        # Doing this every time is O(N).
        # We can maintain a list, but it's error prone.
        # N=512 is small.
        
        s_list = [i for i in range(1, n + 1) if current_mask & (1 << i)]
        not_s_list = [i for i in range(1, n + 1) if not (current_mask & (1 << i))]
        
        if not s_list or not not_s_list:
            break
            
        # Try a few random swaps
        swapped = False
        for _ in range(50): # 50 attempts per iteration
            u = random.choice(s_list)
            v = random.choice(not_s_list)
            
            temp_mask = current_mask & ~(1 << u)
            if can_add(v, temp_mask):
                current_mask = temp_mask | (1 << v)
                swapped = True
                break
                
        if swapped:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            # Update lists? No, we recreate them next loop.
            # But we should restart the "try to add" phase
            continue
            
        # If no swap found in 50 tries, maybe try more or break?
        # Let's try 100 more
        for _ in range(100):
            u = random.choice(s_list)
            v = random.choice(not_s_list)
            temp_mask = current_mask & ~(1 << u)
            if can_add(v, temp_mask):
                current_mask = temp_mask | (1 << v)
                swapped = True
                break
                
        if swapped:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            continue
        else:
            # Stuck? Break local search for this mask
            break

    # Convert best_mask to list
    result = []
    for x in range(1, n + 1):
        if best_mask & (1 << x):
            result.append(x)
            
    return result
