import time
import random
import math
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
    
    # For fast access to other elements in an AP given one element
    # ap_other_elements[ap_idx][x] = list of other 3 elements
    # But simpler: just use the tuple and filter
    # We can precompute a dict: ap_idx -> {elem: other_3_list}
    ap_neighbors = [{} for _ in range(num_aps)]
    for idx, ap in enumerate(all_aps):
        a, b, c, d = ap
        ap_neighbors[idx][a] = [b, c, d]
        ap_neighbors[idx][b] = [a, c, d]
        ap_neighbors[idx][c] = [a, b, d]
        ap_neighbors[idx][d] = [a, b, c]

    # State
    S = set()
    conflict_count = [0] * (N + 1)
    
    def add_element(x):
        """Add x to S, update conflict counts. Assumes conflict_count[x] == 0"""
        if x in S:
            return
        S.add(x)
        for ap_idx in ap_indices_by_num[x]:
            others = ap_neighbors[ap_idx][x]
            # Check if the other 3 are in S? No, we need to update counts for the others.
            # For each other element z in others:
            # The AP is "almost complete" for z if the other 2 (besides z and x) are in S.
            # Wait, the logic is:
            # conflict_count[z] counts the number of APs containing z where the other 3 are in S.
            # If we add x, we affect the counts of neighbors z.
            # For an AP P containing x and z:
            # Before adding x: P had 3 elements in S (including z? No, z is not in S necessarily).
            # Let's re-evaluate.
            # conflict_count[z] = # of APs P s.t. z in P and (P \ {z}) subset S.
            # If we add x to S:
            # For any AP P containing x:
            #   For any z in P \ {x}:
            #   The set P \ {z} now includes x.
            #   If (P \ {z, x}) subset S, then P \ {z} is subset S (since x is now in S).
            #   So if the other 2 elements (besides z and x) were already in S, then conflict_count[z] increases by 1.
            
            for z in others:
                # The other 2 elements in the AP besides x and z
                # P = {x, z, u, v}
                # We need to check if u and v are in S.
                # We can get u, v by removing x and z from the AP tuple.
                # It's faster to precompute pairs.
                pass # Placeholder, we will implement this more efficiently below.

    # Let's precompute for each (ap_idx, x) the list of the other 3 elements.
    # And for each (ap_idx, x, z) the remaining 2 elements.
    # This might be too much memory/time.
    # Alternative:
    # In add_element(x):
    #   For each ap_idx in ap_indices_by_num[x]:
    #     ap = all_aps[ap_idx]
    #     others = [y for y in ap if y != x]
    #     for z in others:
    #       # remaining = [y for y in others if y != z]
    #       remaining = [y for y in others if y != z]
    #       if remaining[0] in S and remaining[1] in S:
    #         conflict_count[z] += 1

    # This is O(degree * 3 * 2) which is fine.

    def try_add(x):
        if conflict_count[x] == 0 and x not in S:
            add_element(x)
            return True
        return False

    def add_element(x):
        S.add(x)
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            others = [y for y in ap if y != x]
            for z in others:
                # Find the other two elements
                # Since ap is a tuple of 4, and we have x and z, the others are the remaining 2
                # We can just check membership of the other 2
                # To avoid list comprehension overhead, we can do:
                # The AP is (a,b,c,d). 
                # If x=a, z=b, then others are c,d.
                # Let's just use a helper or inline logic.
                # For speed, we precomputed ap_neighbors[ap_idx][x] = [b,c,d]
                # We need the pair excluding z.
                # Let's just iterate.
                pass # Implemented in the full loop below

    # Re-implementing add_element for clarity and speed
    def add_element_fast(x):
        S.add(x)
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            # Get the 3 other elements
            o1, o2, o3 = ap[0], ap[1], ap[2], ap[3]
            # This is inefficient. Let's use the precomputed neighbors.
            others = ap_neighbors[ap_idx][x]
            for z in others:
                # The remaining two elements are the ones in 'others' that are not z
                # Since 'others' has 3 elements, we can just check the other two.
                # To avoid creating lists, we can do:
                # If z == others[0], check others[1], others[2]
                # etc.
                # Or just:
                rem = [y for y in others if y != z]
                if rem[0] in S and rem[1] in S:
                    conflict_count[z] += 1

    def remove_element_fast(x):
        S.remove(x)
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            others = ap_neighbors[ap_idx][x]
            for z in others:
                rem = [y for y in others if y != z]
                if rem[0] in S and rem[1] in S:
                    conflict_count[z] -= 1

    def greedy_fill(order):
        for x in order:
            if conflict_count[x] == 0:
                add_element_fast(x)

    # Initial Set: Base-4 {0,1} set
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
    
    # Add base4_set elements first
    for x in base4_set:
        if conflict_count[x] == 0:
            add_element_fast(x)
            
    # Fill remaining with random greedy
    remaining = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(remaining)
    greedy_fill(remaining)
    
    best_set = set(S)
    best_size = len(S)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    rng = random.Random(42)
    
    while time.time() < deadline:
        if T < 0.01:
            T = 0.01 # Keep exploring a bit or break
            # Actually, let's just break or restart with lower T
            break
        
        # Create a copy of current state for the move
        # Since we need to rollback, it's better to simulate the move and then rollback
        # Or just store the changes.
        
        # Move: Remove 1 random element from S, then greedy fill
        if not S:
            break
            
        x_remove = rng.choice(list(S))
        
        # Record state to rollback
        # We remove x_remove
        remove_element_fast(x_remove)
        
        # Try to add elements
        added = []
        candidates = [x for x in range(1, N + 1) if x not in S]
        rng.shuffle(candidates)
        
        for x in candidates:
            if conflict_count[x] == 0:
                add_element_fast(x)
                added.append(x)
        
        new_size = len(S)
        delta = new_size - best_size # This is not right, delta should be vs current_size before move
        # Wait, I need to compare new_size with the size BEFORE the move.
        # Let's restructure.
        
        # This approach is messy with rollback.
        # Better:
        # 1. Store current_size
        # 2. Perform move (remove 1, add many)
        # 3. Calculate delta = new_size - current_size
        # 4. Accept or Reject
        #    If Reject, rollback.
        
        # Let's implement a cleaner SA loop.
        
        current_size = len(S)
        
        # 1. Remove 1 random element
        x_remove = rng.choice(list(S))
        remove_element_fast(x_remove)
        
        # 2. Greedy add
        added = []
        candidates = [x for x in range(1, N + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if conflict_count[x] == 0:
                add_element_fast(x)
                added.append(x)
        
        new_size = len(S)
        delta = new_size - current_size
        
        # 3. Accept/Reject
        if delta > 0 or (T > 0.01 and rng.random() < math.exp(delta / T)):
            # Accept
            if new_size > best_size:
                best_size = new_size
                best_set = set(S)
        else:
            # Reject: Rollback
            # Remove added elements
            for x in added:
                remove_element_fast(x)
            # Add back removed element
            add_element_fast(x_remove)
            
        T *= cooling

    return sorted(best_set)

# Note: The code above has a flaw in the SA loop:
# "candidates = [x for x in range(1, N + 1) if x not in S]" is O(N) every iteration.
# With 512 elements, it's okay, but we can optimize.
# Also, the "remove 1, add many" move is very biased towards growing the set.
# It might get stuck. A "swap" move (remove 1, add 1) or "remove 2, add 2" might be better.
# However, "remove 1, add many" is a standard "repair" move that often works well for packing problems.
# Given the time limit, this should be sufficient to beat the seed.

# Let's refine the code to be robust and efficient.

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    ap_indices_by_num = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
            
    ap_neighbors = [{} for _ in range(num_aps)]
    for idx, ap in enumerate(all_aps):
        a, b, c, d = ap
        ap_neighbors[idx][a] = [b, c, d]
        ap_neighbors[idx][b] = [a, c, d]
        ap_neighbors[idx][c] = [a, b, d]
        ap_neighbors[idx][d] = [a, b, c]

    S = set()
    conflict_count = [0] * (N + 1)
    
    def add_element(x):
        S.add(x)
        for ap_idx in ap_indices_by_num[x]:
            others = ap_neighbors[ap_idx][x]
            # others is a list of 3 elements
            # For each z in others, check if the other 2 are in S
            o1, o2, o3 = others
            # z = o1: check o2, o3
            if o2 in S and o3 in S:
                conflict_count[o1] += 1
            # z = o2: check o1, o3
            if o1 in S and o3 in S:
                conflict_count[o2] += 1
            # z = o3: check o1, o2
            if o1 in S and o2 in S:
                conflict_count[o3] += 1

    def remove_element(x):
        S.remove(x)
        for ap_idx in ap_indices_by_num[x]:
            others = ap_neighbors[ap_idx][x]
            o1, o2, o3 = others
            if o2 in S and o3 in S:
                conflict_count[o1] -= 1
            if o1 in S and o3 in S:
                conflict_count[o2] -= 1
            if o1 in S and o2 in S:
                conflict_count[o3] -= 1

    # Initial Set
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
            
    for x in base4_set:
        if conflict_count[x] == 0:
            add_element(x)
            
    remaining = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(remaining)
    for x in remaining:
        if conflict_count[x] == 0:
            add_element(x)

    best_set = set(S)
    best_size = len(S)
    
    rng = random.Random(42)
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        current_size = len(S)
        
        # Move: Remove 1 random element, then greedy fill
        if not S:
            break
            
        x_remove = rng.choice(list(S))
        remove_element(x_remove)
        
        added = []
        # To speed up, we can just iterate through all elements not in S
        # But checking "not in S" is O(1).
        # We can maintain a list of "available" elements? 
        # No, just iterate 1..N is fine for N=512.
        
        for x in range(1, N + 1):
            if x not in S and conflict_count[x] == 0:
                add_element(x)
                added.append(x)
                
        new_size = len(S)
        delta = new_size - current_size
        
        if delta > 0 or (T > 0.01 and rng.random() < math.exp(delta / T)):
            if new_size > best_size:
                best_size = new_size
                best_set = set(S)
        else:
            for x in added:
                remove_element(x)
            add_element(x_remove)
            
        T *= cooling

    return sorted(best_set)
