import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    
    # Precompute 4-APs
    # Store as list of tuples of 4 indices
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Map element to list of 4-AP indices it belongs to
    # 4-AP index is just the index in all_4aps
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_4aps[x].append(idx)
            
    # Precompute 3-APs
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            all_3aps.append((a, a + d, a + 2 * d))
            
    # Map element to list of 3-AP indices it belongs to
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(idx)

    # State
    # A is a boolean numpy array of size N+1
    A = np.zeros(N + 1, dtype=bool)
    current_3ap_count = 0
    
    # Helper: Check if adding x keeps set 4-AP free
    # Assumes x is NOT in A
    def is_valid_add(x):
        # Check all 4-APs containing x
        for idx in elem_to_4aps[x]:
            ap = all_4aps[idx]
            # Check if the other 3 elements are in A
            # If yes, adding x creates a 4-AP
            # We can break early
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
                    if cnt >= 3:
                        return False
        return True
        
    # Helper: Check if swapping x (out) and y (in) keeps set 4-AP free
    # Assumes x IS in A, y is NOT in A
    def is_valid_swap(x, y):
        # New set A' = A \ {x} \cup {y}
        # We need to ensure no 4-AP is fully contained in A'
        # Any 4-AP in A' must contain y (since A was 4-AP free, any 4-AP in A' not containing y would have been in A)
        # So we only check 4-APs containing y
        for idx in elem_to_4aps[y]:
            ap = all_4aps[idx]
            # Check if the other 3 elements are in A'
            # A' contains y, and A \ {x}
            # So the other 3 elements must be in A \ {x}
            cnt = 0
            for z in ap:
                if z != y:
                    # z must be in A and z != x
                    if A[z] and z != x:
                        cnt += 1
                        if cnt >= 3:
                            return False
        return True

    # Helper: Calculate gain in 3-AP count if we add x to A
    # Assumes x NOT in A
    def gain_add(x):
        gain = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            # Check if other 2 elements are in A
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
            if cnt == 2:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-AP count if we remove x from A
    # Assumes x IS in A
    def loss_remove(x):
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            # Check if other 2 elements are in A
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
            if cnt == 2:
                loss += 1
        return loss

    # Helper: Calculate change in 3-AP count if we swap x (out) and y (in)
    # Assumes x IS in A, y is NOT in A
    def delta_swap(x, y):
        # Remove x: lose 3-APs containing x and 2 others in A
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != x and A[z]:
                    cnt += 1
            if cnt == 2:
                loss += 1
                
        # Add y: gain 3-APs containing y and 2 others in A \ {x}
        gain = 0
        for idx in elem_to_3aps[y]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != y and A[z] and z != x:
                    cnt += 1
            if cnt == 2:
                gain += 1
                
        return gain - loss

    # Greedy Initialization
    remaining = set(range(1, N + 1))
    
    # We want to maximize 3-APs.
    # Greedy: pick element with max gain_add, subject to is_valid_add
    while remaining:
        best_x = None
        best_gain = -1
        # To speed up, we can sample if remaining is large, but N=256 is small
        # However, checking all is O(N * degree). 
        # Let's just iterate.
        for x in remaining:
            if is_valid_add(x):
                g = gain_add(x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        
        if best_x is None:
            break
            
        A[best_x] = True
        current_3ap_count += best_gain
        remaining.remove(best_x)
        
    best_set = A.copy()
    best_score = current_3ap_count
    
    # Simulated Annealing
    # T_start = 10.0, T_end = 0.1
    # Cooling rate: T = T_start * (T_end/T_start)^t
    # Or T = T_start * exp(-alpha * t)
    
    T = 1.0
    cooling = 0.9999 # Adjust based on steps
    
    # Precompute list of elements for random choice
    # We need to pick from A and not-A frequently.
    # Maintaining lists is O(N) to update, which is fine.
    
    # Initial T should be high enough to accept bad moves
    # Max possible delta is small (degree of 3-APs ~ 100)
    # Let's start T=5.0
    
    T = 5.0
    step = 0
    total_steps = 1000000 # Just a large number, time limited
    
    while time.time() < deadline:
        step += 1
        # Choose move type
        # 1. Add
        # 2. Remove
        # 3. Swap
        
        # Pick random move
        r = random.random()
        
        if r < 0.3:
            # Try Add
            # Pick random x not in A
            # Efficient way: pick random from remaining? 
            # Or just pick random 1..N and check if not in A.
            # N=256, so random pick is fast.
            x = random.randint(1, N)
            if not A[x]:
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A[x] = True
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set = A.copy()
                else:
                    # Invalid move, reject
                    pass
        elif r < 0.6:
            # Try Remove
            # Pick random x in A
            # We can maintain a list of active elements for O(1) pick
            # But N is small.
            # Let's just pick random 1..N and check if in A.
            x = random.randint(1, N)
            if A[x]:
                delta = -loss_remove(x)
                # Removing always valid 4-AP wise
                if delta >= 0 or random.random() < np.exp(delta / T):
                    A[x] = False
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set = A.copy()
        else:
            # Try Swap
            # Pick x in A, y not in A
            x = random.randint(1, N)
            y = random.randint(1, N)
            if A[x] and not A[y]:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A[x] = False
                        A[y] = True
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set = A.copy()
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best set found
    # best_set is a numpy array
    # Convert to sorted list of ints
    res = [i for i in range(1, N + 1) if best_set[i]]
    
    if not res:
        # Fallback: return a small valid set
        # e.g. {1, 2, 3, 5, 6, 7, ...} ? 
        # Just return {1}
        return [1]
        
    return res
