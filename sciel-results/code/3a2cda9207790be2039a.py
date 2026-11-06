import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    # 1. Precompute conflict masks
    # mask[i] is an integer where bit j is set if i and j conflict
    # We use 1-based indexing for nodes, so bit 1 corresponds to node 1.
    # Node i is represented by bit (1 << i).
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Helper to check if x can be added to current set represented by S_mask
    def can_add(x, S_mask):
        return (conflict_mask[x] & S_mask) == 0

    # Helper to check if x can be added to S_mask after removing y
    def can_swap(x, y, S_mask):
        # x must not conflict with any node in S except possibly y
        # (conflict_mask[x] & S_mask) should only have bit y set, or be 0
        # Let M = conflict_mask[x] & S_mask
        # We need M == 0 or M == (1 << y)
        M = conflict_mask[x] & S_mask
        return M == 0 or M == (1 << y)

    def get_size(S_mask):
        return bin(S_mask).count('1')

    # 2. Initialization: Greedy
    # We'll run a few greedy passes to find a good starting point
    best_initial_mask = 0
    best_initial_size = 0
    
    for _ in range(10):
        S_mask = 0
        # Random order for greedy
        nodes = list(range(1, N + 1))
        random.shuffle(nodes)
        
        # Try to add nodes in random order
        for x in nodes:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        
        size = get_size(S_mask)
        if size > best_initial_size:
            best_initial_size = size
            best_initial_mask = S_mask

    # Also try starting with all odds and greedily adding evens
    S_mask_odds = 0
    for i in range(1, N + 1, 2):
        S_mask_odds |= (1 << i)
    
    evens = list(range(2, N + 1, 2))
    random.shuffle(evens)
    S_mask_curr = S_mask_odds
    for x in evens:
        if can_add(x, S_mask_curr):
            S_mask_curr |= (1 << x)
    
    size_odds = get_size(S_mask_curr)
    if size_odds > best_initial_size:
        best_initial_size = size_odds
        best_initial_mask = S_mask_curr

    # 3. Simulated Annealing
    S_mask = best_initial_mask
    S_list = [i for i in range(1, N + 1) if (S_mask >> i) & 1]
    
    best_mask = S_mask
    best_size = best_initial_size
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    stagnation = 0
    max_stagnation = 500
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            # Pick random x not in S
            # To avoid O(N) scan, we can pick random and retry
            # Or maintain a list of available nodes. 
            # Given N=256, retry is fast enough.
            while True:
                x = random.randint(1, N)
                if not (S_mask & (1 << x)):
                    break
            
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                S_list.append(x)
                new_size = get_size(S_mask)
                
                if new_size > best_size:
                    best_size = new_size
                    best_mask = S_mask
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
                
        elif move_type < 0.7:
            # Remove
            if S_list:
                idx = random.randint(0, len(S_list) - 1)
                y = S_list[idx]
                # Swap with last for O(1) removal
                S_list[idx] = S_list[-1]
                S_list.pop()
                S_mask &= ~(1 << y)
                
                new_size = get_size(S_mask)
                # Delta energy = -1 (size decreased)
                # Accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    stagnation += 1
                else:
                    # Reject: restore
                    S_mask |= (1 << y)
                    S_list.append(y)
                    stagnation += 1
                    
        else:
            # Swap
            if S_list:
                y = random.choice(S_list)
                # Pick x not in S
                while True:
                    x = random.randint(1, N)
                    if not (S_mask & (1 << x)):
                        break
                
                if can_swap(x, y, S_mask):
                    # Perform swap
                    S_mask &= ~(1 << y)
                    S_mask |= (1 << x)
                    
                    # Update S_list
                    # Find y in S_list and replace with x
                    # O(N) is fine for N=256
                    for i in range(len(S_list)):
                        if S_list[i] == y:
                            S_list[i] = x
                            break
                            
                    # Size unchanged. 
                    # Accept to explore. 
                    # We can use a fixed probability or always accept swaps if valid.
                    # Always accepting valid swaps helps escape local minima where 
                    # a swap enables a future add.
                    stagnation += 1
                    # Note: We don't update best_size here as size is same.
                    # But we might have found a better configuration.
                    
        # Restart if stuck
        if stagnation > max_stagnation:
            T = 0.5
            stagnation = 0
            
        T *= cooling

    # Return the best set found
    return [i for i in range(1, N + 1) if (best_mask >> i) & 1]
