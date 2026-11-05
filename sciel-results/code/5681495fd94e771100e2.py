import time
import random
import math

N = 256

# Precompute 3-APs and 4-APs
three_aps = []
four_aps = []

# three_ap_indices[x] = list of indices of 3-APs containing x
three_ap_indices = [[] for _ in range(N + 1)]
# four_ap_conflicts[x] = list of masks of the other 3 elements in a 4-AP with x
four_ap_conflicts = [[] for _ in range(N + 1)]

def precompute():
    global three_aps, four_aps
    
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            idx = len(three_aps)
            three_aps.append(mask)
            three_ap_indices[a].append(idx)
            three_ap_indices[b].append(idx)
            three_ap_indices[c].append(idx)
            
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # For each element in the 4-AP, store the mask of the other 3
            elements = [a, b, c, e]
            for i, elem in enumerate(elements):
                others_mask = 0
                for j, other in enumerate(elements):
                    if i != j:
                        others_mask |= (1 << (other - 1))
                four_ap_conflicts[elem].append(others_mask)

precompute()

def count_3aps(S_mask):
    """Count 3-APs in S_mask."""
    count = 0
    for mask in three_aps:
        if (S_mask & mask) == mask:
            count += 1
    return count

def can_add(x, S_mask):
    """Check if adding x to S_mask creates a 4-AP."""
    for others_mask in four_ap_conflicts[x]:
        if (S_mask & others_mask) == others_mask:
            return False
    return True

def gain_add(x, S_mask):
    """Calculate number of new 3-APs formed by adding x to S_mask."""
    bit = 1 << (x - 1)
    gain = 0
    for idx in three_ap_indices[x]:
        m = three_aps[idx]
        # Check if the other 2 elements are in S_mask
        if (S_mask & (m ^ bit)) == (m ^ bit):
            gain += 1
    return gain

def loss_remove(x, S_mask):
    """Calculate number of 3-APs lost by removing x from S_mask."""
    bit = 1 << (x - 1)
    loss = 0
    for idx in three_ap_indices[x]:
        m = three_aps[idx]
        # Check if the other 2 elements are in S_mask
        if (S_mask & (m ^ bit)) == (m ^ bit):
            loss += 1
    return loss

def greedy_construct():
    S_mask = 0
    available = list(range(1, N + 1))
    
    while available:
        best_x = -1
        best_gain = -1
        for x in available:
            if not can_add(x, S_mask):
                continue
            g = gain_add(x, S_mask)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x == -1:
            break
            
        S_mask |= (1 << (best_x - 1))
        # Remove best_x and its conflicts from available
        new_available = []
        for x in available:
            if x == best_x:
                continue
            if not can_add(x, S_mask):
                continue
            new_available.append(x)
        available = new_available
        
    return S_mask

def local_search(S_mask, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_S = S_mask
    current_score = count_3aps(S_mask)
    best_S = S_mask
    best_score = current_score
    
    # Precompute list of elements in S for faster iteration
    def get_elements(mask):
        elems = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                elems.append(i)
        return elems
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elems = get_elements(current_S)
        
        for x in elems:
            bit_x = 1 << (x - 1)
            # Remove x
            S_without_x = current_S ^ bit_x
            score_without_x = current_score - loss_remove(x, S_without_x)
            
            # Find best y to add
            best_y = -1
            best_delta = 0
            best_score_y = -1
            
            for y in range(1, N + 1):
                bit_y = 1 << (y - 1)
                if S_without_x & bit_y:
                    continue
                if not can_add(y, S_without_x):
                    continue
                
                g = gain_add(y, S_without_x)
                new_score = score_without_x + g
                if new_score > best_score_y:
                    best_score_y = new_score
                    best_y = y
                    best_delta = new_score - current_score
            
            if best_y != -1 and best_delta > 0:
                current_S = S_without_x | (1 << (best_y - 1))
                current_score = best_score_y
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_S
                    
    return best_S

def simulated_annealing(S_mask, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_S = S_mask
    current_score = count_3aps(S_mask)
    best_S = S_mask
    best_score = current_score
    
    initial_temp = 10.0
    cooling_rate = 0.95
    temp = initial_temp
    
    # Precompute list of elements in S for faster iteration
    def get_elements(mask):
        elems = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                elems.append(i)
        return elems
    
    while time.time() < deadline and temp > 0.01:
        temp *= cooling_rate
        
        # Try a random move: remove x, add y
        elems = get_elements(current_S)
        if not elems:
            break
            
        x = random.choice(elems)
        bit_x = 1 << (x - 1)
        S_without_x = current_S ^ bit_x
        score_without_x = current_score - loss_remove(x, S_without_x)
        
        # Find a random y to add
        valid_ys = []
        for y in range(1, N + 1):
            bit_y = 1 << (y - 1)
            if S_without_x & bit_y:
                continue
            if not can_add(y, S_without_x):
                continue
            valid_ys.append(y)
        
        if not valid_ys:
            continue
            
        y = random.choice(valid_ys)
        bit_y = 1 << (y - 1)
        new_S = S_without_x | bit_y
        new_score = score_without_x + gain_add(y, S_without_x)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/temp) if worse
        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_S = new_S
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_S = current_S
                
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Greedy Construction
    S_mask = greedy_construct()
    
    # 2. Local Search
    S_mask = local_search(S_mask, 30.0)
    
    # 3. Simulated Annealing
    S_mask = simulated_annealing(S_mask, 45.0)
    
    # Convert mask to list
    result = []
    for i in range(N):
        if S_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
