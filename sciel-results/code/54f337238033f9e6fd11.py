import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs data structures
    # three_aps_data: list of (a, b, c)
    # four_aps_data: list of (a, b, c, e)
    three_aps_data = []
    four_aps_data = []
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                three_aps_data.append((a, b, c))
                
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                four_aps_data.append((a, b, c, e))
    
    # Build indices: ap3_indices[x] = list of indices of 3-APs containing x
    ap3_indices = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps_data):
        ap3_indices[a].append(idx)
        ap3_indices[b].append(idx)
        ap3_indices[c].append(idx)
        
    ap4_indices = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(four_aps_data):
        ap4_indices[a].append(idx)
        ap4_indices[b].append(idx)
        ap4_indices[c].append(idx)
        ap4_indices[e].append(idx)
        
    # Precompute "other_mask" for each AP and each position to speed up checks
    # For 3-AP idx with elements (a,b,c):
    # other_mask_0 = mask(b,c), other_mask_1 = mask(a,c), other_mask_2 = mask(a,b)
    # For 4-AP idx with elements (a,b,c,e):
    # other_mask_0 = mask(b,c,e), etc.
    
    ap3_other_masks = [[0,0,0] for _ in range(len(three_aps_data))]
    for idx, (a, b, c) in enumerate(three_aps_data):
        ap3_other_masks[idx][0] = (1 << b) | (1 << c)
        ap3_other_masks[idx][1] = (1 << a) | (1 << c)
        ap3_other_masks[idx][2] = (1 << a) | (1 << b)
        
    ap4_other_masks = [[0,0,0,0] for _ in range(len(four_aps_data))]
    for idx, (a, b, c, e) in enumerate(four_aps_data):
        ap4_other_masks[idx][0] = (1 << b) | (1 << c) | (1 << e)
        ap4_other_masks[idx][1] = (1 << a) | (1 << c) | (1 << e)
        ap4_other_masks[idx][2] = (1 << a) | (1 << b) | (1 << e)
        ap4_other_masks[idx][3] = (1 << a) | (1 << b) | (1 << c)
        
    # Helper: check if adding x to mask creates a 4-AP
    def creates_4ap(mask, x):
        # For each 4-AP containing x, check if the other 3 elements are in mask
        # We need to know which position x is in the 4-AP to select the correct other_mask
        # Alternatively, just check all 4-APs containing x and see if the set of other elements is subset of mask.
        # Since we precomputed ap4_indices[x], we can iterate.
        # To be fast, we precomputed ap4_other_masks. But we need to know which slot x is in.
        # Let's store ap4_pos_indices[x] = list of (idx, pos)
        
        # Wait, I didn't build ap4_pos_indices. Let's build it now or do a generic check.
        # Generic check: for each idx in ap4_indices[x], let (a,b,c,e) = four_aps_data[idx].
        # other = set({a,b,c,e} - {x}).
        # Check if mask contains all elements in other.
        # This is O(3) bit checks per AP.
        
        for idx in ap4_indices[x]:
            a, b, c, e = four_aps_data[idx]
            # Determine which element is x
            # We can just build the mask of the other 3
            om = 0
            if a != x: om |= (1 << a)
            if b != x: om |= (1 << b)
            if c != x: om |= (1 << c)
            if e != x: om |= (1 << e)
            if (mask & om) == om:
                return True
        return False
        
    # Helper: count new 3-APs created by adding x
    def gain_3aps(mask, x):
        gain = 0
        for idx in ap3_indices[x]:
            a, b, c = three_aps_data[idx]
            om = 0
            if a != x: om |= (1 << a)
            if b != x: om |= (1 << b)
            if c != x: om |= (1 << c)
            if (mask & om) == om:
                gain += 1
        return gain
        
    # Helper: count 3-APs lost by removing x
    def loss_3aps(mask, x):
        loss = 0
        for idx in ap3_indices[x]:
            a, b, c = three_aps_data[idx]
            om = 0
            if a != x: om |= (1 << a)
            if b != x: om |= (1 << b)
            if c != x: om |= (1 << c)
            if (mask & om) == om:
                loss += 1
        return loss
        
    # Helper: count total 3-APs in mask
    def count_3aps(mask):
        count = 0
        for a, b, c in three_aps_data:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                count += 1
        return count

    # Greedy Initialization
    def greedy_init(order=None):
        mask = 0
        if order is None:
            order = list(range(1, N + 1))
            random.shuffle(order)
        
        for x in order:
            if not creates_4ap(mask, x):
                mask |= (1 << x)
        return mask

    # Local Search / Hill Climbing
    def local_search(mask, max_iter=5000):
        current_mask = mask
        current_count = count_3aps(current_mask)
        best_mask = current_mask
        best_count = current_count
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try adding elements
            # To make it faster, we can prioritize elements with high potential gain
            # But for now, random or sequential
            candidates = [x for x in range(1, N + 1) if not (current_mask & (1 << x))]
            random.shuffle(candidates)
            
            for x in candidates:
                if creates_4ap(current_mask, x):
                    continue
                gain = gain_3aps(current_mask, x)
                if gain > 0:
                    new_mask = current_mask | (1 << x)
                    new_count = current_count + gain
                    current_mask = new_mask
                    current_count = new_count
                    improved = True
                    break
            
            if not improved:
                # Try swaps (remove one, add another)
                # Pick a random element to remove
                in_mask = [x for x in range(1, N + 1) if (current_mask & (1 << x))]
                if not in_mask:
                    break
                x_in = random.choice(in_mask)
                
                # Try replacing it with a random element not in mask
                # We can try a few random replacements
                replaced = False
                for _ in range(10): # Try 10 random swaps
                    x_out = random.randint(1, N)
                    if current_mask & (1 << x_out):
                        continue
                    if x_out == x_in:
                        continue
                    
                    temp_mask = current_mask & ~(1 << x_in)
                    if creates_4ap(temp_mask, x_out):
                        continue
                    
                    loss = loss_3aps(current_mask, x_in)
                    gain = gain_3aps(temp_mask, x_out)
                    delta = gain - loss
                    
                    if delta > 0:
                        current_mask = temp_mask | (1 << x_out)
                        current_count += delta
                        improved = True
                        replaced = True
                        break
            
            if not improved:
                # Try 2-swap (remove 2, add 2) - maybe too complex, stick to 1-swap for now
                # Or break if no improvement for a while
                # Let's just break to avoid infinite loop if stuck
                # Actually, SA is better for escaping. Let's rely on SA.
                break
                
            if current_count > best_count:
                best_mask = current_mask
                best_count = current_count
                
        return best_mask, best_count

    # Simulated Annealing
    def simulated_annealing(initial_mask, steps=20000):
        current_mask = initial_mask
        current_count = count_3aps(current_mask)
        best_mask = current_mask
        best_count = current_count
        
        T = 5.0
        cooling_rate = 0.999
        min_T = 0.1
        
        for step in range(steps):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                x = random.randint(1, N)
                if current_mask & (1 << x):
                    continue
                if creates_4ap(current_mask, x):
                    continue
                
                delta = gain_3aps(current_mask, x)
                new_mask = current_mask | (1 << x)
                new_count = current_count + delta
                
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
                    
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_count = new_count
                    
            elif move_type < 0.7:
                # Remove
                x = random.randint(1, N)
                if not (current_mask & (1 << x)):
                    continue
                
                delta = -loss_3aps(current_mask, x)
                new_mask = current_mask & ~(1 << x)
                new_count = current_count + delta
                
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
                    
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_count = new_count
                    
            else:
                # Swap
                x_in = random.randint(1, N)
                if not (current_mask & (1 << x_in)):
                    continue
                x_out = random.randint(1, N)
                if current_mask & (1 << x_out):
                    continue
                if x_in == x_out:
                    continue
                    
                temp_mask = current_mask & ~(1 << x_in)
                if creates_4ap(temp_mask, x_out):
                    continue
                
                loss = loss_3aps(current_mask, x_in)
                gain = gain_3aps(temp_mask, x_out)
                delta = gain - loss
                
                new_mask = temp_mask | (1 << x_out)
                new_count = current_count + delta
                
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
                    
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_count = new_count
                    
            T *= cooling_rate
            
        return best_mask, best_count

    # Main Logic
    best_overall_mask = 0
    best_overall_count = 0
    
    # 1. Multiple Greedy Initializations
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
    ]
    
    # Add a few random orders
    for _ in range(5):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy_init(order)
        count = count_3aps(mask)
        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask
            
        # Polish with local search
        if time.time() > deadline - 10:
            break
        lm, lc = local_search(mask, max_iter=1000)
        if lc > best_overall_count:
            best_overall_count = lc
            best_overall_mask = lm

    # 2. Simulated Annealing on the best found so far
    if time.time() < deadline - 5:
        sa_mask, sa_count = simulated_annealing(best_overall_mask, steps=30000)
        if sa_count > best_overall_count:
            best_overall_count = sa_count
            best_overall_mask = sa_mask
            
        # Polish the SA result
        if time.time() < deadline - 2:
            lm, lc = local_search(best_overall_mask, max_iter=2000)
            if lc > best_overall_count:
                best_overall_count = lc
                best_overall_mask = lm

    # 3. Random Restarts if time permits
    while time.time() < deadline - 2:
        mask = greedy_init()
        count = count_3aps(mask)
        
        # Quick SA
        if time.time() < deadline - 2:
            sa_mask, sa_count = simulated_annealing(mask, steps=5000)
            if sa_count > count:
                mask = sa_mask
                count = sa_count
        
        # Quick Local Search
        if time.time() < deadline - 1:
            lm, lc = local_search(mask, max_iter=500)
            if lc > count:
                mask = lm
                count = lc
                
        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask

    # Convert mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
