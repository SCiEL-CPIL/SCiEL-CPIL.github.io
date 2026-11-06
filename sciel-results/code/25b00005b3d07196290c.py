import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # all_3aps_data: list of (mask, a, b, c)
    # all_4aps_data: list of (mask, a, b, c, e)
    all_3aps_data = []
    all_4aps_data = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            mask3 = (1 << a) | (1 << b) | (1 << c)
            all_3aps_data.append((mask3, a, b, c))
            
            if a + 3 * d <= N:
                e = a + 3 * d
                mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << e)
                all_4aps_data.append((mask4, a, b, c, e))
                
    # Build 'others' masks for fast incremental updates
    # three_ap_others[x] = list of masks of the other 2 elements in 3-APs with x
    three_ap_others = [[] for _ in range(N + 1)]
    for mask3, a, b, c in all_3aps_data:
        three_ap_others[a].append((1 << b) | (1 << c))
        three_ap_others[b].append((1 << a) | (1 << c))
        three_ap_others[c].append((1 << a) | (1 << b))
        
    # four_ap_others[x] = list of masks of the other 3 elements in 4-APs with x
    four_ap_others = [[] for _ in range(N + 1)]
    for mask4, a, b, c, e in all_4aps_data:
        four_ap_others[a].append((1 << b) | (1 << c) | (1 << e))
        four_ap_others[b].append((1 << a) | (1 << c) | (1 << e))
        four_ap_others[c].append((1 << a) | (1 << b) | (1 << e))
        four_ap_others[e].append((1 << a) | (1 << b) | (1 << c))

    # Helper: Check if adding x is valid (no 4-AP created)
    def is_valid_add(A_mask, x):
        if A_mask & (1 << x):
            return False
        for other_mask in four_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added
    def gain_3aps(A_mask, x):
        if A_mask & (1 << x):
            return 0
        gain = 0
        for other_mask in three_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed
    def loss_3aps(A_mask, x):
        if not (A_mask & (1 << x)):
            return 0
        loss = 0
        A_mask_without_x = A_mask & ~(1 << x)
        for other_mask in three_ap_others[x]:
            if (A_mask_without_x & other_mask) == other_mask:
                loss += 1
        return loss

    # Count total 3-APs in A_mask (full scan, used for init and check)
    def count_3aps(A_mask):
        count = 0
        for mask3, a, b, c in all_3aps_data:
            if (A_mask & mask3) == mask3:
                count += 1
        return count

    # Initialization: Degree-based Greedy
    # Calculate degree (number of 3-APs) and constraint (number of 4-APs)
    deg3 = [0] * (N + 1)
    deg4 = [0] * (N + 1)
    for x in range(1, N + 1):
        deg3[x] = len(three_ap_others[x])
        deg4[x] = len(four_ap_others[x])
        
    # Sort by density potential: high 3-AP degree, low 4-AP degree
    elements = list(range(1, N + 1))
    elements.sort(key=lambda x: (deg3[x] / (deg4[x] + 1), deg3[x]), reverse=True)
    
    def greedy_init():
        A_mask = 0
        # Shuffle slightly to allow restarts to vary? 
        # Or use the sorted order.
        # Let's use the sorted order for deterministic high-quality start.
        for x in elements:
            if is_valid_add(A_mask, x):
                # Add if it increases 3-AP count, or with some probability
                g = gain_3aps(A_mask, x)
                if g > 0 or random.random() < 0.2:
                    A_mask |= (1 << x)
        return A_mask

    # Simulated Annealing
    best_overall_mask = 0
    best_overall_score = 0
    
    # Run multiple SA instances
    while time.time() < deadline - 5:
        current_mask = greedy_init()
        current_score = count_3aps(current_mask)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_mask = current_mask
            
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        # Precompute list of elements in mask for faster removal/swap selection?
        # We can just iterate 1..N which is 256.
        
        while time.time() < deadline - 5:
            if T < T_min:
                break
                
            move_type = random.random()
            accepted = False
            delta = 0
            new_mask = current_mask
            new_score = current_score
            
            if move_type < 0.3: # Add
                x = random.randint(1, N)
                if not (current_mask & (1 << x)):
                    if is_valid_add(current_mask, x):
                        delta = gain_3aps(current_mask, x)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            new_mask = current_mask | (1 << x)
                            new_score = current_score + delta
                            accepted = True
                            
            elif move_type < 0.6: # Remove
                # Pick random element in mask
                # Optimization: keep a list? 
                # For now, iterate.
                elems_in_mask = []
                for i in range(1, N + 1):
                    if current_mask & (1 << i):
                        elems_in_mask.append(i)
                        
                if elems_in_mask:
                    x = random.choice(elems_in_mask)
                    delta = -loss_3aps(current_mask, x)
                    # Removal is always valid
                    if delta > 0 or random.random() < math.exp(delta / T):
                        new_mask = current_mask & ~(1 << x)
                        new_score = current_score + delta
                        accepted = True
                        
            else: # Swap
                elems_in_mask = []
                for i in range(1, N + 1):
                    if current_mask & (1 << i):
                        elems_in_mask.append(i)
                        
                if elems_in_mask:
                    x_rem = random.choice(elems_in_mask)
                    x_add = random.randint(1, N)
                    if not (current_mask & (1 << x_add)):
                        # Check validity of swap
                        # A_temp = current_mask without x_rem
                        # Check if x_add is valid in A_temp
                        A_temp = current_mask & ~(1 << x_rem)
                        valid = True
                        for other_mask in four_ap_others[x_add]:
                            if (A_temp & other_mask) == other_mask:
                                valid = False
                                break
                                
                        if valid:
                            # Calculate net gain
                            # Gain of adding x_add to A_temp
                            gain = 0
                            for other_mask in three_ap_others[x_add]:
                                if (A_temp & other_mask) == other_mask:
                                    gain += 1
                                    
                            # Loss of removing x_rem from current_mask
                            loss = 0
                            # Note: loss_3aps uses current_mask.
                            # But we are removing x_rem.
                            # The loss is the number of 3-APs containing x_rem 
                            # where the other 2 are in current_mask.
                            # This is exactly what loss_3aps(current_mask, x_rem) calculates.
                            loss = loss_3aps(current_mask, x_rem)
                            
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                new_mask = A_temp | (1 << x_add)
                                new_score = current_score + delta
                                accepted = True
            
            if accepted:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_mask = current_mask
            
            T *= cooling

    # Convert best mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    return result
