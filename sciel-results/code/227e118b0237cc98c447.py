import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = 80  # Safety margin for 90s limit
    
    # Precompute all 3-APs and 4-APs as bitmasks
    three_aps = []
    four_aps = []
    
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
    
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
    
    three_aps_np = np.array(three_aps, dtype=object)
    four_aps_np = np.array(four_aps, dtype=object)
    
    # Precompute element to 3-AP indices and 4-AP indices
    elem_to_3ap = {i: [] for i in range(1, N+1)}
    elem_to_4ap = {i: [] for i in range(1, N+1)}
    
    for idx, (a, b, c) in enumerate([(m.bit_length(), None, None) for m in []]):
        pass
    
    # Rebuild with explicit tuples for element mapping
    three_aps_tuples = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            idx = len(three_aps_tuples)
            three_aps_tuples.append((a, b, c))
            elem_to_3ap[a].append(idx)
            elem_to_3ap[b].append(idx)
            elem_to_3ap[c].append(idx)
    
    four_aps_tuples = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            idx = len(four_aps_tuples)
            four_aps_tuples.append((a, b, c, e))
            elem_to_4ap[a].append(idx)
            elem_to_4ap[b].append(idx)
            elem_to_4ap[c].append(idx)
            elem_to_4ap[e].append(idx)
    
    # Fast validity check: returns True if no 4-AP
    def is_valid(mask):
        for fm in four_aps_np:
            if (mask & fm) == fm:
                return False
        return True
    
    # Count 3-APs
    def count_3aps(mask):
        count = 0
        for tm in three_aps_np:
            if (mask & tm) == tm:
                count += 1
        return count
    
    # Delta for adding element x to mask
    def delta_add(mask, x):
        x_bit = 1 << (x-1)
        if mask & x_bit:
            return 0
        # Check 4-AP validity
        for idx in elem_to_4ap[x]:
            a, b, c, d = four_aps_tuples[idx]
            others_mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d-1))
            others_mask &= ~x_bit
            if (mask & others_mask) == others_mask:
                return None  # Invalid
        # Count new 3-APs
        delta = 0
        for idx in elem_to_3ap[x]:
            a, b, c = three_aps_tuples[idx]
            others_mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            others_mask &= ~x_bit
            if (mask & others_mask) == others_mask:
                delta += 1
        return delta
    
    # Delta for removing element x from mask
    def delta_remove(mask, x):
        x_bit = 1 << (x-1)
        if not (mask & x_bit):
            return 0
        delta = 0
        for idx in elem_to_3ap[x]:
            a, b, c = three_aps_tuples[idx]
            others_mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            others_mask &= ~x_bit
            if (mask & others_mask) == others_mask:
                delta -= 1
        return delta
    
    # Simulated annealing
    best_mask = 0
    best_score = 0
    
    # Greedy initialization
    def greedy_init():
        m = 0
        # Sort by 3-AP potential
        potential = {x: len(elem_to_3ap[x]) for x in range(1, N+1)}
        sorted_elems = sorted(range(1, N+1), key=lambda x: -potential[x])
        for x in sorted_elems:
            d = delta_add(m, x)
            if d is not None and d >= 0:
                m |= (1 << (x-1))
        return m
    
    # Multiple restarts with simulated annealing
    for restart in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Initialize
        if restart == 0:
            current_mask = greedy_init()
        else:
            # Random initialization with bias toward high-potential elements
            current_mask = 0
            potential = {x: len(elem_to_3ap[x]) for x in range(1, N+1)}
            sorted_elems = sorted(range(1, N+1), key=lambda x: -potential[x])
            # Take top 40% randomly
            top_k = int(0.4 * N)
            candidates = sorted_elems[:top_k]
            random.shuffle(candidates)
            for x in candidates:
                d = delta_add(current_mask, x)
                if d is not None:
                    current_mask |= (1 << (x-1))
        
        current_score = count_3aps(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        # Simulated annealing
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.9995
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - start_time > deadline:
                break
            
            # Choose move: add, remove, or swap
            move_type = random.random()
            valid_move = False
            new_mask = current_mask
            new_score = current_score
            
            if move_type < 0.3:
                # Add
                x = random.randint(1, N)
                d = delta_add(current_mask, x)
                if d is not None:
                    new_mask = current_mask | (1 << (x-1))
                    new_score = current_score + d
                    valid_move = True
            elif move_type < 0.6:
                # Remove
                if current_mask:
                    x = random.choice([x for x in range(1, N+1) if current_mask & (1 << (x-1))])
                    d = delta_remove(current_mask, x)
                    new_mask = current_mask & ~(1 << (x-1))
                    new_score = current_score + d
                    valid_move = True
            else:
                # Swap
                if current_mask:
                    x = random.choice([x for x in range(1, N+1) if current_mask & (1 << (x-1))])
                    y = random.randint(1, N)
                    if not (current_mask & (1 << (y-1))):
                        # Remove x, add y
                        temp_mask = current_mask & ~(1 << (x-1))
                        d_add = delta_add(temp_mask, y)
                        if d_add is not None:
                            d_rem = delta_remove(current_mask, x)
                            new_mask = temp_mask | (1 << (y-1))
                            new_score = current_score + d_rem + d_add
                            valid_move = True
            
            if not valid_move:
                continue
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Occasional restart with best
        if restart % 5 == 4 and best_mask != 0:
            current_mask = best_mask
            current_score = best_score
    
    # Convert best_mask to list
    result = []
    for i in range(1, N+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
    
    return sorted(result)
