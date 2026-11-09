import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP conflicts and 3-AP contributions
    four_conflicts = [[] for _ in range(N)]
    three_contributions = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for d in range(1, (N-3)//3 + 1):
        for a in range(1, N - 3*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                four_conflicts[idx].append(mask_other)
                
    # Generate all 3-APs
    for d in range(1, (N-2)//2 + 1):
        for a in range(1, N - 2*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                three_contributions[idx].append(mask_other)
                
    def is_valid_add(x, A_mask):
        bit_x = 1 << x
        if A_mask & bit_x:
            return False
        for mask_other in four_conflicts[x]:
            if (A_mask & mask_other) == mask_other:
                return False
        return True
        
    def get_3ap_gain(x, A_mask):
        gain = 0
        for mask_other in three_contributions[x]:
            if (A_mask & mask_other) == mask_other:
                gain += 1
        return gain
        
    def count_3aps(A_mask):
        total = 0
        temp_mask = A_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            x = lsb.bit_length() - 1
            total += get_3ap_gain(x, A_mask)
            temp_mask ^= lsb
        return total // 3

    # Best-first greedy construction
    A_mask = 0
    available = set(range(N))
    
    while available and time.time() < deadline - 15:
        best_x = -1
        best_gain = -1
        for x in available:
            if is_valid_add(x, A_mask):
                gain = get_3ap_gain(x, A_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
        if best_x == -1:
            break
            
        A_mask |= (1 << best_x)
        available.remove(best_x)
        
    best_A_mask = A_mask
    best_score = count_3aps(A_mask)
    
    # Hill climbing with multiple move types
    current_A_mask = A_mask
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_A_mask = current_A_mask
        valid_move = True
        new_score = -1
        
        if move_type == 'add':
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            if is_valid_add(x, current_A_mask):
                new_A_mask = current_A_mask | (1 << x)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        elif move_type == 'remove':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            if not in_A:
                continue
            x = random.choice(in_A)
            new_A_mask = current_A_mask & ~(1 << x)
            new_score = count_3aps(new_A_mask)
            
        elif move_type == 'swap':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not in_A or not not_in_A:
                continue
            x = random.choice(in_A)
            y = random.choice(not_in_A)
            temp_mask = current_A_mask & ~(1 << x)
            if is_valid_add(y, temp_mask):
                new_A_mask = temp_mask | (1 << y)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        if valid_move:
            if new_score > current_score:
                current_A_mask = new_A_mask
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_A_mask = current_A_mask
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb: remove 3 random elements, add 3 random valid ones
                    for _ in range(3):
                        in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
                        if in_A:
                            x = random.choice(in_A)
                            current_A_mask &= ~(1 << x)
                    for _ in range(3):
                        not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
                        if not not_in_A: break
                        best_x = -1
                        best_gain = -1
                        for x in not_in_A:
                            if is_valid_add(x, current_A_mask):
                                gain = get_3ap_gain(x, current_A_mask)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                        if best_x != -1:
                            current_A_mask |= (1 << best_x)
                    current_score = count_3aps(current_A_mask)
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_A_mask = current_A_mask
                        
    result = []
    for i in range(N):
        if best_A_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP conflicts and 3-AP contributions
    four_conflicts = [[] for _ in range(N)]
    three_contributions = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for d in range(1, (N-3)//3 + 1):
        for a in range(1, N - 3*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1, a+3*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                four_conflicts[idx].append(mask_other)
                
    # Generate all 3-APs
    for d in range(1, (N-2)//2 + 1):
        for a in range(1, N - 2*d + 1):
            idxs = [a-1, a+d-1, a+2*d-1]
            mask_full = 0
            for idx in idxs:
                mask_full |= (1 << idx)
            for i, idx in enumerate(idxs):
                mask_other = mask_full ^ (1 << idx)
                three_contributions[idx].append(mask_other)
                
    def is_valid_add(x, A_mask):
        bit_x = 1 << x
        if A_mask & bit_x:
            return False
        for mask_other in four_conflicts[x]:
            if (A_mask & mask_other) == mask_other:
                return False
        return True
        
    def get_3ap_gain(x, A_mask):
        gain = 0
        for mask_other in three_contributions[x]:
            if (A_mask & mask_other) == mask_other:
                gain += 1
        return gain
        
    def count_3aps(A_mask):
        total = 0
        temp_mask = A_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            x = lsb.bit_length() - 1
            total += get_3ap_gain(x, A_mask)
            temp_mask ^= lsb
        return total // 3

    # Best-first greedy construction
    A_mask = 0
    available = set(range(N))
    
    while available and time.time() < deadline - 15:
        best_x = -1
        best_gain = -1
        for x in available:
            if is_valid_add(x, A_mask):
                gain = get_3ap_gain(x, A_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
        if best_x == -1:
            break
            
        A_mask |= (1 << best_x)
        available.remove(best_x)
        
    best_A_mask = A_mask
    best_score = count_3aps(A_mask)
    
    # Hill climbing with multiple move types
    current_A_mask = A_mask
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_A_mask = current_A_mask
        valid_move = True
        new_score = -1
        
        if move_type == 'add':
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            if is_valid_add(x, current_A_mask):
                new_A_mask = current_A_mask | (1 << x)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        elif move_type == 'remove':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            if not in_A:
                continue
            x = random.choice(in_A)
            new_A_mask = current_A_mask & ~(1 << x)
            new_score = count_3aps(new_A_mask)
            
        elif move_type == 'swap':
            in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
            not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
            if not in_A or not not_in_A:
                continue
            x = random.choice(in_A)
            y = random.choice(not_in_A)
            temp_mask = current_A_mask & ~(1 << x)
            if is_valid_add(y, temp_mask):
                new_A_mask = temp_mask | (1 << y)
                new_score = count_3aps(new_A_mask)
            else:
                valid_move = False
                
        if valid_move:
            if new_score > current_score:
                current_A_mask = new_A_mask
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_A_mask = current_A_mask
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb: remove 3 random elements, add 3 random valid ones
                    for _ in range(3):
                        in_A = [x for x in range(N) if (current_A_mask & (1 << x))]
                        if in_A:
                            x = random.choice(in_A)
                            current_A_mask &= ~(1 << x)
                    for _ in range(3):
                        not_in_A = [x for x in range(N) if not (current_A_mask & (1 << x))]
                        if not not_in_A: break
                        best_x = -1
                        best_gain = -1
                        for x in not_in_A:
                            if is_valid_add(x, current_A_mask):
                                gain = get_3ap_gain(x, current_A_mask)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_x = x
                        if best_x != -1:
                            current_A_mask |= (1 << best_x)
                    current_score = count_3aps(current_A_mask)
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_A_mask = current_A_mask
                        
    result = []
    for i in range(N):
        if best_A_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)
# EVOLVE-BLOCK-END
