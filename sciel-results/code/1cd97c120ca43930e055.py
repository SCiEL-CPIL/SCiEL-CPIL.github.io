import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _delta_score(bits, x, y):
    """
    Compute the change in score when removing x and adding y from the set.
    bits is the bitset BEFORE the move.
    Returns (delta, new_bits).
    """
    new_bits = bits ^ (1 << x) ^ (1 << y)
    delta = 0
    for d in range(1, 256):
        # Check if d was a valid difference before the move
        was_valid = (bits & (bits >> d) & (bits >> (2 * d))) != 0
        # Check if d is a valid difference after the move
        now_valid = (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0
        if was_valid and not now_valid:
            delta -= 1
        elif not was_valid and now_valid:
            delta += 1
    return delta, new_bits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    while time.time() < deadline:
        # Initialize random set
        current_list = random.sample(range(1, MAXV + 1), N)
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        T = 8.0
        T_min = 0.01
        cooling = 0.9995
        
        local_deadline = start_time + min(4.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Choose move type: 1-swap (70%), completion (20%), 2-swap (10%)
            move_type = random.random()
            
            if move_type < 0.7:
                # 1-swap: remove x, add y
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                while True:
                    y = random.randint(1, MAXV)
                    if not ((current_bits >> y) & 1):
                        break
                
                delta, new_bits = _delta_score(current_bits, x, y)
                
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    current_list = current_list[:x_idx] + current_list[x_idx+1:]
                    ins_idx = bisect.bisect_left(current_list, y)
                    current_list.insert(ins_idx, y)
                    current_bits = new_bits
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list.copy()
                        
            elif move_type < 0.9:
                # Completion move: try to complete a progression
                # Pick two elements a, b from the set, try to add a+2d or a-d
                idx1, idx2 = random.sample(range(N), 2)
                a = current_list[idx1]
                b = current_list[idx2]
                d = abs(b - a)
                
                candidates = []
                if a + 2 * d <= MAXV and not ((current_bits >> (a + 2 * d)) & 1):
                    candidates.append((a, a + 2 * d))  # would complete a, a+d, a+2d
                if a - d >= 1 and not ((current_bits >> (a - d)) & 1):
                    candidates.append((a - d, a))  # would complete a-d, a, a+d
                if b + d <= MAXV and not ((current_bits >> (b + d)) & 1):
                    candidates.append((b, b + d))  # would complete b, b+d, b+2d
                if b - 2 * d >= 1 and not ((current_bits >> (b - 2 * d)) & 1):
                    candidates.append((b - 2 * d, b))  # would complete b-2d, b-d, b
                
                if candidates:
                    # Pick a candidate and remove a random element to keep size N
                    target_a, target_b = random.choice(candidates)
                    # The new element to add is the one not in {a, b}
                    if target_a == a:
                        new_elem = target_b
                    elif target_a == b:
                        new_elem = target_a
                    elif target_b == a:
                        new_elem = target_a
                    else:
                        new_elem = target_b
                    
                    # Remove a random element (not the ones we want to keep)
                    remove_idx = random.randint(0, N - 1)
                    x = current_list[remove_idx]
                    if x == new_elem:
                        remove_idx = (remove_idx + 1) % N
                        x = current_list[remove_idx]
                    
                    delta, new_bits = _delta_score(current_bits, x, new_elem)
                    
                    if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                        current_list = current_list[:remove_idx] + current_list[remove_idx+1:]
                        ins_idx = bisect.bisect_left(current_list, new_elem)
                        current_list.insert(ins_idx, new_elem)
                        current_bits = new_bits
                        current_score += delta
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_list = current_list.copy()
                        
            else:
                # 2-swap: remove x1, x2, add y1, y2
                idxs = random.sample(range(N), 2)
                x1 = current_list[idxs[0]]
                x2 = current_list[idxs[1]]
                
                while True:
                    y1 = random.randint(1, MAXV)
                    if not ((current_bits >> y1) & 1) and y1 != x1 and y1 != x2:
                        break
                while True:
                    y2 = random.randint(1, MAXV)
                    if not ((current_bits >> y2) & 1) and y2 != x1 and y2 != x2 and y2 != y1:
                        break
                
                # Compute delta for 2-swap
                intermediate_bits = current_bits ^ (1 << x1) ^ (1 << x2)
                new_bits = intermediate_bits ^ (1 << y1) ^ (1 << y2)
                delta = 0
                for d in range(1, 256):
                    was_valid = (current_bits & (current_bits >> d) & (current_bits >> (2 * d))) != 0
                    now_valid = (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0
                    if was_valid and not now_valid:
                        delta -= 1
                    elif not was_valid and now_valid:
                        delta += 1
                
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    # Update list
                    if idxs[0] > idxs[1]:
                        idxs[0], idxs[1] = idxs[1], idxs[0]
                    
                    new_list = current_list[:idxs[0]] + current_list[idxs[0]+1:idxs[1]] + current_list[idxs[1]+1:]
                    bisect.insort(new_list, y1)
                    bisect.insort(new_list, y2)
                    
                    current_list = new_list
                    current_bits = new_bits
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
