import time
import random
import math

N = 256
MASK_ALL = (1 << N) - 1

# Precomputation
masks_3ap = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for d in range(1, N):
        a = x - d
        c = x + d
        if a >= 1 and c <= N:
            mask = (1 << (a - 1)) | (1 << (c - 1))
            masks_3ap[x].append(mask)
        else:
            break

masks_4ap = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for d in range(1, N):
        if x + 3 * d <= N:
            m = (1 << (x + d - 1)) | (1 << (x + 2 * d - 1)) | (1 << (x + 3 * d - 1))
            masks_4ap[x].append(m)
        if x - d >= 1 and x + 2 * d <= N:
            m = (1 << (x - d - 1)) | (1 << (x + d - 1)) | (1 << (x + 2 * d - 1))
            masks_4ap[x].append(m)
        if x - 2 * d >= 1 and x + d <= N:
            m = (1 << (x - 2 * d - 1)) | (1 << (x - d - 1)) | (1 << (x + d - 1))
            masks_4ap[x].append(m)
        if x - 3 * d >= 1:
            m = (1 << (x - 3 * d - 1)) | (1 << (x - 2 * d - 1)) | (1 << (x - d - 1))
            masks_4ap[x].append(m)

def count_3aps(state):
    count = 0
    s = state
    while s:
        lsb = s & -s
        x = lsb.bit_length()
        for m in masks_3ap[x]:
            if (state & m) == m:
                count += 1
        s ^= lsb
    return count

def is_valid_add(x, state):
    for m in masks_4ap[x]:
        if (state & m) == m:
            return False
    return True

def delta_add(x, state):
    delta = 0
    for m in masks_3ap[x]:
        if (state & m) == m:
            delta += 1
    return delta

def delta_remove(x, state):
    delta = 0
    for m in masks_3ap[x]:
        if (state & m) == m:
            delta += 1
    return delta

def greedy_construct():
    state = 0
    for x in range(1, N + 1):
        if is_valid_add(x, state):
            state |= (1 << (x - 1))
    return state

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_state = greedy_construct()
    best_score = count_3aps(best_state)
    
    current_state = best_state
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            x = random.choice(all_elements)
            if not (current_state & (1 << (x - 1))):
                if is_valid_add(x, current_state):
                    new_state = current_state | (1 << (x - 1))
                    new_score = current_score + delta_add(x, current_state)
                    
                    if new_score >= current_score:
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = current_state
                    else:
                        if random.random() < math.exp((new_score - current_score) / T):
                            current_state = new_state
                            current_score = new_score
                            
        elif move_type < 0.8:
            x = random.choice(all_elements)
            if current_state & (1 << (x - 1)):
                new_state = current_state ^ (1 << (x - 1))
                new_score = current_score - delta_remove(x, current_state)
                
                if new_score >= current_score:
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = current_state
                else:
                    if random.random() < math.exp((new_score - current_score) / T):
                        current_state = new_state
                        current_score = new_score
                        
        else:
            for _ in range(10):
                x = random.choice(all_elements)
                y = random.choice(all_elements)
                if (current_state & (1 << (x - 1))) and not (current_state & (1 << (y - 1))):
                    temp_state = current_state ^ (1 << (x - 1))
                    if is_valid_add(y, temp_state):
                        new_state = temp_state | (1 << (y - 1))
                        lost = delta_remove(x, current_state)
                        gained = delta_add(y, new_state)
                        new_score = current_score - lost + gained
                        
                        if new_score >= current_score:
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = current_state
                        else:
                            if random.random() < math.exp((new_score - current_score) / T):
                                current_state = new_state
                                current_score = new_score
                    break
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    result = []
    s = best_state
    while s:
        lsb = s & -s
        x = lsb.bit_length()
        result.append(x)
        s ^= lsb
        
    return sorted(result)
