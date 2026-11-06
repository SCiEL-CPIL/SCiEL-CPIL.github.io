import random
import time

N = 256

# Precomputation
all_4aps = []
for a in range(1, N + 1):
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        all_4aps.append((a, a+d, a+2*d, a+3*d))

all_4ap_masks = []
for a, b, c, d in all_4aps:
    mask = 0
    mask |= 1 << (a-1)
    mask |= 1 << (b-1)
    mask |= 1 << (c-1)
    mask |= 1 << (d-1)
    all_4ap_masks.append(mask)

elem_to_4aps = {x: [] for x in range(1, N+1)}
for i, (a, b, c, d) in enumerate(all_4aps):
    for x in (a, b, c, d):
        elem_to_4aps[x].append(i)

all_3aps = []
for a in range(1, N + 1):
    max_d = (N - a) // 2
    for d in range(1, max_d + 1):
        all_3aps.append((a, a+d, a+2*d))

all_3ap_masks = []
for a, b, c in all_3aps:
    mask = 0
    mask |= 1 << (a-1)
    mask |= 1 << (b-1)
    mask |= 1 << (c-1)
    all_3ap_masks.append(mask)

def count_3aps_fast(mask):
    count = 0
    for m in all_3ap_masks:
        if (mask & m) == m:
            count += 1
    return count

def construct():
    best_mask = 0
    best_count = -1
    
    start_time = time.time()
    elems_list = list(range(1, N+1))
    
    while True:
        if time.time() - start_time > 55:
            break
            
        random.shuffle(elems_list)
        current_mask = 0
        
        for x in elems_list:
            bit = 1 << (x-1)
            safe = True
            for idx in elem_to_4aps[x]:
                other_mask = all_4ap_masks[idx] ^ bit
                if (current_mask & other_mask) == other_mask:
                    safe = False
                    break
            
            if safe:
                current_mask |= bit
        
        cnt = count_3aps_fast(current_mask)
        if cnt > best_count:
            best_count = cnt
            best_mask = current_mask
            
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
