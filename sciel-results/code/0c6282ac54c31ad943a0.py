import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 512
    
    # Precompute all 4-APs
    ap_list = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(ap_idx)
            elem_to_aps[b].append(ap_idx)
            elem_to_aps[c].append(ap_idx)
            elem_to_aps[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # Precompute "others" masks for each AP and each element position
    others_mask = [[0]*4 for _ in range(num_aps)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        elems = [a, b, c, e]
        for p in range(4):
            m = 0
            for q in range(4):
                if q != p:
                    m |= (1 << (elems[q] - 1))
            others_mask[idx][p] = m
            
    def can_add(x, mask, counts):
        for ap_idx in elem_to_aps[x]:
            if counts[ap_idx] == 3:
                return False
        return True

    def do_add(x, mask, counts, size):
        mask |= (1 << (x - 1))
        size += 1
        for ap_idx in elem_to_aps[x]:
            counts[ap_idx] += 1
        return mask, size

    def do_remove(x, mask, counts, size):
        mask &= ~(1 << (x - 1))
        size -= 1
        for ap_idx in elem_to_aps[x]:
            counts[ap_idx] -= 1
        return mask, size

    best_size = 0
    best_mask = 0
    
    all_elems = list(range(1, n + 1))
    
    num_restarts = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = 0
        counts = [0] * num_aps
        size = 0
        
        # Greedy Phase with random ordering
        candidates = all_elems[:]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, mask, counts):
                mask, size = do_add(x, mask, counts, size)
                
        # Local Search with SA
        T = 1.0
        cooling = 0.99
        min_T = 0.1
        
        max_steps = 20000
        steps = 0
        no_improve_count = 0
        best_local_size = size
        best_local_mask = mask
        best_local_counts = counts[:]
        
        # Maintain in_set and out_set for efficient sampling
        in_set = []
        out_set = []
        in_arr = [False] * (n + 1)
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                in_set.append(x)
                in_arr[x] = True
            else:
                out_set.append(x)
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            action = random.random()
            
            if action < 0.7:
                # Try to add a random element from out_set
                if not out_set:
                    break
                x = out_set[random.randint(0, len(out_set) - 1)]
                if can_add(x, mask, counts):
                    mask, size = do_add(x, mask, counts, size)
                    in_arr[x] = True
                    in_set.append(x)
                    out_set.remove(x)
                    no_improve_count = 0
                    if size > best_local_size:
                        best_local_size = size
                        best_local_mask = mask
                        best_local_counts = counts[:]
                else:
                    no_improve_count += 1
            else:
                # Try a swap
                if not out_set or not in_set:
                    break
                x = out_set[random.randint(0, len(out_set) - 1)]
                y = in_set[random.randint(0, len(in_set) - 1)]
                
                # Remove y
                mask, size = do_remove(y, mask, counts, size)
                in_arr[y] = False
                in_set.remove(y)
                out_set.append(y)
                
                # Try add x
                if can_add(x, mask, counts):
                    mask, size = do_add(x, mask, counts, size)
                    in_arr[x] = True
                    in_set.append(x)
                    out_set.remove(x)
                    no_improve_count = 0
                    if size > best_local_size:
                        best_local_size = size
                        best_local_mask = mask
                        best_local_counts = counts[:]
                else:
                    # Revert: remove x, add y back
                    mask, size = do_remove(x, mask, counts, size)
                    in_arr[x] = False
                    in_set.pop()
                    out_set.append(x)
                    # Add y back to in_set
                    in_arr[y] = True
                    in_set.append(y)
                    # Remove y from out_set
                    out_set.remove(y)
                    no_improve_count += 1
            
            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T
            
            # Restart if stuck
            if no_improve_count > 1000:
                break
        
        # Use the best local solution
        if best_local_size > size:
            mask = best_local_mask
            size = best_local_size
            counts = best_local_counts[:]
        
        # Update global best
        if size > best_size:
            best_size = size
            best_mask = mask

    # Convert best_mask to list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result
