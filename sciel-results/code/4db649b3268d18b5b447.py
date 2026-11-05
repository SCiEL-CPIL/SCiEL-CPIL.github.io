import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 85.0  # Safety margin
    
    # 1. Precompute 4-AP conflicts
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    # 2. Precompute 3-AP partners
    partners = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            partners[a].append((b, c))
            partners[b].append((a, c))
            partners[c].append((a, b))

    def is_valid_add(mask, x):
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def delta_3aps_add(mask, x):
        delta = 0
        for (y, z) in partners[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                delta += 1
        return delta

    def count_3aps(mask):
        count = 0
        for i in range(N):
            if (mask >> i) & 1:
                count += delta_3aps_add(mask & ~(1 << i), i)
        return count // 3

    def run_sa(initial_mask, time_limit):
        current_mask = initial_mask
        current_score = count_3aps(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        steps_since_improve = 0
        
        while time.time() < time_limit:
            move_type = random.random()
            new_mask = current_mask
            new_score = current_score
            valid_move = False
            
            if move_type < 0.5:
                # Add
                x = random.randint(0, N - 1)
                if not ((current_mask >> x) & 1):
                    if is_valid_add(current_mask, x):
                        new_mask = current_mask | (1 << x)
                        new_score = current_score + delta_3aps_add(current_mask, x)
                        valid_move = True
            elif move_type < 0.75:
                # Remove
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    new_mask = current_mask & ~(1 << x)
                    # Score change: - delta_3aps_add(new_mask, x)
                    new_score = current_score - delta_3aps_add(new_mask, x)
                    valid_move = True
            else:
                # Swap
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((current_mask >> y) & 1):
                        temp_mask = current_mask & ~(1 << x)
                        if is_valid_add(temp_mask, y):
                            d_rem = delta_3aps_add(temp_mask, x)
                            d_add = delta_3aps_add(temp_mask, y)
                            new_score = current_score - d_rem + d_add
                            new_mask = temp_mask | (1 << y)
                            valid_move = True

            if valid_move:
                if new_score >= current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        steps_since_improve = 0
                else:
                    delta = current_score - new_score
                    if delta < 1e-9:
                        current_mask = new_mask
                        current_score = new_score
                    else:
                        accept_prob = math.exp(-delta / T)
                        if random.random() < accept_prob:
                            current_mask = new_mask
                            current_score = new_score
                            steps_since_improve += 1
                        else:
                            steps_since_improve += 1
                T *= cooling
                if T < T_min:
                    T = T_min
            else:
                T *= cooling
                if T < T_min:
                    T = T_min
                
            if steps_since_improve > 1000:
                T = 1.0
                steps_since_improve = 0

        return best_mask, best_score

    best_mask_global = 0
    best_score_global = -1
    
    # Run 1: Greedy
    mask1 = 0
    while True:
        best_x = -1
        best_d = 0
        for x in range(N):
            if not ((mask1 >> x) & 1):
                if is_valid_add(mask1, x):
                    d = delta_3aps_add(mask1, x)
                    if d > best_d:
                        best_d = d
                        best_x = x
        if best_x != -1 and best_d > 0:
            mask1 |= (1 << best_x)
        else:
            break
    m, s = run_sa(mask1, deadline - 20.0)
    if s > best_score_global:
        best_score_global = s
        best_mask_global = m

    # Run 2: Random Valid
    if time.time() < deadline - 10.0:
        mask2 = 0
        for i in range(N):
            if random.random() < 0.5:
                mask2 |= (1 << i)
        changed = True
        while changed:
            changed = False
            for x in range(N):
                if (mask2 >> x) & 1:
                    for (y, z, w) in conflicts[x]:
                        if (mask2 >> y) & 1 and (mask2 >> z) & 1 and (mask2 >> w) & 1:
                            mask2 &= ~(1 << x)
                            changed = True
                            break
        m2, s2 = run_sa(mask2, deadline - 5.0)
        if s2 > best_score_global:
            best_score_global = s2
            best_mask_global = m2

    result = [i + 1 for i in range(N) if (best_mask_global >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
