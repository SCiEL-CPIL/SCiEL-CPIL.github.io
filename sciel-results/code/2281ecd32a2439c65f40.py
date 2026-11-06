import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Precomputation
    # 1. Find all 4-APs
    four_aps = []
    for a in range(1, N - 2):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # Map each element to the other 3 elements in 4-APs it belongs to
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for ap in four_aps:
        for i in range(4):
            others = tuple(ap[j] for j in range(4) if j != i)
            four_ap_neighbors[ap[i]].append(others)

    # 2. Find all 3-APs and map to elements
    # We need to calculate the gain in 3-APs when adding x.
    # Gain(x) = count of (a,c) in A such that a+c=2x 
    #          + count of (b,c) in A such that 2b=x+c 
    #          + count of (a,b) in A such that 2b=a+x
    
    # Precompute potential partners for each x
    # mid_partners[x]: list of (a, c) such that a+c=2x, a < c
    # start_partners[x]: list of (b, c) such that 2b=x+c, b < c
    # end_partners[x]: list of (a, b) such that 2b=a+x, a < b
    
    mid_partners = {i: [] for i in range(1, N + 1)}
    start_partners = {i: [] for i in range(1, N + 1)}
    end_partners = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2): # a+c must be even
            s = a + c
            x = s // 2
            if 1 <= x <= N and x > a:
                mid_partners[x].append((a, c))
                
    for x in range(1, N + 1):
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                start_partners[x].append((b, c))
                
    for x in range(1, N + 1):
        for b in range(1, x):
            a = 2 * b - x
            if a > 0 and a < b:
                end_partners[x].append((a, b))

    # Initial Solution: Greedy
    A = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        valid = True
        for others in four_ap_neighbors[x]:
            if all(o in A for o in others):
                valid = False
                break
        if valid:
            A.add(x)
            
    def count_3aps(A_set):
        count = 0
        for x in A_set:
            for a, c in mid_partners[x]:
                if a in A_set and c in A_set:
                    count += 1
            for b, c in start_partners[x]:
                if b in A_set and c in A_set:
                    count += 1
            for a, b in end_partners[x]:
                if a in A_set and b in A_set:
                    count += 1
        return count

    current_score = count_3aps(A)
    best_A = set(A)
    best_score = current_score
    
    # Local Search with Simulated Annealing
    T = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        # Options: Add, Remove, Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_A = set(A)
        gain = 0
        
        if move_type == 'add':
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates: continue
            x = random.choice(candidates)
            
            # Check validity
            valid = True
            for others in four_ap_neighbors[x]:
                if all(o in new_A for o in others):
                    valid = False
                    break
            
            if valid:
                # Calculate gain
                for a, c in mid_partners[x]:
                    if a in new_A and c in new_A:
                        gain += 1
                for b, c in start_partners[x]:
                    if b in new_A and c in new_A:
                        gain += 1
                for a, b in end_partners[x]:
                    if a in new_A and b in new_A:
                        gain += 1
                
                new_A.add(x)
                new_score = current_score + gain
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(new_A)
                
                # Accept/Reject
                if gain >= 0:
                    A = new_A
                    current_score = new_score
                else:
                    if random.random() < math.exp(gain / T):
                        A = new_A
                        current_score = new_score
                        
        elif move_type == 'remove':
            if not A: continue
            x = random.choice(list(A))
            new_A.remove(x)
            
            # Calculate loss (negative gain)
            loss = 0
            for a, c in mid_partners[x]:
                if a in new_A and c in new_A:
                    loss += 1
            for b, c in start_partners[x]:
                if b in new_A and c in new_A:
                    loss += 1
            for a, b in end_partners[x]:
                if a in new_A and b in new_A:
                    loss += 1
            
            new_score = current_score - loss
            if new_score > best_score:
                best_score = new_score
                best_A = set(new_A)
            
            if loss <= 0:
                A = new_A
                current_score = new_score
            else:
                if random.random() < math.exp(-loss / T):
                    A = new_A
                    current_score = new_score
                    
        elif move_type == 'swap':
            if not A: continue
            x = random.choice(list(A))
            candidates = [y for y in range(1, N + 1) if y not in A]
            if not candidates: continue
            y = random.choice(candidates)
            
            new_A.remove(x)
            
            # Check if adding y is valid in new_A
            valid = True
            for others in four_ap_neighbors[y]:
                if all(o in new_A for o in others):
                    valid = False
                    break
            
            if valid:
                # Calculate net gain
                # Loss from removing x
                loss_x = 0
                for a, c in mid_partners[x]:
                    if a in new_A and c in new_A:
                        loss_x += 1
                for b, c in start_partners[x]:
                    if b in new_A and c in new_A:
                        loss_x += 1
                for a, b in end_partners[x]:
                    if a in new_A and b in new_A:
                        loss_x += 1
                        
                # Gain from adding y
                gain_y = 0
                for a, c in mid_partners[y]:
                    if a in new_A and c in new_A:
                        gain_y += 1
                for b, c in start_partners[y]:
                    if b in new_A and c in new_A:
                        gain_y += 1
                for a, b in end_partners[y]:
                    if a in new_A and b in new_A:
                        gain_y += 1
                        
                net_gain = gain_y - loss_x
                new_A.add(y)
                new_score = current_score + net_gain
                
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(new_A)
                
                if net_gain >= 0:
                    A = new_A
                    current_score = new_score
                else:
                    if random.random() < math.exp(net_gain / T):
                        A = new_A
                        current_score = new_score

        T *= cooling
        if T < 0.01:
            T = 0.01
            # Optional: restart or just keep going
            # For now, just keep going with low T (hill climbing)
            
    return sorted(best_A)

# EVOLVE-BLOCK-END
