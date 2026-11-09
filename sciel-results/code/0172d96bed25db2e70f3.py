import time
import random
from math import gcd

N = 128

# Precompute LCM table
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for a in range(1, N + 1):
    for b in range(1, N + 1):
        lcm_table[a][b] = a * b // gcd(a, b)

# Precompute conflict pairs
# conflict_pairs[x] is a list of tuples (a, b) such that {x, a, b} is a bad triple
conflict_pairs = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    for a in range(1, N + 1):
        if a == x:
            continue
        for b in range(a + 1, N + 1):
            if b == x:
                continue
            l1 = lcm_table[x][a]
            l2 = lcm_table[x][b]
            l3 = lcm_table[a][b]
            if l1 == l2 == l3:
                conflict_pairs[x].append((a, b))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_size = 0
    best_set = []
    
    random.seed(42)
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected_mask = 0
        selected_list = []
        
        # Greedy Phase
        for x in candidates:
            can_add = True
            for a, b in conflict_pairs[x]:
                if (selected_mask >> a) & 1:
                    if (selected_mask >> b) & 1:
                        can_add = False
                        break
            if can_add:
                selected_mask |= (1 << x)
                selected_list.append(x)
        
        current_size = len(selected_list)
        
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle selected_list to randomize which y we try to remove first?
            # Or just iterate.
            # Let's iterate through indices.
            for i in range(len(selected_list)):
                y = selected_list[i]
                temp_mask = selected_mask & ~(1 << y)
                
                found_swap = False
                # Try to add any x not in temp_mask
                # Use a shuffled order for x to avoid bias?
                # Just checking 1..128 is fine.
                for x in range(1, N + 1):
                    if (temp_mask >> x) & 1:
                        continue
                    if x == y:
                        continue
                    
                    can_add = True
                    for a, b in conflict_pairs[x]:
                        if (temp_mask >> a) & 1:
                            if (temp_mask >> b) & 1:
                                can_add = False
                                break
                    if can_add:
                        selected_mask = temp_mask | (1 << x)
                        selected_list[i] = x
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected_list)
            
    if not best_set:
        return [1]
        
    return best_set
