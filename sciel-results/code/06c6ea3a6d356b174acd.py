"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 512

def _precompute_conflicts():
    conflicts = [[] for _ in range(N + 1)]
    for p in range(1, N + 1):
        max_d = (N - p) // 3
        for d in range(1, max_d + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            
            mask1 = (1 << t2) | (1 << t3) | (1 << t4)
            conflicts[t1].append(mask1)
            
            mask2 = (1 << t1) | (1 << t3) | (1 << t4)
            conflicts[t2].append(mask2)
            
            mask3 = (1 << t1) | (1 << t2) | (1 << t4)
            conflicts[t3].append(mask3)
            
            mask4 = (1 << t1) | (1 << t2) | (1 << t3)
            conflicts[t4].append(mask4)
            
    return conflicts

def _base3_set(N):
    """Generate set of numbers in [1, N] with no '2' in base-3 representation."""
    s = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            s.add(i)
    return s

def construct():
    random.seed(42)
    conflicts = _precompute_conflicts()
    
    # Start with Base-3 set
    initial_set = _base3_set(N)
    S = 0
    S_list = []
    for x in initial_set:
        S |= (1 << x)
        S_list.append(x)
    
    current_size = len(S_list)
    best_size = current_size
    best_S = S
    best_list = S_list[:]
    
    def can_add(x, S_mask):
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def add_element(x):
        nonlocal S, S_list, current_size
        S |= (1 << x)
        S_list.append(x)
        current_size += 1

    def remove_element(x):
        nonlocal S, S_list, current_size
        S &= ~(1 << x)
        idx = S_list.index(x)
        S_list.pop(idx)
        current_size -= 1

    time_limit = 55.0
    start_time = time.time()
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.99995
    
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < T_min:
            T = T_min

        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    add_element(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S
                        best_list = S_list[:]
        
        elif move_type < 0.9:
            # Try to Swap
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    S_wo_x = S & ~(1 << x)
                    if can_add(y, S_wo_x):
                        S = S_wo_x | (1 << y)
                        idx = S_list.index(x)
                        S_list[idx] = y
                        # Size doesn't change, so no best update needed unless we track something else
                        # But wait, if we swap, size is same. 
                        # However, the structure changes.
        
        else:
            # Try to Remove
            if S_list:
                x = random.choice(S_list)
                if random.random() < 2.71828 ** (-1.0 / T):
                    remove_element(x)
                    
        # Periodic Greedy Fill (try to add all possible)
        if random.random() < 0.05:
             candidates = [x for x in all_elements if not (S & (1 << x))]
             random.shuffle(candidates)
             for x in candidates:
                 if can_add(x, S):
                     add_element(x)
                     if current_size > best_size:
                         best_size = current_size
                         best_S = S
                         best_list = S_list[:]

    return sorted(best_list)

# EVOLVE-BLOCK-END
