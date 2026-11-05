"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 512

def _precompute_conflicts():
    """
    Precompute conflict masks for each element.
    conflicts[x] is a list of masks. Each mask is a 512-bit integer representing
    a set of 3 elements {a, b, c} such that {x, a, b, c} forms a 4-term AP.
    If x is in S, then for all masks in conflicts[x], (S & mask) != mask must hold.
    """
    conflicts = [[] for _ in range(N + 1)]
    
    # Generate all 4-term APs: p, p+d, p+2d, p+3d
    for p in range(1, N + 1):
        max_d = (N - p) // 3
        for d in range(1, max_d + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            
            # For each element, the other 3 form a forbidden mask
            # If t1 is in S, {t2, t3, t4} cannot all be in S.
            mask1 = (1 << t2) | (1 << t3) | (1 << t4)
            conflicts[t1].append(mask1)
            
            mask2 = (1 << t1) | (1 << t3) | (1 << t4)
            conflicts[t2].append(mask2)
            
            mask3 = (1 << t1) | (1 << t2) | (1 << t4)
            conflicts[t3].append(mask3)
            
            mask4 = (1 << t1) | (1 << t2) | (1 << t3)
            conflicts[t4].append(mask4)
            
    return conflicts

def construct():
    random.seed(42)
    conflicts = _precompute_conflicts()
    
    # S is a bitmask of the current set
    S = 0
    current_size = 0
    S_list = []
    
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

    # 1. Greedy Initialization
    best_S = 0
    best_size = 0
    best_list = []
    
    for _ in range(50):
        current_S = 0
        current_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]
            
    S = best_S
    S_list = best_list[:]
    current_size = best_size
    best_size_found = best_size
    best_S_found = best_S
    best_list_found = best_list[:]

    # 2. Simulated Annealing
    time_limit = 55.0
    start_time = time.time()
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.9999
    
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < T_min:
            T = T_min

        move_type = random.random()
        
        if move_type < 0.4:
            # Try to Add
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    add_element(x)
                    if current_size > best_size_found:
                        best_size_found = current_size
                        best_S_found = S
                        best_list_found = S_list[:]
        
        elif move_type < 0.8:
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
        
        else:
            # Try to Remove
            if S_list:
                x = random.choice(S_list)
                # Accept removal with probability exp(-1/T)
                if random.random() < 2.71828 ** (-1.0 / T):
                    remove_element(x)
                    
        # Periodic Greedy Fill
        if random.random() < 0.01:
             candidates = [x for x in all_elements if not (S & (1 << x))]
             random.shuffle(candidates)
             for x in candidates:
                 if can_add(x, S):
                     add_element(x)
                     if current_size > best_size_found:
                         best_size_found = current_size
                         best_S_found = S
                         best_list_found = S_list[:]

    return sorted(best_list_found)

# EVOLVE-BLOCK-END
