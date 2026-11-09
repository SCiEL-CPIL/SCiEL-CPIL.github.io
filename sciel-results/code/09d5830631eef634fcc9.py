import random
import time
from itertools import combinations

def construct():
    n = 16
    all_sets = []
    for c in combinations(range(1, n + 1), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    # Initial Greedy
    random.shuffle(all_sets)
    family = []
    for S in all_sets:
        valid = True
        for i in range(len(family)):
            A = family[i]
            inter_AS = A & S
            for j in range(i + 1, len(family)):
                B = family[j]
                inter_AB = A & B
                if inter_AB == inter_AS:
                    inter_BS = B & S
                    if inter_AB == inter_BS:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            family.append(S)
            
    current_family = set(family)
    current_size = len(family)
    best_family = set(family)
    best_size = current_size
    
    start_time = time.time()
    max_time = 55.0
    T = 1.0
    cooling = 0.9995
    
    def can_add(S, F):
        F_list = list(F)
        for i in range(len(F_list)):
            A = F_list[i]
            inter_AS = A & S
            for j in range(i + 1, len(F_list)):
                B = F_list[j]
                inter_AB = A & B
                if inter_AB == inter_AS:
                    inter_BS = B & S
                    if inter_AB == inter_BS:
                        return False
        return True

    def find_blocker(S, F):
        F_list = list(F)
        for i in range(len(F_list)):
            A = F_list[i]
            inter_AS = A & S
            for j in range(i + 1, len(F_list)):
                B = F_list[j]
                inter_AB = A & B
                if inter_AB == inter_AS:
                    inter_BS = B & S
                    if inter_AB == inter_BS:
                        return A
        return None

    while time.time() - start_time < max_time:
        S = random.choice(all_sets)
        
        if S in current_family:
            # Small chance to remove to explore
            if random.random() < 0.05:
                new_family = current_family - {S}
                new_size = current_size - 1
                # Accept with probability exp(-1/T)
                if random.random() < (2.718281828 ** (-1.0 / T)):
                    current_family = new_family
                    current_size = new_size
            continue
            
        if can_add(S, current_family):
            current_family.add(S)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_family = set(current_family)
        else:
            blocker = find_blocker(S, current_family)
            if blocker is not None:
                base_family = current_family - {blocker}
                if can_add(S, base_family):
                    # Swap
                    current_family = base_family
                    current_family.add(S)
                    # Size unchanged
                    # Accept to move
                    pass
        
        T *= cooling

    result = []
    for mask in best_family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
