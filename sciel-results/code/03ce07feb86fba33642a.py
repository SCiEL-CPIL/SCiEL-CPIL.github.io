import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    fixed = {1, 2, 5, 11}
    candidates = [x for x in range(1, 285) if x not in fixed]
    
    def get_initial_sums(S):
        sums = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def can_add(x, S, sums):
        if x in S:
            return False
        if x + x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    def add_element(x, S, sums):
        S.add(x)
        sums.add(x + x)
        for y in S:
            if y != x:
                sums.add(x + y)

    def remove_element(x, S, sums):
        S.remove(x)
        sums.remove(x + x)
        for y in S:
            sums.remove(x + y)

    # 1. Greedy Initialization
    best_set = set(fixed)
    best_size = len(best_set)
    
    for _ in range(100):
        S = set(fixed)
        sums = get_initial_sums(S)
        cand = candidates[:]
        random.shuffle(cand)
        
        for x in cand:
            if can_add(x, S, sums):
                add_element(x, S, sums)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 2. Simulated Annealing
    current_set = set(best_set)
    current_sums = get_initial_sums(current_set)
    current_size = len(current_set)
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Make copies
        new_set = set(current_set)
        new_sums = get_initial_sums(new_set)
        new_size = len(new_set)
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random element
            x = random.choice(candidates)
            if can_add(x, new_set, new_sums):
                add_element(x, new_set, new_sums)
                new_size += 1
            else:
                # Try a kick: remove 1-2 random non-fixed elements
                removable = [x for x in new_set if x not in fixed]
                if removable:
                    size_before = new_size
                    num_remove = random.randint(1, 2)
                    to_remove = random.sample(removable, min(num_remove, len(removable)))
                    
                    for u in to_remove:
                        remove_element(u, new_set, new_sums)
                        new_size -= 1
                    
                    # Repack greedily
                    cand = [x for x in candidates if x not in new_set]
                    random.shuffle(cand)
                    for x in cand:
                        if can_add(x, new_set, new_sums):
                            add_element(x, new_set, new_sums)
                            new_size += 1
        else:
            # Try a swap
            removable = [x for x in new_set if x not in fixed]
            if removable:
                u = random.choice(removable)
                remove_element(u, new_set, new_sums)
                new_size -= 1
                
                cand = [x for x in candidates if x not in new_set]
                random.shuffle(cand)
                added = False
                for x in cand:
                    if can_add(x, new_set, new_sums):
                        add_element(x, new_set, new_sums)
                        new_size += 1
                        added = True
                        break
                if not added:
                    # Revert swap
                    add_element(u, new_set, new_sums)
                    new_size += 1

        # Acceptance
        delta = new_size - current_size
        if delta >= 0:
            current_set = new_set
            current_sums = new_sums
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
        else:
            if random.random() < pow(2.71828, delta / T):
                current_set = new_set
                current_sums = new_sums
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)

        T *= cooling
        if T < 0.01:
            T = 1.0

    return list(best_set)

