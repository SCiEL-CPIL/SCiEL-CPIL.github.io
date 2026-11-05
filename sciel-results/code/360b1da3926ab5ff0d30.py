import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # 1. Greedy Removal Initialization
    current_set = set(range(1, 101))
    
    def find_subset_summing_to(S, target):
        if not S:
            return None
        S_list = list(S)
        random.shuffle(S_list)
        reachable = {0: (None, None)}
        for x in S_list:
            new_reachable = dict(reachable)
            for s, (prev_x, prev_s) in reachable.items():
                if s + x <= target:
                    new_reachable[s + x] = (x, s)
            reachable = new_reachable
        if target in reachable:
            subset = []
            s = target
            while s > 0:
                x, prev_s = reachable[s]
                subset.append(x)
                s = prev_s
            return subset
        return None

    while time.time() < deadline - 10.0:
        subset = find_subset_summing_to(current_set, 200)
        if subset is None:
            break
        max_elem = max(subset)
        current_set.remove(max_elem)
        
    best_set = current_set.copy()
    best_size = len(best_set)
    
    def compute_reachable(s):
        r = 1
        for x in s:
            r |= (r << x)
        return r

    current_reachable = compute_reachable(best_set)
    
    all_elements = list(range(1, 101))
    
    while time.time() < deadline:
        improved = False
        
        # Try to add a random element not in set
        candidates = [x for x in all_elements if x not in best_set]
        if candidates:
            random.shuffle(candidates)
            for x in candidates[:20]: 
                # Fast check using current_reachable
                # new_reach = current_reachable | (current_reachable << x)
                # if not (new_reach >> 200) & 1:
                if not ((current_reachable | (current_reachable << x)) >> 200) & 1:
                    best_set.add(x)
                    best_size += 1
                    current_reachable = current_reachable | (current_reachable << x)
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try to swap: remove one, add two
        # This requires recomputing reachable
        if best_size > 0:
            remove_x = random.choice(list(best_set))
            candidates = [x for x in all_elements if x not in best_set]
            if len(candidates) >= 2:
                for _ in range(10):
                    add_pair = random.sample(candidates, 2)
                    new_set = best_set - {remove_x}
                    new_set.update(add_pair)
                    # Recompute reachable for new_set
                    r = 1
                    valid = True
                    for y in new_set:
                        r |= (r << y)
                        if (r >> 200) & 1:
                            valid = False
                            break
                    if valid:
                        best_set = new_set
                        best_size = len(best_set)
                        current_reachable = r
                        improved = True
                        break
        
        if not improved:
            # 1-for-1 swap
            if best_size > 0:
                remove_x = random.choice(list(best_set))
                candidates = [x for x in all_elements if x not in best_set]
                if candidates:
                    for _ in range(5):
                        add_x = random.choice(candidates)
                        new_set = best_set - {remove_x}
                        new_set.add(add_x)
                        r = 1
                        valid = True
                        for y in new_set:
                            r |= (r << y)
                            if (r >> 200) & 1:
                                valid = False
                                break
                        if valid and random.random() < 0.5:
                            best_set = new_set
                            best_size = len(best_set)
                            current_reachable = r
                            improved = True
                            break
            if not improved:
                pass

    return list(best_set)
