import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def can_add(S_set, reps, x):
        for y in S_set:
            if reps[x + y] >= 2:
                return False
        if reps[2 * x] >= 2:
            return False
        return True

    def apply_add(S_set, reps, x):
        S_set.add(x)
        for y in S_set:
            if y != x:
                reps[x + y] += 1
        reps[2 * x] += 1

    def apply_remove(S_set, reps, x):
        S_set.remove(x)
        for y in S_set:
            reps[x + y] -= 1
        reps[2 * x] -= 1

    def greedy_randomized():
        S_set = set()
        reps = [0] * 2001
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S_set, reps, x):
                apply_add(S_set, reps, x)
        return S_set, reps

    best_S, best_reps = greedy_randomized()
    best_size = len(best_S)
    
    while time.time() < deadline:
        # 1. Try to add
        missing = [x for x in range(1, 1001) if x not in best_S]
        if not missing:
            break
        random.shuffle(missing)
        added = False
        for x in missing:
            if can_add(best_S, best_reps, x):
                apply_add(best_S, best_reps, x)
                best_size += 1
                added = True
                break
        
        if added:
            continue

        # 2. Try swaps
        S_list = list(best_S)
        missing_list = [x for x in range(1, 1001) if x not in best_S]
        if not missing_list:
            break
            
        swap_success = False
        # Limit swap attempts to avoid wasting time
        for _ in range(100):
            if time.time() > deadline:
                break
            x = random.choice(S_list)
            y = random.choice(missing_list)
            
            # Simulate
            reps_copy = best_reps[:]
            S_copy = best_S.copy()
            
            apply_remove(S_copy, reps_copy, x)
            
            if can_add(S_copy, reps_copy, y):
                # Accept swap
                apply_remove(best_S, best_reps, x)
                apply_add(best_S, best_reps, y)
                
                # Update lists
                S_list[S_list.index(x)] = y
                missing_list[missing_list.index(y)] = x
                
                swap_success = True
                break
        
        if not swap_success:
            # Restart greedy
            new_S, new_reps = greedy_randomized()
            if len(new_S) > best_size:
                best_S, best_reps = new_S, new_reps
                best_size = len(new_S)
            else:
                # Maybe try a different random seed or just continue
                pass

    return list(best_S)
