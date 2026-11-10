import random
import time

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Greedy Initialization
    chosen_set = set()
    counts_dict = {}
    
    for x in range(1, N + 1):
        ok = True
        for y in chosen_set:
            s = x + y
            if counts_dict.get(s, 0) >= 2:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts_dict.get(s, 0) >= 2:
                ok = False
        
        if ok:
            chosen_set.add(x)
            for y in chosen_set:
                if y == x:
                    continue
                s = x + y
                counts_dict[s] = counts_dict.get(s, 0) + 1
            s = 2 * x
            counts_dict[s] = counts_dict.get(s, 0) + 1

    # Local Search Helpers
    def try_add(x):
        if x in chosen_set:
            return False
        ok = True
        for y in chosen_set:
            s = x + y
            if counts_dict.get(s, 0) >= 2:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts_dict.get(s, 0) >= 2:
                ok = False
        
        if not ok:
            return False
        
        chosen_set.add(x)
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            counts_dict[s] = counts_dict.get(s, 0) + 1
        s = 2 * x
        counts_dict[s] = counts_dict.get(s, 0) + 1
        return True

    def try_remove(x):
        if x not in chosen_set:
            return False
        for y in chosen_set:
            if y == x:
                continue
            s = x + y
            counts_dict[s] -= 1
            if counts_dict[s] == 0:
                del counts_dict[s]
        s = 2 * x
        counts_dict[s] -= 1
        if counts_dict[s] == 0:
            del counts_dict[s]
        
        chosen_set.remove(x)
        return True

    def try_swap(x_out, x_in):
        if x_out not in chosen_set or x_in in chosen_set:
            return False
        
        old_sums = []
        for y in chosen_set:
            if y == x_out:
                continue
            s = x_out + y
            old_sums.append(s)
            counts_dict[s] -= 1
            if counts_dict[s] == 0:
                del counts_dict[s]
        s = 2 * x_out
        old_sums.append(s)
        counts_dict[s] -= 1
        if counts_dict[s] == 0:
            del counts_dict[s]
        
        chosen_set.remove(x_out)
        
        if not try_add(x_in):
            chosen_set.add(x_out)
            for s in old_sums:
                counts_dict[s] = counts_dict.get(s, 0) + 1
            return False
        
        return True

    # Search Loop
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # 1. Try to add
        available = [x for x in range(1, N + 1) if x not in chosen_set]
        if available:
            added = False
            attempts = min(100, len(available))
            for _ in range(attempts):
                x = random.choice(available)
                if try_add(x):
                    added = True
                    no_improve = 0
                    break
            if added:
                continue
        
        # 2. Try swap
        if chosen_set:
            x_out = random.choice(list(chosen_set))
            available = [x for x in range(1, N + 1) if x not in chosen_set]
            if available:
                x_in = random.choice(available)
                if try_swap(x_out, x_in):
                    no_improve = 0
                    continue
        
        # 3. Kick: Remove random element
        if chosen_set and no_improve > 100:
            x_rem = random.choice(list(chosen_set))
            try_remove(x_rem)
            no_improve = 0
            continue

        no_improve += 1
        if no_improve > max_no_improve:
            break

    return list(chosen_set)
