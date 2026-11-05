import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    required = [1, 2, 5, 11]
    max_val = 284
    required_set = set(required)
    
    # Initial sums
    initial_sums = [2, 3, 6, 12, 4, 7, 13, 10, 16, 22]
    
    def make_mask(sums):
        mask = 0
        for s in sums:
            mask |= (1 << s)
        return mask
    
    initial_mask = make_mask(initial_sums)
    
    best_set = required[:]
    best_size = len(best_set)
    
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    def can_add(x, curr_set, mask):
        if (mask >> (2 * x)) & 1:
            return False
        for y in curr_set:
            if (mask >> (x + y)) & 1:
                return False
        return True
    
    def greedy_fill(curr_set, mask, pool, order_key=None):
        if order_key is not None:
            pool = sorted(pool, key=order_key)
        for x in pool:
            if x in curr_set:
                continue
            if can_add(x, curr_set, mask):
                mask |= (1 << (2 * x))
                for y in curr_set:
                    mask |= (1 << (x + y))
                curr_set.append(x)
        return curr_set, mask
    
    def compute_mask(s):
        mask = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                mask |= (1 << (s[i] + s[j]))
        return mask
    
    # Try various greedy strategies
    def run_greedy_variants():
        nonlocal best_set, best_size
        
        # Ascending
        curr = required[:]
        mask = initial_mask
        curr, mask = greedy_fill(curr, mask, candidates[:], lambda x: x)
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
        
        # Descending
        curr = required[:]
        mask = initial_mask
        curr, mask = greedy_fill(curr, mask, candidates[:], lambda x: -x)
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
        
        # Random restarts - many iterations
        for _ in range(2000):
            if time.time() > deadline:
                break
            curr = required[:]
            mask = initial_mask
            pool = candidates[:]
            random.shuffle(pool)
            curr, mask = greedy_fill(curr, mask, pool)
            if len(curr) > best_size:
                best_size = len(curr)
                best_set = curr[:]
            if best_size >= 20:
                break
    
    # Local search: try to improve by swapping
    def local_search(base, max_iters=300):
        nonlocal best_set, best_size
        nonlocal deadline
        
        curr = base[:]
        curr_mask = compute_mask(curr)
        
        improved = True
        it = 0
        while improved and it < max_iters:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            removable = [x for x in curr if x not in required_set]
            if not removable:
                break
            random.shuffle(removable)
            
            for rem in removable:
                if time.time() > deadline:
                    break
                
                # Try removing rem and finding best replacement
                new_list = [x for x in curr if x != rem]
                new_set = set(new_list)
                
                # Compute new mask
                new_mask = 0
                for i in range(len(new_list)):
                    for j in range(i, len(new_list)):
                        new_mask |= (1 << (new_list[i] + new_list[j]))
                
                # Find best candidate to add
                best_x = None
                best_score = -1
                for x in candidates:
                    if x in new_set:
                        continue
                    if can_add(x, new_list, new_mask):
                        # Heuristic: prefer x that minimizes future conflicts
                        # Simple: prefer larger x
                        score = x
                        if score > best_score:
                            best_score = score
                            best_x = x
                
                if best_x is not None:
                    # Try this replacement
                    test = new_list + [best_x]
                    test_set = new_set | {best_x}
                    test_mask = new_mask
                    test_mask |= (1 << (2 * best_x))
                    for y in new_list:
                        test_mask |= (1 << (best_x + y))
                    
                    # Greedy fill
                    pool = [x for x in candidates if x not in test_set]
                    random.shuffle(pool)
                    for x in pool:
                        if can_add(x, test, test_mask):
                            test_mask |= (1 << (2 * x))
                            for y in test:
                                test_mask |= (1 << (x + y))
                            test.append(x)
                    
                    if len(test) > len(curr):
                        curr = test
                        curr_mask = test_mask
                        improved = True
                        if len(curr) > best_size:
                            best_size = len(curr)
                            best_set = curr[:]
                        break
                    elif len(test) == len(curr) and random.random() < 0.2:
                        curr = test
                        curr_mask = test_mask
                        improved = True
                        break
        
        return curr
    
    run_greedy_variants()
    
    if best_size >= 20:
        return sorted(best_set)
    
    # Local search
    while time.time() < deadline:
        if best_size >= 20:
            break
        
        improved = local_search(best_set, max_iters=150)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
        
        # Fresh greedy + local search
        curr = required[:]
        mask = initial_mask
        pool = candidates[:]
        random.shuffle(pool)
        curr, mask = greedy_fill(curr, mask, pool)
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
        
        improved = local_search(curr, max_iters=80)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    return sorted(best_set)
