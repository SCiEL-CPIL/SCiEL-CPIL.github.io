import time
import random

def construct():
    start = time.time()
    deadline = start + 145.0
    
    prescribed = (1, 2, 5, 11)
    
    def build_sums(a):
        s = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s.add(a[i] + a[j])
        return s
    
    def can_add(x, a, sums_used):
        for y in a:
            if x + y in sums_used:
                return False
        if 2 * x in sums_used:
            return False
        return True
    
    def greedy_extend(a, sums_used, order=None):
        a = list(a)
        sums_used = set(sums_used)
        a_set = set(a)
        
        if order is None:
            candidates = list(range(12, 285))
        else:
            candidates = [x for x in order if x not in a_set]
        
        for x in candidates:
            if x in a_set:
                continue
            if can_add(x, a, sums_used):
                a.append(x)
                a_set.add(x)
                for y in a[:-1]:
                    sums_used.add(x + y)
                sums_used.add(2 * x)
        return a, sums_used
    
    def local_search(a, sums_used, max_iters=30000):
        a = list(a)
        sums_used = set(sums_used)
        a_set = set(a)
        n = len(a)
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            # Try to add a random element
            for _ in range(50):
                x = random.randint(1, 284)
                if x in a_set:
                    continue
                if can_add(x, a, sums_used):
                    a.append(x)
                    a_set.add(x)
                    for y in a[:-1]:
                        sums_used.add(x + y)
                    sums_used.add(2 * x)
                    n += 1
                    break
        
        return a, sums_used
    
    best = list(prescribed)
    best_len = len(best)
    
    for restart in range(200):
        if time.time() >= deadline - 5:
            break
        
        a = list(prescribed)
        sums_used = build_sums(a)
        
        # Vary greedy order
        if restart % 3 == 0:
            a, sums_used = greedy_extend(a, sums_used)
        elif restart % 3 == 1:
            order = list(range(12, 285))
            random.shuffle(order)
            a, sums_used = greedy_extend(a, sums_used, order)
        else:
            order = list(range(284, 11, -1))
            a, sums_used = greedy_extend(a, sums_used, order)
        
        a, sums_used = local_search(a, sums_used, max_iters=10000)
        
        if len(a) > best_len:
            best = a
            best_len = len(a)
    
    # Final push
    if time.time() < deadline - 1:
        a_final, sums_final = local_search(best, build_sums(best), max_iters=30000)
        if len(a_final) > best_len:
            best = a_final
    
    return best
