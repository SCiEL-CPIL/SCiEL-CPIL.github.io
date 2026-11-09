import time
import random

def _can_add(S, x):
    """
    Check if x can be added to set S without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c in S_new.
    New triple must involve x. Let triple be {x, a, b} with a, b in S.
    Sums: x+a, x+b, a+b.
    Must have x+a in S_new, x+b in S_new, a+b in S_new.
    Since x not in S:
    x+a in S_new => x+a in S (as x+a != x)
    x+b in S_new => x+b in S
    a+b in S_new => a+b in S or a+b == x
    """
    candidates = []
    for a in S:
        if x + a in S:
            candidates.append(a)
            
    n = len(candidates)
    if n < 2:
        return True
        
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if a + b in S or a + b == x:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n_max = 128
    all_nums = set(range(1, n_max + 1))
    
    best_set = set()
    best_size = 0
    
    # Initial greedy construction with random orders
    for _ in range(20):
        if time.time() > deadline - 60:
            break
            
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        
        current_set = set()
        
        for cand in candidates:
            if _can_add(current_set, cand):
                current_set.add(cand)
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    # Iterated Local Search
    current_set = set(best_set)
    
    while time.time() < deadline - 5:
        improved = False
        missing = list(all_nums - current_set)
        random.shuffle(missing)
        
        for cand in missing:
            if time.time() > deadline - 5:
                break
            if _can_add(current_set, cand):
                current_set.add(cand)
                improved = True
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        if not improved:
            # Kick: remove random elements to escape local optimum
            if len(current_set) > 10:
                k = random.randint(2, 5)
                to_remove = random.sample(list(current_set), k=k)
                for x in to_remove:
                    current_set.remove(x)
            else:
                # If set is small, just restart greedy?
                # Or just break?
                # Let's try to add more
                pass

    # Final polish
    missing = list(all_nums - best_set)
    for cand in missing:
        if time.time() > deadline - 1:
            break
        if _can_add(best_set, cand):
            best_set.add(cand)
            
    result = sorted(best_set)
    return [int(x) for x in result]
