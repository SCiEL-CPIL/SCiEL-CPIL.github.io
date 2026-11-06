def construct():
    import math
    import random
    
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Candidates that are diagonal-safe
    all_candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    # Build conflict graph
    conflicts = {a: set() for a in all_candidates}
    for idx, a in enumerate(all_candidates):
        for b in all_candidates[idx+1:]:
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    def try_add_fast(s, v):
        return len(conflicts[v] & s) == 0
        
    def local_search_improve(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True
                    
        return s

    # ILS (Iterated Local Search)
    random.seed(42)
    
    # Initial solution: Greedy
    curr_set = set()
    shuffled = all_candidates[:]
    random.shuffle(shuffled)
    for v in shuffled:
        if try_add_fast(curr_set, v):
            curr_set.add(v)
    curr_set = local_search_improve(curr_set)
    
    best_set = set(curr_set)
    best_size = len(curr_set)
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    max_iters = 5000
    
    for _ in range(max_iters):
        # Perturb: remove k random nodes
        k = random.randint(2, 6)
        perturb_set = set(curr_set)
        if len(perturb_set) > k:
            to_remove = random.sample(list(perturb_set), k)
            for u in to_remove:
                perturb_set.remove(u)
        else:
            # If set is small, just restart?
            perturb_set = set()
            shuffled = all_candidates[:]
            random.shuffle(shuffled)
            for v in shuffled:
                if try_add_fast(perturb_set, v):
                    perturb_set.add(v)
                    
        # Local Search on perturbed set
        new_set = local_search_improve(perturb_set)
        
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = set(new_set)
            
        # Accept/Reject based on SA
        delta = len(new_set) - len(curr_set)
        if delta >= 0:
            curr_set = new_set
        else:
            if random.random() < math.exp(delta / T):
                curr_set = new_set
                
        T *= cooling
        if T < T_min:
            break

    result = sorted(best_set)
    
    # Final verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
