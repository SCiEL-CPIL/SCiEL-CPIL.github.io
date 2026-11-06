def construct():
    from math import gcd
    import random
    import time

    n = 256
    nums = list(range(2, n + 1))
    
    # Build conflict graph: edge if lcm(a,b) <= 256
    conflict = {a: set() for a in nums}
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            g = gcd(a, b)
            l = a * b // g
            if l <= n:
                conflict[a].add(b)
                conflict[b].add(a)
    
    weight = {a: 1.0 / a for a in nums}
    
    def greedy_init(seed):
        # Randomized greedy to get a diverse starting point
        random.seed(seed)
        remaining = nums[:]
        random.shuffle(remaining)
        selected = set()
        for a in remaining:
            if not conflict[a] & selected:
                selected.add(a)
        return selected

    def calculate_score(s):
        return sum(weight[a] for a in s)

    best_set = set()
    best_score = 0.0
    
    start_time = time.time()
    time_limit = 55.0
    
    # Run multiple SA instances
    for run in range(100):
        if time.time() - start_time > time_limit - 1.0:
            break
            
        # Initialize with greedy
        current_set = greedy_init(run)
        current_score = calculate_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing
        T = 0.05
        T_min = 0.0001
        cooling = 0.995
        
        while T > T_min:
            if time.time() - start_time > time_limit - 0.5:
                break
                
            # Choose move type
            move_type = random.choice(['swap', 'add', 'remove'])
            
            new_set = set(current_set)
            delta = 0.0
            valid_move = False
            
            if move_type == 'add':
                # Try to add a random node
                candidates = [a for a in nums if a not in current_set]
                if candidates:
                    v = random.choice(candidates)
                    if not conflict[v] & current_set:
                        new_set.add(v)
                        delta = weight[v]
                        valid_move = True
            
            elif move_type == 'remove':
                # Try to remove a random node
                if current_set:
                    u = random.choice(list(current_set))
                    new_set.remove(u)
                    delta = -weight[u]
                    valid_move = True
            
            elif move_type == 'swap':
                # Try to swap u in set with v not in set
                if current_set:
                    u = random.choice(list(current_set))
                    candidates = [a for a in nums if a not in current_set]
                    if candidates:
                        v = random.choice(candidates)
                        # Check if v is compatible with current_set - {u}
                        rest = current_set - {u}
                        if not conflict[v] & rest:
                            new_set.remove(u)
                            new_set.add(v)
                            delta = weight[v] - weight[u]
                            valid_move = True
            
            if not valid_move:
                continue
                
            # Metropolis criterion
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_set = new_set
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            
            T *= cooling
            
    return list(best_set)
