def construct():
    import time
    from math import gcd
    import random
    
    start_time = time.time()
    DEADLINE = 85.0
    
    max_val = 256
    nums = list(range(2, max_val + 1))
    n = len(nums)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            if lcm(a, b) <= max_val:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Initial: all [129, 256]
    selected = set(range(127, 256))
    
    def score(sol):
        return sum(weights[i] for i in sol)
    
    best_score = score(selected)
    best_sol = set(selected)
    
    # Greedy add small numbers in order of decreasing weight
    small_indices = list(range(127))
    small_indices.sort(key=lambda i: -weights[i])
    
    for i in small_indices:
        if time.time() > DEADLINE:
            break
        if not (conflicts[i] & selected):
            selected.add(i)
            new_score = score(selected)
            if new_score > best_score:
                best_score = new_score
                best_sol = set(selected)
    
    # Local search with hill climbing
    random.seed(42)
    current = set(best_sol)
    cur_score = best_score
    
    for _ in range(100000):
        if time.time() > DEADLINE:
            break
        
        improved = False
        
        # Try single additions
        for i in range(n):
            if i not in current and not (conflicts[i] & current):
                current.add(i)
                cur_score += weights[i]
                improved = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_sol = set(current)
                break
        
        if not improved:
            # Try single swaps
            if current:
                rem_list = list(current)
                random.shuffle(rem_list)
                for rem in rem_list:
                    if time.time() > DEADLINE:
                        break
                    temp = current - {rem}
                    # Find best add
                    best_add = -1
                    best_new_score = cur_score - weights[rem]
                    for add in range(n):
                        if add not in temp and not (conflicts[add] & temp):
                            ns = cur_score - weights[rem] + weights[add]
                            if ns > best_new_score:
                                best_new_score = ns
                                best_add = add
                    if best_add >= 0:
                        current = temp
                        current.add(best_add)
                        cur_score = best_new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_sol = set(current)
                        break
        
        if not improved:
            # Perturbation
            if len(current) > 1:
                rem = random.choice(list(current))
                current.remove(rem)
                cur_score -= weights[rem]
    
    # Final greedy pass
    for i in range(n):
        if time.time() > DEADLINE:
            break
        if i not in best_sol and not (conflicts[i] & best_sol):
            best_sol.add(i)
            best_score = score(best_sol)
    
    return [nums[i] for i in best_sol]
