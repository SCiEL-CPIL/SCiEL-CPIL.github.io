import time
import math
import random

def construct():
    n = 256
    
    # 1. Precomputation
    is_prime = [True] * (n + 1)
    if n >= 0: is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n+1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * (n + 1)
    masks[1] = 0
    for i in range(2, n + 1):
        mask = 0
        for p in primes:
            if i % p == 0:
                mask |= (1 << prime_index[p])
        masks[i] = mask
        
    numbers = list(range(n, 1, -1))
    num_count = len(numbers)
    
    # Greedy Initialization
    greedy_set = []
    greedy_mask = 0
    greedy_sum = 0
    for v in numbers:
        if (masks[v] & greedy_mask) == 0:
            greedy_set.append(v)
            greedy_mask |= masks[v]
            greedy_sum += v
            
    best_sum = greedy_sum
    best_set = greedy_set[:]
    
    start_time = time.time()
    deadline = 85.0
    
    # Local search with 1-swap and 2-swap moves
    def local_search(current_set, current_sum):
        nonlocal best_sum, best_set
        improved = True
        while improved:
            if time.time() - start_time > deadline:
                return
            improved = False
            current_mask = 0
            for v in current_set:
                current_mask |= masks[v]
            
            # Try to add a number
            for v in numbers:
                if v in current_set:
                    continue
                if (masks[v] & current_mask) == 0:
                    current_set.append(v)
                    current_sum += v
                    current_mask |= masks[v]
                    improved = True
                    break
            
            # Try 1-swap: remove one, add one
            if not improved:
                for remove_v in current_set:
                    if time.time() - start_time > deadline:
                        return
                    new_mask = current_mask & ~masks[remove_v]
                    for v in numbers:
                        if v in current_set:
                            continue
                        if (masks[v] & new_mask) == 0:
                            gain = v - remove_v
                            if gain > 0:
                                current_set.remove(remove_v)
                                current_set.append(v)
                                current_sum += gain
                                current_mask = new_mask | masks[v]
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-swap: remove one, add two
            if not improved:
                for remove_v in current_set:
                    if time.time() - start_time > deadline:
                        return
                    new_mask = current_mask & ~masks[remove_v]
                    for v1 in numbers:
                        if v1 in current_set:
                            continue
                        if (masks[v1] & new_mask) != 0:
                            continue
                        for v2 in numbers:
                            if v2 in current_set or v2 == v1:
                                continue
                            if (masks[v2] & new_mask) != 0:
                                continue
                            if (masks[v1] & masks[v2]) != 0:
                                continue
                            gain = v1 + v2 - remove_v
                            if gain > 0:
                                current_set.remove(remove_v)
                                current_set.extend([v1, v2])
                                current_sum += gain
                                current_mask = new_mask | masks[v1] | masks[v2]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
    
    local_search(best_set[:], best_sum)
    
    # B&B
    def dfs(idx, current_sum, used_mask, selected):
        nonlocal best_sum, best_set
        
        if time.time() - start_time > deadline:
            raise TimeoutError()
            
        bound_sum = current_sum
        for j in range(idx, num_count):
            if (masks[numbers[j]] & used_mask) == 0:
                bound_sum += numbers[j]
                
        if bound_sum <= best_sum:
            return
            
        if idx == num_count:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = selected[:]
            return
            
        val = numbers[idx]
        mask = masks[val]
        
        if (used_mask & mask) == 0:
            selected.append(val)
            dfs(idx + 1, current_sum + val, used_mask | mask, selected)
            selected.pop()
            
        dfs(idx + 1, current_sum, used_mask, selected)
        
    try:
        dfs(0, 0, 0, [])
    except TimeoutError:
        pass
        
    best_set.append(1)
    return best_set
