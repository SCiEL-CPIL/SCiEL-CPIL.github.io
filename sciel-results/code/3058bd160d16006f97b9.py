# EVOLVE-BLOCK-START
import time
import random

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    deadline = time.time() + 85
    n = 256
    all_nums = list(range(2, n + 1))
    
    # Precompute prime factors
    pf = {x: _prime_factors(x) for x in all_nums}
    
    # Precompute conflict structure
    # For each pair (a,b), determine which c values would create a conflict {a,b,c}
    # conflict_c[a][b] = set of c values such that a|b*c or b|a*c or c|a*b
    
    # For efficiency, precompute for each vertex v, the set of pairs (a,b) that conflict with v
    # conflict_pairs[v] = set of (min(a,b), max(a,b)) where {v,a,b} is a conflict
    
    num_v = len(all_nums)
    val_to_idx = {v: i for i, v in enumerate(all_nums)}
    idx_to_val = all_nums
    
    conflict_pairs = [set() for _ in range(num_v)]
    
    # Compute all conflict triples
    for i in range(num_v):
        a = all_nums[i]
        for j in range(i+1, num_v):
            b = all_nums[j]
            for k in range(j+1, num_v):
                c = all_nums[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    conflict_pairs[i].add((j, k))
                    conflict_pairs[j].add((i, k))
                    conflict_pairs[k].add((i, j))
    
    # Function to check if a set of indices is valid
    def _is_valid(indices):
        idx_set = set(indices)
        for i in idx_set:
            for (j, k) in conflict_pairs[i]:
                if j in idx_set and k in idx_set:
                    return False
        return True
    
    # Function to check if adding v to current set is valid
    def _can_add(current_set, v):
        for (j, k) in conflict_pairs[v]:
            if j in current_set and k in current_set:
                return False
        return True
    
    # Greedy construction with different orderings
    def _greedy(order):
        current = set()
        for v in order:
            if _can_add(current, v):
                current.add(v)
        return current
    
    # Try multiple greedy strategies
    best_set = set()
    
    # Strategy 1: descending order
    order1 = sorted(range(num_v), key=lambda i: all_nums[i], reverse=True)
    s1 = _greedy(order1)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: ascending order  
    order2 = sorted(range(num_v), key=lambda i: all_nums[i])
    s2 = _greedy(order2)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: by number of prime factors (fewer first)
    order3 = sorted(range(num_v), key=lambda i: len(pf[all_nums[i]]))
    s3 = _greedy(order3)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Strategy 4: by conflict degree (fewer conflicts first)
    order4 = sorted(range(num_v), key=lambda i: len(conflict_pairs[i]))
    s4 = _greedy(order4)
    if len(s4) > len(best_set):
        best_set = s4
    
    # Local search: try to improve by swapping or adding
    def _local_search(current_set, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to add a vertex
            for v in range(num_v):
                if v not in current_set and _can_add(current_set, v):
                    current_set.add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try to swap: remove one, add one
            for v_remove in list(current_set):
                temp_set = current_set - {v_remove}
                for v_add in range(num_v):
                    if v_add not in temp_set and _can_add(temp_set, v_add):
                        current_set = temp_set | {v_add}
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try to remove one, add two
            for v_remove in list(current_set):
                temp_set = current_set - {v_remove}
                added = 0
                for v_add1 in range(num_v):
                    if v_add1 not in temp_set and _can_add(temp_set, v_add1):
                        temp_set.add(v_add1)
                        added += 1
                        break
                if added == 1:
                    for v_add2 in range(num_v):
                        if v_add2 not in temp_set and _can_add(temp_set, v_add2):
                            temp_set.add(v_add2)
                            added += 1
                            break
                if added == 2:
                    current_set = temp_set
                    improved = True
                    break
        
        return current_set
    
    best_set = _local_search(best_set, max_iters=500)
    
    # Try multiple random restarts with local search
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        
        # Random greedy
        random_order = list(range(num_v))
        random.shuffle(random_order)
        curr = _greedy(random_order)
        curr = _local_search(curr, max_iters=200)
        if len(curr) > len(best_set):
            best_set = curr
    
    # Convert to values
    result = [all_nums[i] for i in best_set]
    result.sort()
    
    return list(result)

# EVOLVE-BLOCK-END
