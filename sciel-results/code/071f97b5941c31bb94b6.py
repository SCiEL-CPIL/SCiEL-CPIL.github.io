import random
import time
import math

def _cost(elements):
    """
    Returns (num_duplicate_sums, num_divisibility_pairs).
    We want both to be 0.
    """
    n = len(elements)
    if n == 0:
        return 0, 0
    
    # Generate subset sums
    sums = set()
    sums.add(0) # Empty sum, though task says non-empty. 
    # Wait, task says "Form every NONEMPTY subset sum".
    # So we should not include 0.
    # But for generating, it's easier to include 0 and remove it at the end?
    # Or just generate non-empty.
    
    # Efficient generation
    current_sums = set()
    for x in elements:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        current_sums.update(new_sums)
        
    # current_sums contains all non-empty subset sums
    M = len(current_sums)
    
    # Check duplicates
    # The set construction handles duplicates automatically, 
    # but we need to know if there were duplicates to penalize.
    # Actually, if we just use a set, we lose the count of duplicates.
    # We need to count how many pairs (i,j) have same sum.
    # Let's count frequencies.
    freq = {}
    for s in current_sums:
        freq[s] = freq.get(s, 0) + 1
        
    num_dupes = 0
    for count in freq.values():
        if count > 1:
            # Number of pairs with same sum is C(count, 2)
            num_dupes += count * (count - 1) // 2
            
    # Check divisibility
    # Only check distinct values
    distinct_sums = sorted(freq.keys())
    sum_set = set(distinct_sums)
    max_sum = distinct_sums[-1] if distinct_sums else 0
    
    num_div_pairs = 0
    for a in distinct_sums:
        # Check multiples of a
        # If b is a multiple of a, b = k*a
        # We need to count how many b in sum_set are multiples of a, b != a
        # And avoid double counting. 
        # If we count for each a, the number of b > a such that b % a == 0, 
        # we count each pair (a,b) exactly once.
        for b in range(2 * a, max_sum + 1, a):
            if b in sum_set:
                num_div_pairs += 1
                
    return num_dupes, num_div_pairs

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    if _validate(best_set): # Reusing old validator for initial check? No, let's use cost.
         # Actually, let's just initialize with a known good small set or random.
         pass
         
    # Let's start with a random set of size 5
    random.seed(42)
    
    # Helper to check if a set is valid (cost is 0,0)
    def is_valid(elements):
        d, v = _cost(elements)
        return d == 0 and v == 0

    # Try to build up from small k
    # Start with k=5
    current_set = []
    # Find a valid set of size 5
    # Use simple greedy first to get a base
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    for x in candidates:
        if len(current_set) >= 5:
            break
        test_set = current_set + [x]
        if is_valid(test_set):
            current_set.append(x)
            
    if len(current_set) < 5:
        # Fallback to hardcoded if greedy fails
        current_set = [31, 32, 34, 36, 40]
        if not is_valid(current_set):
             # Try another
             current_set = [100, 101, 103, 107, 109]
             
    best_set = current_set[:]
    best_size = len(best_set)
    
    # Now try to increase size using Simulated Annealing
    # For each target size K from best_size + 1 to 15:
    #   Try to find a valid set of size K
    
    for target_k in range(best_size + 1, 16):
        if time.time() > deadline - 10: # Safety margin
            break
            
        # Initialize with best_set + random element
        current = best_set[:]
        # Add a random element
        candidates = [x for x in range(1, 257) if x not in current]
        if not candidates:
            break
        current.append(random.choice(candidates))
        
        # Annealing
        T = 10.0
        cooling = 0.95
        accepted = False
        
        # We want to reach cost 0
        # Cost function: d + v
        def get_total_cost(elements):
            d, v = _cost(elements)
            return d + v
            
        current_cost = get_total_cost(current)
        
        if current_cost == 0:
            best_set = current[:]
            best_size = target_k
            continue
            
        # SA Loop
        max_iters = 5000
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random index to change
            idx = random.randint(0, target_k - 1)
            old_val = current[idx]
            
            # Pick a new value
            candidates = [x for x in range(1, 257) if x not in current]
            if not candidates:
                break
            new_val = random.choice(candidates)
            
            current[idx] = new_val
            new_cost = get_total_cost(current)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost == 0:
                    best_set = current[:]
                    best_size = target_k
                    accepted = True
                    break
            else:
                current[idx] = old_val
                
            T *= cooling
            
        if not accepted:
            # If we couldn't reach size target_k, try a few more times with different starts?
            # Or just break to save time for other sizes?
            # Let's try 3 restarts for this size
            success = False
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                current = best_set[:]
                candidates = [x for x in range(1, 257) if x not in current]
                if not candidates: break
                current.append(random.choice(candidates))
                
                T = 10.0
                cooling = 0.95
                current_cost = get_total_cost(current)
                if current_cost == 0:
                    success = True
                    break
                    
                for i in range(2000):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(0, target_k - 1)
                    old_val = current[idx]
                    candidates = [x for x in range(1, 257) if x not in current]
                    if not candidates: break
                    new_val = random.choice(candidates)
                    
                    current[idx] = new_val
                    new_cost = get_total_cost(current)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        if current_cost == 0:
                            success = True
                            break
                    else:
                        current[idx] = old_val
                    T *= cooling
                    if current_cost == 0:
                        break
                        
                if success:
                    break
                    
            if success:
                best_set = current[:]
                best_size = target_k
            else:
                # If we fail to find a valid set of size target_k, 
                # it might be hard to get target_k + 1.
                # But maybe a different configuration works?
                # For now, break to avoid wasting time.
                # However, sometimes size 8 is hard but 9 is easy? Unlikely.
                break

    return list(best_set)

def _validate(elements):
    # Kept for compatibility, though not used in new construct logic directly
    # except maybe for final verification?
    n = len(elements)
    if n == 0:
        return True
    
    sums = set()
    num_subsets = 1 << n
    for mask in range(1, num_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
        
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return False
    return True
