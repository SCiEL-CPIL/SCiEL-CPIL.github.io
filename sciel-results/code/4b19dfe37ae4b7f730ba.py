import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    fixed_elements = [1, 2, 5, 11]
    N = 284
    target_size = 20
    
    # Helper to compute cost and sum counts
    def get_initial_state(elements):
        sum_counts = {}
        cost = 0
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                if s in sum_counts:
                    sum_counts[s] += 1
                    # When a count goes from c to c+1, collisions increase by c
                    # Total collisions for count c is c*(c-1)/2
                    # Increase is (c+1)*c/2 - c*(c-1)/2 = c
                    cost += sum_counts[s] - 1 
                else:
                    sum_counts[s] = 1
        return sum_counts, cost

    def greedy_fallback():
        a = list(fixed_elements)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for x in range(12, N + 1):
            new_sums = {x + y for y in a} | {2 * x}
            if len(new_sums) == len(a) + 1 and not (new_sums & sums):
                a.append(x)
                sums |= new_sums
                if len(a) == target_size:
                    return a
        return a

    best_set = greedy_fallback()
    best_size = len(best_set)
    best_cost = 0 if best_size == target_size else float('inf')
    
    if best_size == target_size:
        # Verify it's actually valid (greedy should produce valid)
        return best_set

    # Local Search / Simulated Annealing to find size 20
    available = [x for x in range(1, N + 1) if x not in fixed_elements]
    
    # We will try to find a valid set of size 20
    # If we fail, we return the best partial set found (though greedy is likely better for partials)
    
    max_restarts = 500
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Initialize random set of size 20 containing fixed elements
        s = list(fixed_elements)
        # Sample 16 random distinct elements from available
        s += random.sample(available, target_size - len(fixed_elements))
        random.shuffle(s)
        
        sum_counts, current_cost = get_initial_state(s)
        
        if current_cost == 0:
            return s
            
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.9995
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random index to change
            idx = random.randint(0, target_size - 1)
            old_val = s[idx]
            
            # If we pick a fixed element, we cannot change it.
            # We should only pick from variable indices.
            # Since fixed elements are at specific indices only if we kept them there,
            # but we shuffled. So we must check.
            if old_val in fixed_elements:
                # Find a non-fixed index
                # This is inefficient. Better to maintain indices of variables.
                # But for simplicity, we can just retry or skip.
                # Let's identify variable indices.
                var_indices = [i for i, v in enumerate(s) if v not in fixed_elements]
                if not var_indices:
                    continue
                idx = random.choice(var_indices)
                old_val = s[idx]
            
            # Pick a new value
            # To avoid duplicates in the set, we pick from available \ {old_val} \ {other values in s}
            # This is O(K) to check.
            # Alternatively, pick random from 1..N and check if it's in s.
            new_val = random.randint(1, N)
            if new_val == old_val:
                continue
            if new_val in s:
                continue
                
            # Calculate delta cost
            # Remove old_val contributions
            # Note: s contains old_val at idx. Other elements are s without old_val.
            # We iterate over all other elements.
            
            # Optimization: Pre-compute list of other elements?
            # For K=20, iterating 19 times is fine.
            
            # 1. Decrease counts for sums involving old_val
            # We need to subtract collisions.
            # When count goes from c to c-1, collisions decrease by (c-1)
            
            delta = 0
            
            # Sums involving old_val and other elements
            for j in range(target_size):
                if j == idx:
                    continue
                other = s[j]
                sum_val = old_val + other
                c = sum_counts[sum_val]
                # Collision contribution before: c*(c-1)/2
                # Collision contribution after: (c-1)*(c-2)/2
                # Change: - (c-1)
                delta -= (c - 1)
                sum_counts[sum_val] = c - 1
                
            # Sum 2*old_val
            sum_val = 2 * old_val
            c = sum_counts[sum_val]
            delta -= (c - 1)
            sum_counts[sum_val] = c - 1
            
            # 2. Increase counts for sums involving new_val
            for j in range(target_size):
                if j == idx:
                    continue
                other = s[j]
                sum_val = new_val + other
                c = sum_counts.get(sum_val, 0)
                # Collision contribution before: c*(c-1)/2
                # Collision contribution after: (c+1)*c/2
                # Change: + c
                delta += c
                sum_counts[sum_val] = c + 1
                
            # Sum 2*new_val
            sum_val = 2 * new_val
            c = sum_counts.get(sum_val, 0)
            delta += c
            sum_counts[sum_val] = c + 1
            
            new_cost = current_cost + delta
            
            # Accept?
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                s[idx] = new_val
                current_cost = new_cost
                if current_cost == 0:
                    return s
            else:
                # Reject: Revert sum_counts
                # We didn't actually update s, but we updated sum_counts.
                # We need to revert sum_counts to the state before the move.
                
                # Revert: Remove new_val contributions
                for j in range(target_size):
                    if j == idx:
                        continue
                    other = s[j]
                    sum_val = new_val + other
                    sum_counts[sum_val] -= 1
                sum_val = 2 * new_val
                sum_counts[sum_val] -= 1
                
                # Revert: Add old_val contributions
                for j in range(target_size):
                    if j == idx:
                        continue
                    other = s[j]
                    sum_val = old_val + other
                    sum_counts[sum_val] += 1
                sum_val = 2 * old_val
                sum_counts[sum_val] += 1
                
            T *= cooling_rate
            if T < T_min:
                break
                
    return best_set
