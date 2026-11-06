import random
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 150s limit
    
    base_elements = [1, 2, 5, 11]
    target_size = 20
    max_val = 284
    
    def find_solution_20():
        for restart in range(2000):
            if time.time() > deadline:
                return None
            
            # Initialize
            available = [x for x in range(1, max_val + 1) if x not in base_elements]
            random.shuffle(available)
            current_set = base_elements + available[:16]
            current_set.sort()
            
            # Build initial sum counts
            sum_count = {}
            n = len(current_set)
            for i in range(n):
                for j in range(i, n):
                    s = current_set[i] + current_set[j]
                    if s in sum_count:
                        sum_count[s] += 1
                    else:
                        sum_count[s] = 1
                        
            # Count conflicts (sums with count > 1)
            num_conflicts = sum(1 for c in sum_count.values() if c > 1)
            
            if num_conflicts == 0:
                return current_set
            
            # Repair Loop
            max_iter = 2000
            for _ in range(max_iter):
                if time.time() > deadline:
                    return None
                
                if num_conflicts == 0:
                    return current_set
                
                # Find a conflicting sum
                # We can pick a random sum with count > 1
                # To do this efficiently, we can just iterate through sum_count
                # But sum_count has ~200 keys.
                # We can also maintain a list of conflicting sums?
                # Or just pick a random key and check?
                # Better: Pick a random element, get its sums, check if any are conflicts?
                # No, just pick a random key from sum_count until we find one with count > 1.
                
                conflict_sum = None
                # Since num_conflicts is small usually, this is fast
                # Worst case, we scan all keys.
                for s, c in sum_count.items():
                    if c > 1:
                        conflict_sum = s
                        break
                        
                if conflict_sum is None:
                    return current_set # Should be caught by num_conflicts check
                
                # Find pairs summing to conflict_sum
                pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if current_set[i] + current_set[j] == conflict_sum:
                            pairs.append((i, j))
                            # Also include i=j case?
                            # If i==j, 2*x = conflict_sum.
                            # My loop above is i < j.
                            # Need to check i == j separately.
                # Check i == j
                for i in range(n):
                    if 2 * current_set[i] == conflict_sum:
                        pairs.append((i, i))
                        
                if not pairs:
                    # Should not happen
                    continue
                    
                i, j = random.choice(pairs)
                
                # Determine movable indices
                movable = []
                if current_set[i] not in base_elements:
                    movable.append(i)
                if i != j and current_set[j] not in base_elements:
                    movable.append(j)
                    
                if not movable:
                    continue
                    
                idx = random.choice(movable)
                old_val = current_set[idx]
                
                # Pick new value
                new_val = random.randint(1, max_val)
                if new_val in current_set:
                    continue
                
                # Update sum_count and num_conflicts
                # Remove old sums
                # Sums involving old_val:
                # 1. old_val + old_val (if idx == idx, i.e., 2*old_val)
                # 2. old_val + current_set[k] for k != idx
                
                # Helper to decrement count
                def dec_sum(s):
                    nonlocal num_conflicts
                    if s in sum_count:
                        old_c = sum_count[s]
                        if old_c == 2:
                            num_conflicts -= 1
                        sum_count[s] -= 1
                        if sum_count[s] == 0:
                            del sum_count[s]
                            
                # Helper to increment count
                def inc_sum(s):
                    nonlocal num_conflicts
                    if s in sum_count:
                        old_c = sum_count[s]
                        if old_c == 1:
                            num_conflicts += 1
                        sum_count[s] += 1
                    else:
                        sum_count[s] = 1
                        # If count becomes 1, no conflict added
                        # Wait, if count was 0 (new), it becomes 1. No conflict.
                        # If count was 1, it becomes 2. Conflict added.
                        # So:
                        # if old_c == 1: num_conflicts += 1
                        # else: num_conflicts unchanged (new key or old_c >= 2)
                        
                # Decrement old sums
                # 2 * old_val
                dec_sum(2 * old_val)
                for k in range(n):
                    if k != idx:
                        dec_sum(old_val + current_set[k])
                        
                # Increment new sums
                # 2 * new_val
                inc_sum(2 * new_val)
                for k in range(n):
                    if k != idx:
                        inc_sum(new_val + current_set[k])
                        
                # Update set
                current_set[idx] = new_val
                current_set.sort()
                
                # Note: Indices in current_set changed after sort.
                # But we only use current_set for values and membership.
                # The sum_count is updated based on values, so it's consistent.
                # We don't use indices for sum_count updates anymore.
                
            # End repair loop
            pass
            
        return None

    sol = find_solution_20()
    if sol:
        return sol
        
    # Fallback
    a = base_elements.copy()
    sums = set()
    for i in range(len(a)):
        for j in range(i, len(a)):
            sums.add(a[i] + a[j])
            
    for x in range(1, max_val + 1):
        if x in a:
            continue
        new_sums = {x + y for y in a} | {2 * x}
        if len(new_sums) == len(a) + 1 and not (new_sums & sums):
            a.append(x)
            a.sort()
            sums |= new_sums
            
    return a
