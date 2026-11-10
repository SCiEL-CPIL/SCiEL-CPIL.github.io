import time
import random

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin

    def is_valid(sums, max_k):
        union = 0
        total_count = 0
        for k in range(max_k + 1):
            s = sums[k]
            if s:
                union |= s
                total_count += s.bit_count()
        return union.bit_count() == total_count

    def compute_sums(elements):
        sums = [0] * (len(elements) + 1)
        sums[0] = 1
        for x in elements:
            # Update in reverse to avoid using x multiple times in same step
            for k in range(len(elements), 0, -1):
                sums[k] |= (sums[k-1] << x)
        return sums

    def try_greedy(order):
        sums = [0] * 61
        sums[0] = 1
        chosen = []
        max_k = 0
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check if adding x keeps validity
            # We need to compute new sums to check.
            # Optimization: Only compute up to max_k + 1
            new_sums = [0] * 61
            for k in range(max_k + 1, 0, -1):
                new_sums[k] = sums[k] | ((sums[k-1] << x))
            new_sums[0] = sums[0]
            
            if is_valid(new_sums, max_k + 1):
                sums = new_sums
                max_k += 1
                chosen.append(x)
                
        return chosen

    # Base strategy: Greedy with various orderings
    best = []
    
    # 1. Powers of 2 as a strong baseline
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    # Check if we can add 256
    test = base + [256]
    sums = compute_sums(test)
    if is_valid(sums, len(test)):
        best = test
    else:
        best = base
        
    best_len = len(best)
    
    # 2. Randomized Greedy
    rng = random.Random(42)
    for trial in range(2000): # More trials
        if time.time() > deadline - 5:
            break
            
        # Random ordering
        order = list(range(1, 257))
        rng.shuffle(order)
        
        # Bias towards small numbers? Large numbers?
        # Small numbers create dense low sums. Large numbers create sparse high sums.
        # Mixing might be good.
        
        r = try_greedy(order)
        if len(r) > best_len:
            best = r
            best_len = len(r)
            
    # 3. Local Search / Hill Climbing
    # Start from best found so far
    current = list(best)
    current_set = set(current)
    
    def get_conflict_elements(elements):
        """Return a set of elements involved in collisions."""
        sums = compute_sums(elements)
        n = len(elements)
        # Find if valid
        if is_valid(sums, n):
            return set()
            
        # Find a collision
        # We need to find S_k intersect S_l.
        # Then find subsets.
        # This is expensive. 
        # Alternative: Just return a random element? 
        # Better: Identify which k, l collide.
        colliding_k = -1
        colliding_l = -1
        for k in range(n + 1):
            for l in range(k + 1, n + 1):
                if sums[k] & sums[l]:
                    colliding_k = k
                    colliding_l = l
                    break
            if colliding_k != -1:
                break
                
        if colliding_k == -1:
            return set()
            
        # Find the specific sum that collides
        common = sums[colliding_k] & sums[colliding_l]
        # Pick one bit
        sum_val = (common & -common).bit_length() - 1
        
        # Now find a subset of size colliding_k that sums to sum_val
        # This is a subset sum problem. 
        # Given the small size, we can backtrack.
        
        def find_subset(target, k, available):
            if k == 0:
                return [] if target == 0 else None
            if not available:
                return None
            # Try including the last element
            last = available[-1]
            if last <= target:
                res = find_subset(target - last, k - 1, available[:-1])
                if res is not None:
                    return [last] + res
            # Try excluding
            res = find_subset(target, k, available[:-1])
            if res is not None:
                return res
            return None

        # We need to find subset of size colliding_k summing to sum_val
        # And subset of size colliding_l summing to sum_val
        # We can just pick one subset to break.
        # Let's pick the one with smaller cardinality? Or just the first one found.
        
        # To speed up, we can just remove a random element from the set 
        # and hope it breaks the collision. 
        # But we want to be guided.
        
        # Let's try to find the subset.
        subset_k = find_subset(sum_val, colliding_k, current)
        if subset_k:
            return set(subset_k)
            
        subset_l = find_subset(sum_val, colliding_l, current)
        if subset_l:
            return set(subset_l)
            
        return {current[0]} # Fallback

    for step in range(5000):
        if time.time() > deadline - 2:
            break
            
        conflicts = get_conflict_elements(current)
        if not conflicts:
            # Valid, try to add
            unused = [x for x in range(1, 257) if x not in current_set]
            if not unused:
                break
            # Try to add a random unused element
            x = rng.choice(unused)
            new_current = current + [x]
            sums = compute_sums(new_current)
            if is_valid(sums, len(new_current)):
                current = new_current
                current_set.add(x)
                best = current
                best_len = len(current)
            else:
                # Try swapping
                # Remove a conflicting element if we had one, but we are valid now.
                # So just try adding another one?
                # Or try swapping an existing small one with a large unused one?
                pass
        else:
            # Invalid, remove a conflicting element
            # Pick one from conflicts
            to_remove = rng.choice(list(conflicts))
            current = [x for x in current if x != to_remove]
            current_set.discard(to_remove)
            
    return list(best)
