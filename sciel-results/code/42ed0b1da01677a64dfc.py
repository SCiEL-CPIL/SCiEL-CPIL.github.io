import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper: check if we can add element x to set a (with existing sums)
    def can_add(a, sums, x):
        # Check all new sums involving x
        for y in a:
            s = x + y
            if s in sums:
                return False
        # Check 2*x
        if 2 * x in sums:
            return False
        # Also need to check that the new sums don't collide with each other
        # Since all new sums are x+y for different y, they are distinct if y are distinct
        # And 2*x is distinct from x+y iff x != y, which is true since x not in a
        return True
    
    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}
    
    # Greedy expansion: try to add elements in various orders
    def greedy_expand(initial_list, order):
        a = list(initial_list)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for x in order:
            if x in set(a):
                continue
            if can_add(a, sums, x):
                a.append(x)
                sums |= get_new_sums(a, x)
        return a
    
    # Try multiple greedy strategies with different orderings
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2
    
    # Strategy 3: Randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        # Create a random ordering of candidates
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        # Also try with some bias toward larger or smaller values
        if restart % 2 == 0:
            # Bias toward larger
            candidates.sort(key=lambda x: random.random() * 100 + x)
        else:
            # Bias toward smaller
            candidates.sort(key=lambda x: random.random() * 100 - x)
        
        a3 = greedy_expand(fixed_elements, candidates)
        if len(a3) > len(best):
            best = a3
    
    # Local search / hill climbing: try to improve by swapping elements
    def local_search(a, max_iter=1000):
        a = list(a)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # Try to remove an element and add a better one
            for i in range(len(a)):
                if a[i] in fixed_set:
                    continue  # Don't remove fixed elements
                
                # Try removing a[i] and adding each candidate
                temp_a = [x for j, x in enumerate(a) if j != i]
                temp_sums = set()
                for k, x in enumerate(temp_a):
                    for y in temp_a[k:]:
                        temp_sums.add(x + y)
                
                # Find candidates that can be added
                for cand in range(1, 285):
                    if cand in fixed_set or cand in temp_a:
                        continue
                    if can_add(temp_a, temp_sums, cand):
                        # Check if this improves (we want to maximize size, so any addition is good
                        # but we also want to keep the set "dense" for future improvements)
                        # For now, just accept if it maintains size or we can add more later
                        # Actually, we're doing swap: remove one, add one, so size stays same
                        # This doesn't help maximize size. Let's try a different approach.
                        pass
            
            # Alternative: try to add more elements after removing some
            # This is complex, so let's try a different local search
        
        return a
    
    # Better approach: iterative improvement by trying to insert elements
    def iterative_improve(a, max_rounds=100):
        a = list(a)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            added_any = False
            # Try adding each candidate
            for cand in range(1, 285):
                if cand in set(a) or cand in fixed_set:
                    continue
                if can_add(a, sums, cand):
                    a.append(cand)
                    sums |= get_new_sums(a, cand)
                    added_any = True
            
            if not added_any:
                # Try to remove a non-fixed element and see if we can add multiple new ones
                for i in range(len(a)):
                    if a[i] in fixed_set:
                        continue
                    
                    removed = a[i]
                    temp_a = [x for j, x in enumerate(a) if j != i]
                    temp_sums = set()
                    for k, x in enumerate(temp_a):
                        for y in temp_a[k:]:
                            temp_sums.add(x + y)
                    
                    # Try to add as many candidates as possible
                    new_a = list(temp_a)
                    new_sums = set(temp_sums)
                    count_before = len(new_a)
                    
                    for cand in range(1, 285):
                        if cand in set(new_a) or cand in fixed_set:
                            continue
                        if can_add(new_a, new_sums, cand):
                            new_a.append(cand)
                            new_sums |= get_new_sums(new_a, cand)
                    
                    if len(new_a) > len(a):
                        a = new_a
                        sums = new_sums
                        added_any = True
                        break
                    elif len(new_a) == len(a) and random.random() < 0.1:
                        # Occasionally accept same-size improvements for diversity
                        a = new_a
                        sums = new_sums
                        added_any = True
                        break
            
            if not added_any:
                break
        
        return a
    
    # Apply iterative improvement to the best solution found
    best = iterative_improve(best, max_rounds=200)
    
    # Try more random restarts with iterative improvement
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Random initial expansion
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        a = greedy_expand(fixed_elements, candidates)
        
        # Iterative improvement
        a = iterative_improve(a, max_rounds=50)
        
        if len(a) > len(best):
            best = a
    
    return best
