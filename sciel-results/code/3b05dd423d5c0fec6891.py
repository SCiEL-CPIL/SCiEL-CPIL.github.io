"""Sidon set construction with bitmask conflict checking and local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Max sum is 284 + 284 = 568
    MAX_SUM = 568
    
    def compute_sum_mask(elements):
        """Compute bitmask of all pairwise sums (a+b for a<=b)."""
        mask = 0
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                mask |= (1 << s)
        return mask
    
    def can_add_bitmask(element, elements, sum_mask):
        """Check if element can be added using bitmask. Returns True if valid."""
        # Check 2*element
        if (sum_mask >> (2 * element)) & 1:
            return False
        # Check element + x for each x in elements
        for x in elements:
            s = element + x
            if (sum_mask >> s) & 1:
                return False
        return True
    
    def greedy_construct(seed_rng, elements_order):
        """Greedy construction trying elements in given order."""
        current = list(base)
        current_set = set(base)
        sum_mask = compute_sum_mask(current)
        
        for x in elements_order:
            if x in current_set:
                continue
            if can_add_bitmask(x, current, sum_mask):
                current.append(x)
                current_set.add(x)
                # Update sum mask
                sum_mask |= (1 << (2 * x))
                for y in current[:-1]:
                    sum_mask |= (1 << (x + y))
        
        return current, sum_mask
    
    def local_search_1swap(elements, sum_mask, seed_rng, max_iters=200):
        """Try to improve by swapping one element out for another."""
        current = list(elements)
        current_set = set(elements)
        cur_mask = sum_mask
        n = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try removing a random element and adding a better one
            idx = seed_rng.randint(0, n - 1)
            removed = current[idx]
            
            # Compute mask without this element
            mask_without = cur_mask
            # Remove sums involving 'removed'
            for i, x in enumerate(current):
                if i == idx:
                    continue
                s = removed + x
                # Only clear if this sum isn't shared with other pairs
                # For simplicity, rebuild
                pass
            
            # Rebuild mask without the removed element
            remaining = current[:idx] + current[idx+1:]
            new_mask = compute_sum_mask(remaining)
            
            # Try to add a new element
            best_new = None
            best_score = -1
            for x in range(1, 285):
                if x in current_set or x == removed:
                    continue
                if can_add_bitmask(x, remaining, new_mask):
                    # Score: prefer elements that allow more future additions
                    # Simple score: number of available candidates after adding
                    candidate_remaining = [y for y in range(1, 285) if y not in current_set and y != removed and y != x]
                    count = 0
                    for y in candidate_remaining:
                        if can_add_bitmask(y, remaining + [x], new_mask | (1 << (2*x)) | sum((1 << (x+z)) for z in remaining)):
                            count += 1
                    if count > best_score:
                        best_score = count
                        best_new = x
            
            if best_new is not None:
                current[idx] = best_new
                current_set.discard(removed)
                current_set.add(best_new)
                cur_mask = compute_sum_mask(current)
        
        return current, cur_mask
    
    def local_search_2swap(elements, sum_mask, seed_rng, max_iters=100):
        """Try to improve by swapping two elements out for two new ones."""
        current = list(elements)
        current_set = set(elements)
        cur_mask = sum_mask
        n = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick two random indices to swap out
            i, j = seed_rng.sample(range(n), 2)
            removed1, removed2 = current[i], current[j]
            
            remaining = [x for k, x in enumerate(current) if k != i and k != j]
            new_mask = compute_sum_mask(remaining)
            
            # Try to add two new elements
            best_pair = None
            best_score = -1
            
            # Sample a subset of candidates for speed
            candidates = [x for x in range(1, 285) if x not in current_set and x != removed1 and x != removed2]
            if len(candidates) > 50:
                candidates = seed_rng.sample(candidates, 50)
            
            for x in candidates:
                if not can_add_bitmask(x, remaining, new_mask):
                    continue
                mask_with_x = new_mask | (1 << (2*x))
                for y in remaining:
                    mask_with_x |= (1 << (x + y))
                
                for z in candidates:
                    if z <= x:
                        continue
                    if z == removed1 or z == removed2:
                        continue
                    if can_add_bitmask(z, remaining + [x], mask_with_x):
                        # Valid pair
                        score = len(remaining) + 2
                        if score > best_score:
                            best_score = score
                            best_pair = (x, z)
            
            if best_pair is not None:
                current[i] = best_pair[0]
                current[j] = best_pair[1]
                current_set.discard(removed1)
                current_set.discard(removed2)
                current_set.add(best_pair[0])
                current_set.add(best_pair[1])
                cur_mask = compute_sum_mask(current)
        
        return current, cur_mask
    
    best_solution = list(base)
    best_size = len(base)
    
    # Multiple randomized restarts with different orderings
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed_rng = random.Random(42 + restart * 1000)
        
        # Create a randomized order of candidates
        all_candidates = [x for x in range(1, 285) if x not in base_set]
        
        # Mix of random and heuristic orderings
        if restart % 3 == 0:
            # Random order
            elements_order = all_candidates[:]
            seed_rng.shuffle(elements_order)
        elif restart % 3 == 1:
            # Prefer spread-out elements
            def spread_score(x):
                min_dist = min(abs(x - e) for e in base)
                return min_dist
            elements_order = sorted(all_candidates, key=spread_score, reverse=True)
            # Add some randomness
            for i in range(len(elements_order)):
                j = seed_rng.randint(i, len(elements_order) - 1)
                elements_order[i], elements_order[j] = elements_order[j], elements_order[i]
        else:
            # Prefer larger elements (to maximize coverage)
            elements_order = sorted(all_candidates, reverse=True)
            for i in range(len(elements_order)):
                j = seed_rng.randint(i, len(elements_order) - 1)
                elements_order[i], elements_order[j] = elements_order[j], elements_order[i]
        
        # Greedy construction
        solution, sum_mask = greedy_construct(seed_rng, elements_order)
        
        if len(solution) > best_size:
            best_size = len(solution)
            best_solution = list(solution)
        
        # Local search to improve
        if len(solution) >= 15:
            solution, sum_mask = local_search_1swap(solution, sum_mask, seed_rng, max_iters=50)
            if len(solution) > best_size:
                best_size = len(solution)
                best_solution = list(solution)
            
            if len(solution) >= 17:
                solution, sum_mask = local_search_2swap(solution, sum_mask, seed_rng, max_iters=30)
                if len(solution) > best_size:
                    best_size = len(solution)
                    best_solution = list(solution)
        
        # Early exit if we've found optimal
        if best_size >= 20:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
