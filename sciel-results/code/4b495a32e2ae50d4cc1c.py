def construct():
    from fractions import Fraction
    import time
    import random
    
    deadline = time.time() + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    best_sum = Fraction(0)
    best_indices = []
    
    def greedy_fill(current_sum, exclude_set=None):
        """Greedily add fractions to maximize sum while keeping < 1."""
        total = current_sum
        selected = []
        if exclude_set is None:
            exclude_set = set()
        for i, f in enumerate(fractions):
            if i in exclude_set:
                continue
            if total + f < 1:
                total += f
                selected.append(i)
        return total, selected
    
    # Initial: greedy fill from empty
    total, selected = greedy_fill(Fraction(0))
    if total > best_sum:
        best_sum = total
        best_indices = selected
    
    # Enumerate subsets of first 15 fractions, greedily fill rest
    split = 15
    large_fracs = fractions[:split]
    
    def enumerate_large(idx, current_sum, selected_large):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        if idx == split:
            total, selected_small = greedy_fill(current_sum, set(selected_large))
            all_selected = selected_large + [i for i in selected_small if i not in set(selected_large)]
            if total < 1 and total > best_sum:
                best_sum = total
                best_indices = all_selected
            return
        
        # Try including
        new_sum = current_sum + large_fracs[idx]
        if new_sum < 1:
            selected_large.append(idx)
            enumerate_large(idx + 1, new_sum, selected_large)
            selected_large.pop()
        
        # Try excluding
        enumerate_large(idx + 1, current_sum, selected_large)
    
    enumerate_large(0, Fraction(0), [])
    
    # Multi-start local search with random initializations
    random.seed(42)
    
    def local_search(start_indices, max_iters=200):
        nonlocal best_sum, best_indices
        
        if not start_indices:
            return
        
        curr_indices = list(start_indices)
        curr_set = set(curr_indices)
        curr_sum = sum(fractions[i] for i in curr_indices)
        
        if curr_sum >= 1:
            return
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding elements
            for i in range(n):
                if i not in curr_set:
                    if curr_sum + fractions[i] < 1:
                        new_sum = curr_sum + fractions[i]
                        if new_sum > best_sum:
                            curr_indices.append(i)
                            curr_set.add(i)
                            curr_sum = new_sum
                            best_sum = new_sum
                            best_indices = list(curr_indices)
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try 1-swap: remove i, add j
            for i in curr_indices[:]:
                for j in range(n):
                    if j in curr_set:
                        continue
                    new_sum = curr_sum - fractions[i] + fractions[j]
                    if new_sum < 1 and new_sum > best_sum:
                        curr_indices = [x for x in curr_indices if x != i] + [j]
                        curr_set = set(curr_indices)
                        curr_sum = new_sum
                        best_sum = new_sum
                        best_indices = list(curr_indices)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove i1, i2, add j1, j2
            if len(curr_indices) >= 2:
                for idx1 in range(len(curr_indices)):
                    for idx2 in range(idx1 + 1, len(curr_indices)):
                        i1 = curr_indices[idx1]
                        i2 = curr_indices[idx2]
                        remove_sum = fractions[i1] + fractions[i2]
                        for j1 in range(n):
                            if j1 in curr_set:
                                continue
                            for j2 in range(j1 + 1, n):
                                if j2 in curr_set:
                                    continue
                                add_sum = fractions[j1] + fractions[j2]
                                new_sum = curr_sum - remove_sum + add_sum
                                if new_sum < 1 and new_sum > best_sum:
                                    new_indices = [x for x in curr_indices if x not in (i1, i2)] + [j1, j2]
                                    curr_indices = new_indices
                                    curr_set = set(curr_indices)
                                    curr_sum = new_sum
                                    best_sum = new_sum
                                    best_indices = list(curr_indices)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if improved:
                continue
            
            # Try removing one element then greedily refilling
            if len(curr_indices) > 1:
                for i in curr_indices[:]:
                    new_indices = [x for x in curr_indices if x != i]
                    new_set = set(new_indices)
                    new_sum = curr_sum - fractions[i]
                    # Greedily add back
                    temp_sum, temp_added = greedy_fill(new_sum, new_set)
                    if temp_sum > best_sum:
                        curr_indices = new_indices + temp_added
                        curr_set = set(curr_indices)
                        curr_sum = temp_sum
                        best_sum = temp_sum
                        best_indices = list(curr_indices)
                        improved = True
                        break
            
            if not improved:
                break
        
        return curr_indices, curr_sum
    
    # Run local search from initial solution
    if best_indices and time.time() < deadline - 10:
        local_search(best_indices)
    
    # Multi-start: try random initializations
    num_starts = 50
    for start_num in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Create a random starting set
        start_set = set()
        # Randomly pick some elements
        num_to_pick = random.randint(3, 15)
        candidates = list(range(n))
        random.shuffle(candidates)
        for i in candidates[:num_to_pick]:
            if sum(fractions[j] for j in start_set) + fractions[i] < 1:
                start_set.add(i)
        
        # Greedily fill the rest
        total, added = greedy_fill(sum(fractions[i] for i in start_set), start_set)
        start_indices = list(start_set) + added
        
        if start_indices:
            local_search(start_indices, max_iters=50)
    
    return best_indices
