<<<<<<< SEARCH
def _local_search(initial, max_time=45):
    """Local search to minimize max Sidon subset size."""
    start_time = time.time()
    current = list(initial)
    current_score = _max_sidon_subset_size(current)
    
    best = list(current)
    best_score = current_score
    
    if best_score <= 3:
        return best, best_score
    
    random.seed(12345)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try single-element replacements
        for i in range(12):
            if time.time() - start_time >= max_time:
                break
            for _ in range(30):
                new_val = random.randint(0, 128)
                if new_val in current and new_val != current[i]:
                    continue
                new_arr = list(current)
                new_arr[i] = new_val
                if len(set(new_arr)) != 12:
                    continue
                score = _max_sidon_subset_size(new_arr)
                if score < current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                        if best_score <= 3:
                            return best, best_score
                    improved = True
                    break
                elif score == best_score and random.random() < 0.05:
                    current = new_arr
                    improved = True
                    break
        
        if not improved:
            # Multi-element perturbation
            new_arr = list(current)
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in new_arr or new_val == new_arr[idx]:
                    new_arr[idx] = new_val
            if len(set(new_arr)) == 12:
                score = _max_sidon_subset_size(new_arr)
                if score <= current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                        if best_score <= 3:
                            return best, best_score
    
    return best, best_score
=======
def _local_search(initial, max_time=45):
    """Local search to minimize max Sidon subset size."""
    start_time = time.time()
    current = list(initial)
    current_score = _max_sidon_subset_size(current)
    
    best = list(current)
    best_score = current_score
    
    if best_score <= 3:
        return best, best_score
    
    random.seed(12345)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try single-element replacements
        for i in range(12):
            if time.time() - start_time >= max_time:
                break
            for _ in range(50): # Increased attempts
                new_val = random.randint(0, 128)
                if new_val in current and new_val != current[i]:
                    continue
                new_arr = list(current)
                new_arr[i] = new_val
                if len(set(new_arr)) != 12:
                    continue
                score = _max_sidon_subset_size(new_arr)
                if score < current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                        if best_score <= 3:
                            return best, best_score
                    improved = True
                    break
                elif score == best_score and random.random() < 0.1:
                    current = new_arr
                    improved = True
                    break
        
        if not improved:
            # Multi-element perturbation (swap 2 elements)
            new_arr = list(current)
            idx1, idx2 = random.sample(range(12), 2)
            new_val1 = random.randint(0, 128)
            new_val2 = random.randint(0, 128)
            
            # Ensure distinctness
            if new_val1 in new_arr and new_val1 != new_arr[idx1]:
                continue
            if new_val2 in new_arr and new_val2 != new_arr[idx2] and new_val2 != new_val1:
                continue
            if new_val1 == new_val2:
                continue
                
            new_arr[idx1] = new_val1
            new_arr[idx2] = new_val2
            
            if len(set(new_arr)) == 12:
                score = _max_sidon_subset_size(new_arr)
                if score <= current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                        if best_score <= 3:
                            return best, best_score
    
    return best, best_score
>>>>>>> REPLACE

Finally, update `construct` to use the improved search.

