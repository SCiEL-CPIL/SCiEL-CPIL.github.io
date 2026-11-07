<<<<<<< SEARCH
def _branch_and_bound(mask, deadline=0.0):
    """
    Branch and bound to find maximum independent set.
    """
    if time.time() > deadline:
        return mask
    
    current_size = mask.bit_count()
    best_mask = mask
    best_size = current_size
    
    elems = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    if len(elems) <= 30:
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if time.time() > deadline:
                    return best_mask
                
                a = elems[i]
                b = elems[j]
                reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                
                candidate = _greedy_fill(reduced)
                if candidate.bit_count() > best_size:
                    best_size = candidate.bit_count()
                    best_mask = candidate
    else:
        num_samples = 100
        for _ in range(num_samples):
            if time.time() > deadline:
                return best_mask
            
            if len(elems) < 2:
                break
            a, b = random.sample(elems, 2)
            reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
            
            candidate = _greedy_fill(reduced)
            if candidate.bit_count() > best_size:
                best_size = candidate.bit_count()
                best_mask = candidate

    return best_mask
=======
def _branch_and_bound(mask, deadline=0.0):
    """
    Branch and bound to find maximum independent set.
    """
    if time.time() > deadline:
        return mask
    
    current_size = mask.bit_count()
    best_mask = mask
    best_size = current_size
    
    elems = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    if len(elems) <= 30:
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if time.time() > deadline:
                    return best_mask
                
                a = elems[i]
                b = elems[j]
                reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                
                candidate = _greedy_fill(reduced)
                if candidate.bit_count() > best_size:
                    best_size = candidate.bit_count()
                    best_mask = candidate
    else:
        # Adaptive number of samples based on remaining time
        remaining = deadline - time.time()
        num_samples = min(300, max(50, int(remaining * 20)))
        for _ in range(num_samples):
            if time.time() > deadline:
                return best_mask
            
            if len(elems) < 2:
                break
            a, b = random.sample(elems, 2)
            reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
            
            candidate = _greedy_fill(reduced)
            if candidate.bit_count() > best_size:
                best_size = candidate.bit_count()
                best_mask = candidate

    return best_mask
>>>>>>> REPLACE
