import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s wall clock, but we want to be safe)
    
    n = 4096
    best_set = []
    
    def get_new_sums(x, S):
        """Compute all triple sums involving x and elements of S."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        for i in range(len(S)):
            y = S[i]
            for j in range(i, len(S)):
                z = S[j]
                new_sums.add(x + y + z)
        return new_sums
    
    def is_valid_addition(x, S, existing_sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        new_sums = get_new_sums(x, S)
        expected = 1 + len(S) + len(S) * (len(S) + 1) // 2
        if len(new_sums) != expected:
            return False
        if new_sums & existing_sums:
            return False
        return True
    
    def greedy_build(ordering, deadline):
        """Build a set using greedy algorithm with given ordering."""
        S = []
        existing_sums = set()
        
        for x in ordering:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S, existing_sums):
                S.append(x)
                existing_sums.update(get_new_sums(x, S[:-1]))
        
        return S
    
    # Run 1: Smallest First greedy (baseline)
    S1 = []
    sums1 = set()
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        new_sums = get_new_sums(x, S1)
        expected = 1 + len(S1) + len(S1) * (len(S1) + 1) // 2
        if len(new_sums) == expected and not (new_sums & sums1):
            S1.append(x)
            sums1.update(new_sums)
    
    if len(S1) > len(best_set):
        best_set = S1.copy()
    
    # Run 2: Power-of-3 inspired construction with perturbations
    # Use 3^k as a base, then add small perturbations
    base_power3 = []
    val = 1
    while val <= n:
        base_power3.append(val)
        val *= 3
    
    # Try to build around power-of-3 structure
    for seed in range(30):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Start with some power-of-3 values and add others
        S = []
        existing_sums = set()
        
        # First add some power-of-3 values
        for p in base_power3:
            if p > n:
                break
            if is_valid_addition(p, S, existing_sums):
                S.append(p)
                existing_sums.update(get_new_sums(p, S[:-1]))
        
        # Then try to add other values in random order
        remaining = [x for x in range(1, n + 1) if x not in set(S)]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S, existing_sums):
                S.append(x)
                existing_sums.update(get_new_sums(x, S[:-1]))
        
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Run 3: Multiple randomized greedy attempts with different orderings
    for seed in range(80):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Create candidates with various randomization strategies
        candidates = list(range(1, n + 1))
        
        strategy = seed % 6
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            # Bias toward smaller values
            candidates.sort(key=lambda x: x + random.randint(-100, 100))
        elif strategy == 2:
            # Bias toward larger values
            candidates.sort(key=lambda x: -x + random.randint(-100, 100))
        elif strategy == 3:
            # Mix of small and large
            small = list(range(1, n // 2 + 1))
            large = list(range(n // 2 + 1, n + 1))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
        elif strategy == 4:
            # Random with some structure - pick in chunks
            chunk_size = 100
            for i in range(0, n, chunk_size):
                chunk = list(range(i + 1, min(i + chunk_size + 1, n + 1)))
                random.shuffle(chunk)
                candidates.extend(chunk)
        else:
            # Weighted random - prefer values that are "spread out"
            weights = [1.0 / (1 + abs(x - n/2)) for x in range(1, n + 1)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            candidates = list(range(1, n + 1))
            # Reorder based on weighted random selection
            selected = []
            remaining_idx = list(range(len(candidates)))
            for _ in range(len(candidates)):
                if not remaining_idx:
                    break
                r = random.random()
                cumulative = 0
                for idx in remaining_idx:
                    cumulative += weights[idx]
                    if cumulative >= r:
                        selected.append(candidates[idx])
                        remaining_idx.remove(idx)
                        break
            candidates = selected
        
        S = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S, existing_sums):
                S.append(x)
                existing_sums.update(get_new_sums(x, S[:-1]))
        
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Run 4: Local search - try to replace elements to fit more
    if time.time() < deadline - 5.0:
        current = best_set.copy()
        if current:
            current_sums = set()
            S_sorted = sorted(current)
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                current_sums.add(3 * a)
                for j in range(i, len(S_sorted)):
                    b = S_sorted[j]
                    current_sums.add(2 * a + b)
                    for k in range(j, len(S_sorted)):
                        c = S_sorted[k]
                        current_sums.add(a + b + c)
            
            # Try to add elements that were skipped
            used = set(current)
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in used:
                    continue
                if is_valid_addition(x, current, current_sums):
                    current.append(x)
                    current_sums.update(get_new_sums(x, current[:-1]))
                    used.add(x)
            
            if len(current) > len(best_set):
                best_set = current.copy()
    
    # Run 5: Try to improve by removing and re-adding elements
    if time.time() < deadline - 10.0:
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            current = best_set.copy()
            current_sums = set()
            S_sorted = sorted(current)
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                current_sums.add(3 * a)
                for j in range(i, len(S_sorted)):
                    b = S_sorted[j]
                    current_sums.add(2 * a + b)
                    for k in range(j, len(S_sorted)):
                        c = S_sorted[k]
                        current_sums.add(a + b + c)
            
            # Try removing one element and see if we can add two new ones
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                
                removed = current[i]
                temp_set = current[:i] + current[i+1:]
                
                # Recompute sums for temp_set
                temp_sums = set()
                temp_sorted = sorted(temp_set)
                for ii in range(len(temp_sorted)):
                    a = temp_sorted[ii]
                    temp_sums.add(3 * a)
                    for jj in range(ii, len(temp_sorted)):
                        b = temp_sorted[jj]
                        temp_sums.add(2 * a + b)
                        for kk in range(jj, len(temp_sorted)):
                            c = temp_sorted[kk]
                            temp_sums.add(a + b + c)
                
                # Try to add new elements
                new_elements = []
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if x in set(temp_set) or x == removed:
                        continue
                    if is_valid_addition(x, temp_set, temp_sums):
                        new_elements.append(x)
                        temp_set.append(x)
                        temp_sums.update(get_new_sums(x, temp_set[:-1]))
                        if len(new_elements) >= 2:
                            break
                
                if len(new_elements) >= 2:
                    # We improved by 1
                    current = temp_set
                    current_sums = temp_sums
                    improved = True
                    break
            
            if len(current) > len(best_set):
                best_set = current.copy()
    
    return best_set
