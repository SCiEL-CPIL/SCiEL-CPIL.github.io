import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safe margin before 90s kill

    # Precompute for each x (0-indexed) the list of conflict masks.
    # A conflict mask for x is a bitmask of 3 other elements such that
    # if all 3 are in S, then x cannot be added (they form a 4-AP with x).
    conflict_masks = [[] for _ in range(N)]
    
    for d in range(1, (N // 3) + 1):
        for a in range(0, N - 3 * d):
            a0 = a
            a1 = a + d
            a2 = a + 2 * d
            a3 = a + 3 * d
            
            m0 = (1 << a1) | (1 << a2) | (1 << a3)
            m1 = (1 << a0) | (1 << a2) | (1 << a3)
            m2 = (1 << a0) | (1 << a1) | (1 << a3)
            m3 = (1 << a0) | (1 << a1) | (1 << a2)
            
            conflict_masks[a0].append(m0)
            conflict_masks[a1].append(m1)
            conflict_masks[a2].append(m2)
            conflict_masks[a3].append(m3)
    
    # Convert to tuples for faster iteration
    conflict_masks = [tuple(m) for m in conflict_masks]
    
    # Precompute degree (number of conflict masks) for ordering
    degrees = [len(cm) for cm in conflict_masks]
    
    def can_add(x, S_mask):
        """Check if x (0-indexed) can be added to set represented by S_mask (int bitmask)"""
        if (S_mask >> x) & 1:
            return False
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True
    
    def get_size(S_mask):
        return bin(S_mask).count('1')
    
    def greedy_construct(seed, strategy='random'):
        """
        Greedy construction with different ordering strategies.
        strategy: 'random', 'low_degree', 'high_degree', 'random_weighted'
        """
        rng = random.Random(seed)
        S_mask = 0
        elements = list(range(N))
        
        if strategy == 'random':
            rng.shuffle(elements)
        elif strategy == 'low_degree':
            elements.sort(key=lambda x: degrees[x])
            rng.shuffle(elements)  # slight randomization
            # Actually, let's do a weighted shuffle based on degree
            elements = list(range(N))
            weights = [1.0 / (degrees[x] + 1) for x in elements]
            total_w = sum(weights)
            for i in range(len(elements) - 1, 0, -1):
                r = rng.random() * total_w
                cumsum = 0
                for j in range(i + 1):
                    cumsum += weights[j]
                    if cumsum >= r:
                        elements[i], elements[j] = elements[j], elements[i]
                        weights[i], weights[j] = weights[j], weights[i]
                        break
        elif strategy == 'high_degree':
            elements.sort(key=lambda x: -degrees[x])
            rng.shuffle(elements)
        elif strategy == 'random_weighted':
            # Weight inversely proportional to degree
            elements = list(range(N))
            weights = [1.0 / (degrees[x] + 1) for x in elements]
            total_w = sum(weights)
            for i in range(len(elements) - 1, 0, -1):
                r = rng.random() * total_w
                cumsum = 0
                for j in range(i + 1):
                    cumsum += weights[j]
                    if cumsum >= r:
                        elements[i], elements[j] = elements[j], elements[i]
                        weights[i], weights[j] = weights[j], weights[i]
                        break
        else:
            rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        
        return S_mask
    
    def local_search_swap(S_mask, rng, time_budget):
        """
        Local search using swap operations:
        Remove one element, then greedily try to add others.
        Also try single-add improvements.
        """
        current_size = get_size(S_mask)
        best_size = current_size
        best_S_mask = S_mask
        all_elements = list(range(N))
        no_improve_count = 0
        
        while time.time() - start_time < time_budget:
            if no_improve_count > 200:
                break
            
            improved = False
            
            # Try to add any element that can be added
            candidates = all_elements[:]
            rng.shuffle(candidates)
            for x in candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << x)
                    current_size += 1
                    improved = True
            
            # If we improved, update best
            if current_size > best_size:
                best_size = current_size
                best_S_mask = S_mask
                no_improve_count = 0
            elif not improved:
                no_improve_count += 1
            
            # Try swap: remove a random element, then greedily add
            if current_size > 0 and time.time() - start_time < time_budget:
                x_remove = rng.choice(all_elements)
                if (S_mask >> x_remove) & 1:
                    S_mask &= ~(1 << x_remove)
                    current_size -= 1
                    
                    # Try to add elements
                    candidates = all_elements[:]
                    rng.shuffle(candidates)
                    for x in candidates:
                        if can_add(x, S_mask):
                            S_mask |= (1 << x)
                            current_size += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_S_mask = S_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        # Revert if worse
                        if current_size < best_size:
                            S_mask = best_S_mask
                            current_size = best_size
        
        return best_S_mask
    
    def local_search_multi_swap(S_mask, rng, time_budget):
        """
        More aggressive local search: try multiple removals then rebuild.
        """
        current_size = get_size(S_mask)
        best_size = current_size
        best_S_mask = S_mask
        all_elements = list(range(N))
        no_improve_count = 0
        
        while time.time() - start_time < time_budget:
            if no_improve_count > 300:
                break
            
            # Try removing 1-3 random elements and rebuilding
            num_remove = rng.choice([1, 2, 3])
            removed = []
            new_S_mask = S_mask
            new_size = current_size
            
            for _ in range(num_remove):
                x = rng.choice(all_elements)
                if (new_S_mask >> x) & 1:
                    new_S_mask &= ~(1 << x)
                    new_size -= 1
                    removed.append(x)
            
            # Greedily add elements
            candidates = all_elements[:]
            rng.shuffle(candidates)
            for x in candidates:
                if can_add(x, new_S_mask):
                    new_S_mask |= (1 << x)
                    new_size += 1
            
            if new_size > best_size:
                best_size = new_size
                best_S_mask = new_S_mask
                S_mask = new_S_mask
                current_size = new_size
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return best_S_mask
    
    # Main optimization loop
    best_S_mask = 0
    best_size = 0
    
    # Try different greedy strategies
    strategies = ['random', 'low_degree', 'high_degree', 'random_weighted']
    for strategy in strategies:
        if time.time() - start_time > time_limit - 10:
            break
        S_mask = greedy_construct(0, strategy)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
    
    # Local search on best
    rng = random.Random(42)
    if time.time() - start_time < time_limit - 5:
        S_mask = local_search_swap(best_S_mask, rng, time_limit - 5)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
    
    # More restarts with local search
    seed = 1
    while time.time() - start_time < time_limit - 3:
        strategy = strategies[seed % len(strategies)]
        S_mask = greedy_construct(seed, strategy)
        size = get_size(S_mask)
        
        if size >= best_size - 5:
            S_mask = local_search_swap(S_mask, rng, time_limit - 3)
            size = get_size(S_mask)
        
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
        
        seed += 1
    
    # Final multi-swap local search
    if time.time() - start_time < time_limit - 1:
        S_mask = local_search_multi_swap(best_S_mask, rng, time_limit - 1)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
    
    # Convert bitmask to list of 1-indexed integers
    result = [i + 1 for i in range(N) if (best_S_mask >> i) & 1]
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
