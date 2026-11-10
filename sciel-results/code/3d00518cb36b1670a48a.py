import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline for 90s limit safety
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    all_subsets = list(range(1 << n))
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute subset/superset relationships using bit tricks
    # For each mask, precompute all its subsets and supersets
    # Actually, for speed, let's just use the bit operations directly
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def count_comparables_in_list(masks):
        count = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                a, b = masks[i], masks[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
    
    # Strategy: Multi-start with different initialization strategies
    # Then local search with simulated annealing on each
    
    def greedy_initial():
        """Greedy: add sets that maximize immediate comparable pairs"""
        masks = []
        mask_set = set()
        
        # Start with empty and full
        masks.append(0)
        mask_set.add(0)
        masks.append(full_mask)
        mask_set.add(full_mask)
        
        # Greedily add sets
        while len(masks) < m:
            best_gain = -1
            best_mask = None
            for c in all_subsets:
                if c in mask_set:
                    continue
                gain = 0
                for o in masks:
                    if (c & o) == c or (c & o) == o:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = c
            if best_mask is None or best_gain <= 0:
                # Fill with random
                while len(masks) < m:
                    c = random.choice(all_subsets)
                    if c not in mask_set:
                        masks.append(c)
                        mask_set.add(c)
                break
            masks.append(best_mask)
            mask_set.add(best_mask)
        
        return masks
    
    def chain_initial():
        """Create a long chain and fill around it"""
        masks = []
        mask_set = set()
        
        # Build a maximal chain: empty, add one element at a time
        chain = [0]
        current = 0
        for i in range(n):
            current |= (1 << i)
            chain.append(current)
        
        # Add as many chain elements as we can
        for c in chain:
            if len(masks) >= m:
                break
            if c not in mask_set:
                masks.append(c)
                mask_set.add(c)
        
        # Fill remaining greedily
        while len(masks) < m:
            best_gain = -1
            best_mask = None
            for c in all_subsets:
                if c in mask_set:
                    continue
                gain = 0
                for o in masks:
                    if (c & o) == c or (c & o) == o:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = c
            if best_mask is None or best_gain <= 0:
                while len(masks) < m:
                    c = random.choice(all_subsets)
                    if c not in mask_set:
                        masks.append(c)
                        mask_set.add(c)
                break
            masks.append(best_mask)
            mask_set.add(best_mask)
        
        return masks
    
    def layered_initial():
        """Pick sets from multiple layers to maximize cross-layer comparability"""
        masks = []
        mask_set = set()
        
        # Group subsets by size
        by_size = [[] for _ in range(n + 1)]
        for c in all_subsets:
            by_size[popcount[c]].append(c)
        
        # We want to pick sets that are comparable across sizes
        # Strategy: pick a "spine" of sets where each is a subset of the next
        # Then add neighbors
        
        # Start with a random chain
        chain = [0]
        current = 0
        random_order = list(range(n))
        random.shuffle(random_order)
        for i in random_order:
            current |= (1 << i)
            chain.append(current)
        
        for c in chain:
            if len(masks) >= m:
                break
            if c not in mask_set:
                masks.append(c)
                mask_set.add(c)
        
        # Fill greedily
        while len(masks) < m:
            best_gain = -1
            best_mask = None
            for c in all_subsets:
                if c in mask_set:
                    continue
                gain = 0
                for o in masks:
                    if (c & o) == c or (c & o) == o:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = c
            if best_mask is None or best_gain <= 0:
                while len(masks) < m:
                    c = random.choice(all_subsets)
                    if c not in mask_set:
                        masks.append(c)
                        mask_set.add(c)
                break
            masks.append(best_mask)
            mask_set.add(best_mask)
        
        return masks
    
    def random_dense_initial():
        """Random initialization biased toward middle sizes"""
        masks = []
        mask_set = set()
        
        # Bias toward sizes 4-8 which have good comparable potential
        for _ in range(m):
            attempts = 0
            while attempts < 1000:
                # Sample from middle sizes preferentially
                size = random.choice([3,4,5,6,7,8,9])
                candidates = [c for c in all_subsets if popcount[c] == size]
                c = random.choice(candidates)
                if c not in mask_set:
                    masks.append(c)
                    mask_set.add(c)
                    break
                attempts += 1
            else:
                c = random.choice(all_subsets)
                while c in mask_set:
                    c = random.choice(all_subsets)
                masks.append(c)
                mask_set.add(c)
        
        return masks
    
    # Local search with simulated annealing
    def local_search(masks, time_limit):
        """Simulated annealing local search"""
        masks = list(masks)
        mask_set = set(masks)
        N = len(masks)
        
        # Build comparability matrix
        comp = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                a, b = masks[i], masks[j]
                if (a & b) == a or (a & b) == b:
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        # Compute current score
        score = sum(comp[i][j] for i in range(N) for j in range(i + 1, N))
        
        best_masks = list(masks)
        best_score = score
        
        # Simulated annealing parameters
        T = 10.0  # initial temperature
        T_min = 0.1
        cooling = 0.99995
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < T_min:
                T = T_min
            
            # Pick a random index to replace
            idx = random.randint(0, N - 1)
            old_mask = masks[idx]
            
            # Pick a random replacement
            new_mask = random.choice(all_subsets)
            
            if new_mask == old_mask:
                continue
            
            if new_mask in mask_set:
                continue
            
            # Calculate delta
            old_contrib = sum(comp[idx][j] for j in range(N) if j != idx)
            
            new_contrib = 0
            for j in range(N):
                if j == idx:
                    continue
                o = masks[j]
                if (new_mask & o) == new_mask or (new_mask & o) == o:
                    new_contrib += 1
            
            delta = new_contrib - old_contrib
            
            # Metropolis criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept the move
                for j in range(N):
                    if j == idx:
                        continue
                    comp[idx][j] = 0
                    comp[j][idx] = 0
                    o = masks[j]
                    if (new_mask & o) == new_mask or (new_mask & o) == o:
                        comp[idx][j] = 1
                        comp[j][idx] = 1
                
                masks[idx] = new_mask
                mask_set.remove(old_mask)
                mask_set.add(new_mask)
                score += delta
                
                if score > best_score:
                    best_score = score
                    best_masks = list(masks)
            
            T *= cooling
        
        return best_masks, best_score
    
    # Multi-start local search
    initializers = [
        greedy_initial,
        chain_initial,
        layered_initial,
        random_dense_initial,
    ]
    
    best_overall = None
    best_overall_score = -1
    
    # Run each initializer and then local search
    num_starts = 4
    time_per_start = 40.0 / num_starts  # ~10s per start
    
    for init_func in initializers:
        if time.time() > deadline - 5.0:
            break
        
        init_masks = init_func()
        init_score = count_comparables_in_list(init_masks)
        
        # Local search
        ls_time = min(time_per_start, deadline - time.time() - 5.0)
        if ls_time < 1.0:
            break
        
        result_masks, result_score = local_search(init_masks, ls_time)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_masks
    
    # If we haven't found anything, use greedy
    if best_overall is None:
        best_overall = greedy_initial()
    
    # Final polish: try a few more local search iterations on the best
    if time.time() < deadline - 3.0:
        polish_time = min(5.0, deadline - time.time() - 3.0)
        if polish_time > 0.5:
            result_masks, result_score = local_search(best_overall, polish_time)
            if result_score > best_overall_score:
                best_overall = result_masks
                best_overall_score = result_score
    
    # Convert to list of lists
    result = []
    for mask in best_overall:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
    
    return result
