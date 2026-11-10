import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or vice versa
    # Using a flat list for speed
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s
    
    # 1. Greedy Initialization with precomputed matrix
    # Start with empty set and full set, then greedily add sets that maximize comparability
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}
    
    for _ in range(m - 2):
        best_mask = -1
        best_score = -1
        for mask in range(num_masks):
            if mask in used_masks:
                continue
            score = 0
            base = mask * num_masks
            for cm in current_masks:
                if comp[base + cm]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            break

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # 2. Simulated Annealing with precomputed comparability
    random.seed(42)
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 5.0
        cooling_rate = 0.99995
        min_T = 0.001
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask: mostly small perturbations
            r = random.random()
            if r < 0.6:
                # Flip one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        # Remove a bit
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        # Add a bit
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            else:
                # Random mask
                new_mask = random.randint(0, num_masks - 1)
            
            if new_mask == old_mask:
                continue
            
            if new_mask in used_masks:
                continue
            
            # Calculate delta efficiently
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                old_comp = comp[old_mask * num_masks + current_masks[j]]
                new_comp = comp[new_mask * num_masks + current_masks[j]]
                delta += new_comp - old_comp
            
            if delta > 0:
                # Accept
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                # Simulated annealing acceptance
                if T > min_T:
                    p = random.random()
                    if p < min(1.0, (delta / T) if T > 0 else 0):
                        # Accept bad move
                        used_masks.remove(old_mask)
                        used_masks.add(new_mask)
                        current_masks[idx] = new_mask
                        current_score += delta
                        no_improve_steps += 1
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            # Cool down
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break
        
        restarts += 1
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
