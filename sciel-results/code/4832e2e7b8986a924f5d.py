import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    def fast_popcount(x):
        return x.bit_count()
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = fast_popcount(current_mask)
    
    # Phase 1: Add evens with 0 conflicts, then try evens with minimal conflicts
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True
        # Also try odds that might have been excluded
        for o in odds_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> o) & 1:
                continue
            if (conflict_mask[o] & current_mask) == 0:
                current_mask |= (1 << o)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True
    
    # Phase 1b: Try adding evens that cause minimal conflicts (remove those conflicts, add even)
    changed = True
    while changed and time.time() < DEADLINE - 45:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 45:
                break
            if (current_mask >> e) & 1:
                continue
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            # Only add if net gain (remove k conflicts, add 1, net = 1-k)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << e)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True
        # Also try odds with minimal conflicts
        for o in odds_indices:
            if time.time() > DEADLINE - 45:
                break
            if (current_mask >> o) & 1:
                continue
            conflicts = conflict_mask[o] & current_mask
            num_conflicts = fast_popcount(conflicts)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << o)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True
    
    # Phase 2: Simulated Annealing with improved moves
    T = 3.0
    cooling = 0.99985
    max_iters = 500000
    no_improve_limit = 8000
    
    current_size = best_size
    no_improve = 0
    restart_count = 0
    max_restarts = 12
    
    # Precompute bit lists for quick access
    all_indices = list(range(N))
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        r = random.random()
        if r < 0.45:
            # Single add move: try to add a random element not in set
            # Prefer evens since they're more likely to have conflicts
            if random.random() < 0.7:
                # Pick random even not in set
                e = random.choice(evens_indices)
                if (current_mask >> e) & 1:
                    continue
            else:
                o = random.choice(odds_indices)
                if (current_mask >> o) & 1:
                    continue
                e = o
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            new_size = current_size - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                no_improve = 0
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.5:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
                        no_improve += 1
        elif r < 0.65:
            # Swap move: remove 1 element, add 1 element
            # Pick random element from current set
            for _ in range(50):
                rm_idx = random.randint(0, N - 1)
                if (current_mask >> rm_idx) & 1:
                    break
            else:
                continue
            
            temp_mask = current_mask & ~(1 << rm_idx)
            
            # Try to find a zero-conflict candidate to add
            found = False
            for _ in range(30):
                e = random.randint(0, N - 1)
                if (temp_mask >> e) & 1:
                    continue
                if (conflict_mask[e] & temp_mask) == 0:
                    current_mask = temp_mask | (1 << e)
                    found = True
                    no_improve = 0
                    break
            if not found:
                # Try any candidate with low conflict
                for _ in range(10):
                    e = random.randint(0, N - 1)
                    if (temp_mask >> e) & 1:
                        continue
                    c = conflict_mask[e] & temp_mask
                    if fast_popcount(c) <= 1:
                        current_mask = temp_mask | (1 << e)
                        found = True
                        no_improve += 1
                        break
        elif r < 0.80:
            # 2-swap move: remove 2 elements, add 2 elements
            # Pick 2 random elements from current set
            rm1 = rm2 = -1
            for _ in range(100):
                rm1 = random.randint(0, N - 1)
                if (current_mask >> rm1) & 1:
                    break
            if rm1 < 0:
                continue
            for _ in range(100):
                rm2 = random.randint(0, N - 1)
                if rm2 != rm1 and (current_mask >> rm2) & 1:
                    break
            if rm2 < 0:
                continue
            
            temp_mask = current_mask & ~(1 << rm1) & ~(1 << rm2)
            
            # Find two zero-conflict candidates
            c1 = c2 = -1
            for _ in range(40):
                idx = random.randint(0, N - 1)
                if (temp_mask >> idx) & 1:
                    continue
                if (conflict_mask[idx] & temp_mask) == 0:
                    c1 = idx
                    break
            if c1 < 0:
                continue
            for _ in range(40):
                idx = random.randint(0, N - 1)
                if (temp_mask >> idx) & 1 or idx == c1:
                    continue
                if (conflict_mask[idx] & temp_mask) == 0 and not ((conflict_mask[c1] >> idx) & 1):
                    c2 = idx
                    break
            if c2 < 0:
                continue
            
            current_mask = temp_mask | (1 << c1) | (1 << c2)
            no_improve = 0
        elif r < 0.92:
            # Multi-element exchange: remove k elements, add 1
            k = random.choice([2, 3])
            removed = []
            for _ in range(k):
                for _2 in range(50):
                    rm = random.randint(0, N - 1)
                    if (current_mask >> rm) & 1 and rm not in removed:
                        removed.append(rm)
                        break
                else:
                    continue
            
            if len(removed) < k:
                continue
            
            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)
            
            # Find a candidate with 0 conflicts
            e = -1
            for _ in range(40):
                idx = random.randint(0, N - 1)
                if (temp_mask >> idx) & 1:
                    continue
                if (conflict_mask[idx] & temp_mask) == 0:
                    e = idx
                    break
            
            if e < 0:
                # Try low-conflict candidate
                for _ in range(20):
                    idx = random.randint(0, N - 1)
                    if (temp_mask >> idx) & 1:
                        continue
                    c = conflict_mask[idx] & temp_mask
                    if fast_popcount(c) <= 1:
                        e = idx
                        break
            
            if e < 0:
                continue
            
            new_size = current_size - k + 1
            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                no_improve = 0
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
                        no_improve += 1
        else:
            # Random removal to escape local minima
            for _ in range(50):
                r_idx = random.randint(0, N - 1)
                if (current_mask >> r_idx) & 1:
                    break
            else:
                continue
            current_mask &= ~(1 << r_idx)
            current_size -= 1
            no_improve += 1
        
        T *= cooling
        if T < 0.05:
            T = 0.05
        
        # Restart if stuck
        if no_improve > no_improve_limit and restart_count < max_restarts:
            restart_count += 1
            no_improve = 0
            T = max(T, 2.0)
            # Perturb: remove a few random elements
            for _ in range(random.randint(2, 5)):
                for _2 in range(50):
                    rm = random.randint(0, N - 1)
                    if (current_mask >> rm) & 1:
                        current_mask &= ~(1 << rm)
                        current_size -= 1
                        break
            # Greedily re-add zero-conflict elements
            changed = True
            while changed:
                changed = False
                for idx in all_indices:
                    if (current_mask >> idx) & 1:
                        continue
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_size += 1
                        changed = True
            if current_size > best_size:
                best_mask = current_mask
                best_size = current_size

    # Final polish: greedy add any element with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 1:
                break
            if (best_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & best_mask) == 0:
                best_mask |= (1 << idx)
                best_size += 1
                changed = True
    
    # Second polish pass: try adding elements that cause at most 1 conflict if net gain
    changed = True
    while changed and time.time() < DEADLINE - 0.5:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 0.5:
                break
            if (best_mask >> idx) & 1:
                continue
            conflicts = conflict_mask[idx] & best_mask
            num_conf = fast_popcount(conflicts)
            if num_conf <= 1:
                new_mask = (best_mask & ~conflicts) | (1 << idx)
                new_size = best_size - num_conf + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                    changed = True
    
    # Third polish: try 2-element exchanges for net gain
    if time.time() < DEADLINE - 0.2:
        changed = True
        while changed and time.time() < DEADLINE - 0.2:
            changed = False
            for idx in range(N):
                if time.time() > DEADLINE - 0.2:
                    break
                if (best_mask >> idx) & 1:
                    continue
                conflicts = conflict_mask[idx] & best_mask
                num_conf = fast_popcount(conflicts)
                if num_conf == 2:
                    # Try removing both conflicts and adding idx
                    new_mask = (best_mask & ~conflicts) | (1 << idx)
                    new_size = best_size - 2 + 1
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                        changed = True

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
