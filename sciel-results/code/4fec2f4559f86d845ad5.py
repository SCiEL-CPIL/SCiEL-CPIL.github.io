import time
import random
import math

def construct():
    t0 = time.time()
    # Use a safety margin of 15s to ensure we return well before 60s
    DEADLINE = t0 + 45.0
    
    N = 256
    
    # Precompute conflict masks
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Precompute conflict degree for each node
    conflict_degree = [bin(conflict_mask[i]).count('1') for i in range(N)]
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def get_size(mask):
        return bin(mask).count('1')
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0
    
    def valid_set(mask):
        indices = [i for i in range(N) if (mask >> i) & 1]
        for ii in range(len(indices)):
            for jj in range(ii+1, len(indices)):
                a = indices[ii]
                b = indices[jj]
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy construction
    sorted_by_degree = sorted(range(N), key=lambda i: conflict_degree[i])
    
    # Greedy: add nodes in order of increasing conflict degree
    mask = 0
    for i in sorted_by_degree:
        if can_add(i, mask):
            mask |= (1 << i)
    sz = get_size(mask)
    if sz > best_size:
        best_mask = mask
        best_size = sz
    
    # Greedy: start with all odds, then greedily add evens
    mask = 0
    for i in odds_indices:
        mask |= (1 << i)
    
    def conflicts_with_mask(idx, m):
        return bin(conflict_mask[idx] & m).count('1')
    
    evens_sorted = sorted(evens_indices, key=lambda i: conflicts_with_mask(i, mask))
    for i in evens_sorted:
        if can_add(i, mask):
            mask |= (1 << i)
    sz = get_size(mask)
    if sz > best_size:
        best_mask = mask
        best_size = sz
    
    # Check time before starting local search
    if time.time() > DEADLINE - 10:
        result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
        return result
    
    # Strategy 2: Local search (reduced iterations)
    current_mask = best_mask
    current_size = best_size
    
    def neighborhood_move(mask, size):
        """Fast neighborhood move: try 1-remove 2-add, then 1-remove 1-add."""
        indices_in = [i for i in range(N) if (mask >> i) & 1]
        indices_out = [i for i in range(N) if not ((mask >> i) & 1)]
        
        if not indices_in or not indices_out:
            return mask, size, False
        
        # Try removing 1, adding 2 (net +1) - most valuable
        for _ in range(10):
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            # Build candidate list
            candidates = []
            for i in indices_out:
                if can_add(i, trial):
                    candidates.append(i)
            
            if len(candidates) >= 2:
                # Try to find 2 non-conflicting candidates
                for _ in range(15):
                    e1_idx = random.randint(0, len(candidates) - 1)
                    e2_idx = random.randint(0, len(candidates) - 1)
                    if e1_idx == e2_idx:
                        continue
                    e1 = candidates[e1_idx]
                    e2 = candidates[e2_idx]
                    if not ((conflict_mask[e1] >> e2) & 1):
                        new_mask = trial | (1 << e1) | (1 << e2)
                        return new_mask, size + 1, True
        
        # Try removing 1, adding 1 (net 0, for exploration)
        for _ in range(5):
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            candidates = []
            for i in indices_out:
                if can_add(i, trial):
                    candidates.append(i)
            
            if candidates:
                e1 = random.choice(candidates)
                new_mask = trial | (1 << e1)
                return new_mask, size, True
        
        return mask, size, False
    
    # Local search with reduced iterations
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    max_iter = 50000  # Reduced from 500000
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        new_mask, new_size, made_move = neighborhood_move(current_mask, current_size)
        
        if made_move:
            if new_size >= current_size:
                current_mask = new_mask
                current_size = new_size
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
            else:
                delta = current_size - new_size
                if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    current_mask = new_mask
                    current_size = new_size
        
        temperature = max(temperature * cooling, min_temp)
        
        if iteration % 5000 == 0 and temperature < 0.1:
            temperature = 0.5
    
    # Check time before restarts
    if time.time() > DEADLINE - 8:
        # Final polish
        for i in range(N):
            if not ((best_mask >> i) & 1) and can_add(i, best_mask):
                best_mask |= (1 << i)
        
        if not valid_set(best_mask):
            best_mask = 0
            for i in odds_indices:
                best_mask |= (1 << i)
        
        result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
        return result
    
    # Strategy 3: Fewer restarts with shorter search
    for restart in range(5):  # Reduced from 20
        if time.time() > DEADLINE - 3:
            break
        
        if restart % 2 == 0:
            # Start with all odds
            mask = 0
            for i in odds_indices:
                mask |= (1 << i)
        else:
            # Start with low-conflict-degree nodes
            mask = 0
            for i in sorted_by_degree[:128]:
                if can_add(i, mask):
                    mask |= (1 << i)
        
        # Greedy add
        for i in sorted_by_degree:
            if not ((mask >> i) & 1) and can_add(i, mask):
                mask |= (1 << i)
        
        sz = get_size(mask)
        if sz > best_size:
            best_mask = mask
            best_size = sz
        
        # Very short local search
        current_mask = mask
        current_size = sz
        temperature = 0.5
        
        for iteration in range(5000):  # Reduced from 50000
            if time.time() > DEADLINE - 2:
                break
            
            new_mask, new_size, made_move = neighborhood_move(current_mask, current_size)
            
            if made_move:
                if new_size >= current_size:
                    current_mask = new_mask
                    current_size = new_size
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                else:
                    delta = current_size - new_size
                    if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                        current_mask = new_mask
                        current_size = new_size
            
            temperature = max(temperature * cooling, min_temp)
    
    # Final polish
    for i in range(N):
        if not ((best_mask >> i) & 1) and can_add(i, best_mask):
            best_mask |= (1 << i)
    
    # Verify
    if not valid_set(best_mask):
        best_mask = 0
        for i in odds_indices:
            best_mask |= (1 << i)
        best_size = len(odds_indices)
    
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
