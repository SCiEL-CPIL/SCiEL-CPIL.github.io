"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline, 8s safety margin
    
    N = 256
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Use bitmask for speed
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Compute degrees
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    
    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected
    
    # Try degree-ordered greedy (ascending degree = fewer conflicts first)
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    best_mask = greedy_init(degree_order)
    best_size = bin(best_mask).count('1')
    
    # Also try the base set: numbers ≡ 1 mod 3 (filtered for forbidden)
    base_indices = [i for i, x in enumerate(valid_numbers) if x % 3 == 1]
    base_mask = 0
    for i in base_indices:
        base_mask |= (1 << i)
    base_size = bin(base_mask).count('1')
    if base_size > best_size:
        best_mask = base_mask
        best_size = base_size
    
    # Try several random orderings for initialization
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        mask = greedy_init(random_order)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_mask = mask
            best_size = sz
    
    # Local search: iterated improvement
    def local_search(current_mask, current_size, max_iters):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if current_size == 0:
                break
            
            sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
            if not sel_indices:
                break
            
            # Try removing one random selected element
            remove_idx = random.choice(sel_indices)
            trial_mask = current_mask & ~(1 << remove_idx)
            trial_size = current_size - 1
            
            # Try to add elements greedily (in random order)
            unselected = [i for i in range(M) if not (trial_mask >> i) & 1]
            random.shuffle(unselected)
            
            added = 0
            for i in unselected:
                if (adj[i] & trial_mask) == 0:
                    trial_mask |= (1 << i)
                    trial_size += 1
                    added += 1
                if added >= 3:
                    break
            
            if trial_size > current_size:
                current_mask = trial_mask
                current_size = trial_size
        
        return current_mask, current_size
    
    # Run local search from best
    current_mask, current_size = local_search(best_mask, best_size, 30000)
    if current_size > best_size:
        best_mask = current_mask
        best_size = current_size
    
    # More iterations with different random seeds
    for seed in [123, 456, 789, 101, 202]:
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        # Random restart
        random_order = list(range(M))
        random.shuffle(random_order)
        mask = greedy_init(random_order)
        sz = bin(mask).count('1')
        
        # Local search from this restart
        mask, sz = local_search(mask, sz, 20000)
        
        if sz > best_size:
            best_mask = mask
            best_size = sz
    
    # Final polish: more local search from best
    random.seed(999)
    current_mask, current_size = local_search(best_mask, best_size, 20000)
    if current_size > best_size:
        best_mask = current_mask
        best_size = current_size
    
    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
