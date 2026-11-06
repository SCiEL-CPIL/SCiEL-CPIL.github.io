import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask
    # conflict_mask[i] has bit j set if i and j conflict
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

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    # Helper: check if adding idx is valid
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    # Helper: get size
    def get_size(mask):
        return bin(mask).count('1')

    # Phase 1: Greedy Add Evens
    changed = True
    while changed:
        changed = False
        for e in evens_indices:
            if (current_mask >> e) & 1:
                continue
            if can_add(e, current_mask):
                current_mask |= (1 << e)
                changed = True
                sz = get_size(current_mask)
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz
        if time.time() > DEADLINE - 5:
            break

    # Phase 2: Targeted Local Search
    # For each even, find which odds it conflicts with
    # Then try to remove those odds and add multiple evens
    
    # Precompute for each even: list of odds it conflicts with
    even_conflicts = {}
    for e in evens_indices:
        conflicting_odds = [o for o in odds_indices if (conflict_mask[e] >> o) & 1]
        even_conflicts[e] = conflicting_odds
    
    # Sort evens by number of conflicts (fewer conflicts = easier to add)
    sorted_evens = sorted(evens_indices, key=lambda e: len(even_conflicts[e]))
    
    all_indices = list(range(N))
    
    max_iters = 500000  # Will be limited by time
    
    # Track recent moves to avoid cycling
    recent_states = set()
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        
        r = random.random()
        current_sz = get_size(current_mask)
        
        if r < 0.4:
            # Strategy 1: Try to add an even by removing its conflicting odds
            # Pick an even with few conflicts
            e = random.choice(sorted_evens[:50])  # Focus on easier evens
            if (current_mask >> e) & 1:
                continue
            
            # Get conflicting odds
            conflicting = [o for o in even_conflicts[e] if (current_mask >> o) & 1]
            
            if not conflicting:
                # Can add directly
                current_mask |= (1 << e)
                sz = get_size(current_mask)
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz
            elif len(conflicting) <= 2:
                # Remove conflicting odds and add this even
                # Also try to add other evens that become available
                trial_mask = current_mask
                for o in conflicting:
                    trial_mask &= ~(1 << o)
                
                # Try to add the target even
                if can_add(e, trial_mask):
                    trial_mask |= (1 << e)
                    
                    # Try to add other evens that were blocked by the removed odds
                    added_count = 1
                    for e2 in sorted_evens:
                        if e2 == e:
                            continue
                        if (trial_mask >> e2) & 1:
                            continue
                        if can_add(e2, trial_mask):
                            trial_mask |= (1 << e2)
                            added_count += 1
                    
                    new_sz = get_size(trial_mask)
                    # Accept if we gain size, or explore with small probability
                    if new_sz > current_sz:
                        current_mask = trial_mask
                        if new_sz > best_size:
                            best_mask = current_mask
                            best_size = new_sz
                    elif random.random() < 0.05:
                        current_mask = trial_mask
        
        elif r < 0.6:
            # Strategy 2: Remove 1-2 odds, add 2-3 evens
            # Pick odds that are "high value" (block many evens)
            # For each odd, count how many evens it blocks
            odd_block_count = {}
            for o in odds_indices:
                if not ((current_mask >> o) & 1):
                    continue
                block_count = 0
                for e in evens_indices:
                    if (conflict_mask[o] >> e) & 1:
                        block_count += 1
                odd_block_count[o] = block_count
            
            # Pick high-blocking odds to remove
            high_block_odds = [o for o in odds_indices 
                              if (current_mask >> o) & 1 and odd_block_count.get(o, 0) > 5]
            
            if not high_block_odds:
                continue
            
            if len(high_block_odds) >= 2 and random.random() < 0.5:
                # Remove 2 odds
                o1, o2 = random.sample(high_block_odds, 2)
            else:
                # Remove 1 odd
                o1 = random.choice(high_block_odds)
                o2 = None
            
            trial_mask = current_mask & ~(1 << o1)
            if o2 is not None:
                trial_mask &= ~(1 << o2)
            
            # Try to add as many evens as possible
            added = 0
            for e in sorted_evens:
                if (trial_mask >> e) & 1:
                    continue
                if can_add(e, trial_mask):
                    trial_mask |= (1 << e)
                    added += 1
            
            new_sz = get_size(trial_mask)
            if new_sz > current_sz:
                current_mask = trial_mask
                if new_sz > best_size:
                    best_mask = current_mask
                    best_size = new_sz
            elif new_sz == current_sz and random.random() < 0.1:
                current_mask = trial_mask
        
        elif r < 0.8:
            # Strategy 3: Simulated annealing style - accept worse moves to escape local optima
            to_remove = random.choice(all_indices)
            if not ((current_mask >> to_remove) & 1):
                continue
            
            trial_mask = current_mask & ~(1 << to_remove)
            
            # Try to add a random even
            e = random.choice(evens_indices)
            if (trial_mask >> e) & 1:
                continue
            
            if can_add(e, trial_mask):
                new_mask = trial_mask | (1 << e)
                # Size is same, accept with small probability to explore
                if random.random() < 0.15:
                    current_mask = new_mask
        
        else:
            # Strategy 4: Try to add multiple evens by removing their union of conflicts
            # Pick 2-3 evens, find all odds they conflict with, remove those odds, add the evens
            num_evens_to_try = random.choice([2, 3])
            candidates = random.sample(evens_indices, min(num_evens_to_try, len(evens_indices)))
            
            # Find union of conflicts
            all_conflicts = set()
            for e in candidates:
                for o in even_conflicts[e]:
                    if (current_mask >> o) & 1:
                        all_conflicts.add(o)
            
            if len(all_conflicts) > 3:
                continue  # Too many odds to remove
            
            trial_mask = current_mask
            for o in all_conflicts:
                trial_mask &= ~(1 << o)
            
            # Add the candidate evens
            added = 0
            for e in candidates:
                if can_add(e, trial_mask):
                    trial_mask |= (1 << e)
                    added += 1
            
            # Also try to add other evens that become available
            for e2 in sorted_evens:
                if e2 in candidates:
                    continue
                if (trial_mask >> e2) & 1:
                    continue
                if can_add(e2, trial_mask):
                    trial_mask |= (1 << e2)
                    added += 1
            
            new_sz = get_size(trial_mask)
            if new_sz > current_sz:
                current_mask = trial_mask
                if new_sz > best_size:
                    best_mask = current_mask
                    best_size = new_sz
            elif new_sz == current_sz and random.random() < 0.08:
                current_mask = trial_mask

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result

