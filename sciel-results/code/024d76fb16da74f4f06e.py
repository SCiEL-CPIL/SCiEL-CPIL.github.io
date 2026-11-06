"""
Uses a bitset (Python integer) to track reachable sums up to TARGET=200.
Starts with upper half {51..100}, breaks forbidden triplets via hitting set,
then greedily adds lower half elements using bitset feasibility checks.
"""
import time

TARGET = 200
MASK = (1 << (TARGET + 1)) - 1  # Bits 0..200

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin under 90s limit
    
    # Phase 1: Initialize with upper half {51, ..., 100}
    upper_half = list(range(51, 101))
    selected = set(upper_half)
    
    # Find all forbidden triplets in upper_half
    forbidden_triplets = []
    n = len(upper_half)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if upper_half[i] + upper_half[j] + upper_half[k] == TARGET:
                    forbidden_triplets.append((i, j, k))
    
    # Phase 2: Greedy Hitting Set to remove elements from upper_half
    in_set = [True] * n
    active_triplets = list(forbidden_triplets)
    
    while active_triplets:
        if time.time() > deadline:
            break
        # Count frequency of each index in active_triplets
        freq = [0] * n
        for t in active_triplets:
            for idx in t:
                if in_set[idx]:
                    freq[idx] += 1
        
        # Find index with max frequency
        max_idx = -1
        max_freq = -1
        for i in range(n):
            if in_set[i] and freq[i] > max_freq:
                max_freq = freq[i]
                max_idx = i
        
        if max_idx == -1:
            break
            
        # Remove this index
        in_set[max_idx] = False
        selected.discard(upper_half[max_idx])
        
        # Update active_triplets: remove any triplet containing max_idx
        active_triplets = [t for t in active_triplets if max_idx not in t]
        
    # Phase 3: Build reachable sums bitset for current selected set
    # Bit s is set if sum s is reachable
    reachable = 1  # Only sum 0 is reachable initially (bit 0 set)
    for x in selected:
        reachable |= (reachable << x) & MASK
    
    # Verify no 200
    if (reachable >> TARGET) & 1:
        return [1]
    
    # Phase 4: Greedily add elements from lower half {1, ..., 50}
    # Try multiple orderings to maximize cardinality
    best_selected = set(selected)
    best_reachable = reachable
    best_card = len(selected)
    
    # Try descending order
    for x in range(50, 0, -1):
        if x in selected:
            continue
        # Check if adding x creates sum TARGET
        # Adding x is safe if (TARGET - x) is NOT reachable
        if not ((reachable >> (TARGET - x)) & 1):
            selected.add(x)
            reachable |= (reachable << x) & MASK
            if len(selected) > best_card:
                best_card = len(selected)
                best_selected = set(selected)
                best_reachable = reachable
    
    # Try ascending order
    selected = set(upper_half)
    in_set = [True] * n
    active_triplets = list(forbidden_triplets)
    while active_triplets:
        if time.time() > deadline:
            break
        freq = [0] * n
        for t in active_triplets:
            for idx in t:
                if in_set[idx]:
                    freq[idx] += 1
        max_idx = -1
        max_freq = -1
        for i in range(n):
            if in_set[i] and freq[i] > max_freq:
                max_freq = freq[i]
                max_idx = i
        if max_idx == -1:
            break
        in_set[max_idx] = False
        selected.discard(upper_half[max_idx])
        active_triplets = [t for t in active_triplets if max_idx not in t]
    
    reachable = 1
    for x in selected:
        reachable |= (reachable << x) & MASK
    
    if not ((reachable >> TARGET) & 1):
        for x in range(1, 51):
            if x in selected:
                continue
            if not ((reachable >> (TARGET - x)) & 1):
                selected.add(x)
                reachable |= (reachable << x) & MASK
                if len(selected) > best_card:
                    best_card = len(selected)
                    best_selected = set(selected)
                    best_reachable = reachable
    
    # Try random orderings
    import random
    rng = random.Random(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        selected = set(upper_half)
        in_set = [True] * n
        active_triplets = list(forbidden_triplets)
        while active_triplets:
            if time.time() > deadline:
                break
            freq = [0] * n
            for t in active_triplets:
                for idx in t:
                    if in_set[idx]:
                        freq[idx] += 1
            max_idx = -1
            max_freq = -1
            for i in range(n):
                if in_set[i] and freq[i] > max_freq:
                    max_freq = freq[i]
                    max_idx = i
            if max_idx == -1:
                break
            in_set[max_idx] = False
            selected.discard(upper_half[max_idx])
            active_triplets = [t for t in active_triplets if max_idx not in t]
        
        reachable = 1
        for x in selected:
            reachable |= (reachable << x) & MASK
        
        if not ((reachable >> TARGET) & 1):
            candidates = list(range(1, 51))
            rng.shuffle(candidates)
            for x in candidates:
                if x in selected:
                    continue
                if not ((reachable >> (TARGET - x)) & 1):
                    selected.add(x)
                    reachable |= (reachable << x) & MASK
                    if len(selected) > best_card:
                        best_card = len(selected)
                        best_selected = set(selected)
                        best_reachable = reachable
    
    # Final verification
    if ((best_reachable >> TARGET) & 1):
        return [100, 99, 98, 97, 96]
    
    return list(best_selected)
