import random
import time

def is_valid(S):
    K = len(S)
    # sums_by_cardinality
    sums = [set() for _ in range(K + 1)]
    # Iterate over all subsets
    # For K up to 20, 2^K is manageable.
    # Using bitmask
    for mask in range(1 << K):
        s = 0
        card = 0
        for i in range(K):
            if mask & (1 << i):
                s += S[i]
                card += 1
        sums[card].add(s)
        
    # Check intersections
    for i in range(K + 1):
        for j in range(i + 1, K + 1):
            if sums[i].intersection(sums[j]):
                return False
    return True

def count_conflicts(S):
    K = len(S)
    sums = [set() for _ in range(K + 1)]
    for mask in range(1 << K):
        s = 0
        card = 0
        for i in range(K):
            if mask & (1 << i):
                s += S[i]
                card += 1
        sums[card].add(s)
        
    conflicts = 0
    # Map sum to list of cardinalities
    sum_to_cards = {}
    for k in range(K + 1):
        for s in sums[k]:
            if s in sum_to_cards:
                sum_to_cards[s].append(k)
            else:
                sum_to_cards[s] = [k]
                
    for s, cards in sum_to_cards.items():
        if len(cards) > 1:
            # Count pairs of cardinalities that conflict
            conflicts += len(cards) * (len(cards) - 1) // 2
    return conflicts

def find_set(K, time_limit=5.0):
    start_time = time.time()
    best_set = None
    best_conflicts = float('inf')
    
    while time.time() - start_time < time_limit:
        # Random initialization
        S = random.sample(range(1, 257), K)
        conflicts = count_conflicts(S)
        
        if conflicts == 0:
            return S
        
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_set = S[:]
            
        # Local search
        current_S = S[:]
        current_conflicts = conflicts
        no_improve = 0
        while time.time() - start_time < time_limit:
            # Pick random index to change
            idx = random.randint(0, K - 1)
            old_val = current_S[idx]
            
            # Pick new value
            new_val = random.randint(1, 256)
            if new_val in current_S:
                continue
                
            current_S[idx] = new_val
            new_conflicts = count_conflicts(current_S)
            
            if new_conflicts < current_conflicts:
                current_conflicts = new_conflicts
                no_improve = 0
                if new_conflicts == 0:
                    return current_S
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_set = current_S[:]
            else:
                # Revert
                current_S[idx] = old_val
                no_improve += 1
                if no_improve > 100:
                    break
        
    return best_set

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    best_size = 0
    best_result = []
    
    # Start with known valid size 9
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256
    initial = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if is_valid(initial):
        best_size = 9
        best_result = initial[:]
        
    # Try to find larger sets
    for K in range(10, 30):
        if time.time() - start_time > deadline:
            break
            
        # Allocate remaining time proportionally or fixed chunk
        chunk_time = min(5.0, deadline - (time.time() - start_time))
        if chunk_time < 1.0:
            break
            
        S = find_set(K, chunk_time)
        if S is not None and is_valid(S):
            best_size = K
            best_result = S
            # Continue to try larger K
        else:
            # If we fail to find K, maybe K+1 is possible? 
            # But usually if K is hard, K+1 is harder.
            # However, sometimes a specific K is hard due to random chance.
            # Let's try one more time for K with less time?
            # Or just stop?
            # Given the "superincreasing" lower bound logic, if we can't find 10, 
            # it might be impossible or very rare.
            # But let's try K+1 just in case? No, that's risky.
            # Let's assume monotonic difficulty for now.
            break
            
    # Ensure types are correct
    return [int(x) for x in best_result]
