import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    MAX_SUM = 2 * N

    # Bitset representation
    # S_bits: integer where bit i is 1 if i is in S
    # Tight_bits: integer where bit k is 1 if counts[k] > 1
    
    def is_valid_add(S_bits, counts, x, N):
        # Check 2x
        if counts[2 * x] > 1:
            return False
        # Check x+y for y in S
        # (S_bits << x) has bits set at positions x+y for all y in S
        # We need to check if any of these positions k have counts[k] > 1
        # i.e., if (S_bits << x) & Tight_bits is non-zero
        # Note: We need to mask out bits beyond MAX_SUM?
        # S_bits has bits up to N. Shifted by x, bits go up to N+x <= 2N.
        # Tight_bits has bits up to 2N.
        # Python ints handle this fine.
        if (S_bits << x) & Tight_bits:
            return False
        return True

    def update_counts_add(counts, S, x):
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1

    def update_counts_remove(counts, S, x):
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def update_tight_bits(counts, Tight_bits, x, S, adding):
        # This is tricky to update incrementally for Tight_bits efficiently
        # Because counts change for many sums.
        # We can just recompute Tight_bits from counts? O(MAX_SUM) = 2000 ops.
        # This is fast enough.
        # Or we can update it carefully.
        # Let's just recompute it in the main loop if it's fast.
        pass

    # Phase 1: Greedy Initialization
    # We'll use the fast bitset check for the greedy pass too.
    best_S = set()
    best_S_bits = 0
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    best_Tight_bits = 0
    
    # Forward Greedy
    S = set()
    S_bits = 0
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    Tight_bits = 0
    
    for x in range(1, N + 1):
        if is_valid_add(S_bits, counts, x, N):
            S.add(x)
            S_bits |= (1 << x)
            update_counts_add(counts, S, x)
            # Update Tight_bits
            # We need to check which sums became tight
            # Sums changed: 2x, and x+y for y in S (old S before adding x? No, S includes x now in update_counts_add? 
            # My update_counts_add adds to S first? No, I passed S. Let's be careful.
            # Let's rewrite update to take old S.
            
    # Refining the update logic for Tight_bits
    # It's easier to just recompute Tight_bits after each add/remove if the set of changed sums is small.
    # Changed sums for adding x: {2x} U {x+y | y in S_old}
    # There are |S_old| + 1 such sums.
    # We can check these specific sums to update Tight_bits.
    
    def add_element(S, S_bits, counts, Tight_bits, x):
        # Update counts
        old_S = S.copy() # For tracking changed sums? No, we just need to know which sums changed.
        changed_sums = [2*x]
        for y in S:
            counts[x + y] += 1
            changed_sums.append(x + y)
        counts[2*x] += 1
        
        # Update Tight_bits
        for k in changed_sums:
            if counts[k] > 1:
                Tight_bits |= (1 << k)
            else:
                Tight_bits &= ~(1 << k)
                
        S.add(x)
        S_bits |= (1 << x)
        return S, S_bits, counts, Tight_bits

    def remove_element(S, S_bits, counts, Tight_bits, x):
        changed_sums = [2*x]
        for y in S:
            if y != x:
                counts[x + y] -= 1
                changed_sums.append(x + y)
        counts[2*x] -= 1
        
        # Update Tight_bits
        for k in changed_sums:
            if counts[k] > 1:
                Tight_bits |= (1 << k)
            else:
                Tight_bits &= ~(1 << k)
                
        S.discard(x)
        S_bits &= ~(1 << x)
        return S, S_bits, counts, Tight_bits

    # Phase 1: Greedy
    S, S_bits, counts, Tight_bits = set(), 0, np.zeros(MAX_SUM + 1, dtype=np.int32), 0
    for x in range(1, N + 1):
        if is_valid_add(S_bits, counts, x, N):
            S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
            
    best_S = S.copy()
    best_S_bits = S_bits
    best_counts = counts.copy()
    best_Tight_bits = Tight_bits
    best_size = len(S)
    
    # Phase 2: Local Search
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline - 10:
            break
            
        if len(S) < 2:
            break
            
        # Pick random element to remove
        old_x = random.choice(list(S))
        S, S_bits, counts, Tight_bits = remove_element(S, S_bits, counts, Tight_bits, old_x)
        
        # Try to find a valid element to add
        found = False
        # Scan candidates
        # To speed up, we can just iterate through 1..N
        # But we can also shuffle a list of candidates
        candidates = [x for x in range(1, N + 1) if x not in S and x != old_x]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(S_bits, counts, x, N):
                S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
                found = True
                break
                
        if not found:
            # Revert
            S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, old_x)
        else:
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_S_bits = S_bits
                best_counts = counts.copy()
                best_Tight_bits = Tight_bits

    # Phase 3: Final Polish - try to add any valid element
    S, S_bits, counts, Tight_bits = best_S, best_S_bits, best_counts.copy(), best_Tight_bits
    added = True
    while added and time.time() < deadline - 2:
        added = False
        for x in range(1, N + 1):
            if x not in S and is_valid_add(S_bits, counts, x, N):
                S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
                added = True
                break

    return list(S)
