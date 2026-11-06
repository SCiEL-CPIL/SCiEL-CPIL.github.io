import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    N = 128
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # 2. Precompute bad triples and conflict masks
    # bad_z[x][y] is a bitmask of z such that {x, y, z} is a bad triple
    # We only need x < y for storage, but access will be symmetric.
    # Actually, let's store for all pairs (x,y) with x != y.
    bad_z = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Identify all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    # Mark conflicts
                    bad_z[a][b] |= (1 << c)
                    bad_z[b][a] |= (1 << c)
                    bad_z[a][c] |= (1 << b)
                    bad_z[c][a] |= (1 << b)
                    bad_z[b][c] |= (1 << a)
                    bad_z[c][b] |= (1 << a)

    # 3. Branch and Bound
    # State: 
    #   current_mask: bitmask of selected elements
    #   blocked_mask: bitmask of elements that CANNOT be added because they form a bad triple with some pair in current_mask
    #   next_idx: index of the next element to consider (0..N)
    #   current_count: number of elements in current_mask
    
    # To improve pruning, we order elements.
    # Elements with higher "conflict degree" should be decided earlier?
    # Or lower? Usually, in MIS, high degree nodes are processed first to prune early.
    # Degree of x = number of y such that bad_z[x][y] != 0.
    degrees = [0] * (N + 1)
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x != y and bad_z[x][y]:
                degrees[x] += 1
    # Sort indices by degree descending
    order = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    # Map original index to position in order? 
    # It's easier to just iterate through the sorted list.
    # Let's use a recursive function that takes position in the 'order' list.
    
    best_count = 0
    best_mask = 0
    
    # Precompute for each position i in 'order', the element value
    # We will recurse on position p in the 'order' array.
    # available_elements are those not yet processed and not blocked.
    # Actually, standard B&B on ordered vertices:
    # We process vertices in a fixed order.
    # At step p, we decide whether to include order[p].
    # We maintain 'current_mask' and 'blocked_mask'.
    # 'blocked_mask' accumulates constraints from 'current_mask'.
    # If we include v = order[p], we update blocked_mask |= OR(bad_z[v][u] for u in current_mask).
    # Note: bad_z[v][u] might contain elements with index < p (already processed) or > p (future).
    # We only care about future elements being blocked.
    # So we can mask bad_z[v][u] with 'future_mask' to only update relevant bits.
    
    # Precompute future_mask for each position p
    # future_mask[p] has bits set for order[p+1], order[p+2], ...
    future_masks = [0] * N
    m = 0
    for p in range(N - 1, -1, -1):
        future_masks[p] = m
        m |= (1 << order[p])
    
    def dfs(p, current_mask, blocked_mask, current_count):
        nonlocal best_count, best_mask
        
        # Time check
        if time.time() > deadline:
            return
        
        # Upper bound: current_count + number of remaining elements in order[p:]
        remaining = N - p
        if current_count + remaining <= best_count:
            return
            
        if p == N:
            if current_count > best_count:
                best_count = current_count
                best_mask = current_mask
            return
        
        v = order[p]
        v_bit = 1 << v
        
        # Option 1: Exclude v
        # No change to current_mask or blocked_mask
        dfs(p + 1, current_mask, blocked_mask, current_count)
        
        # Option 2: Include v
        # Check if v is blocked
        if blocked_mask & v_bit:
            return
            
        # If not blocked, we can include it
        # New blocked_mask: 
        # We need to add constraints from v with all u in current_mask.
        # new_blocked = blocked_mask
        # For each u in current_mask: new_blocked |= bad_z[v][u]
        # To speed this up, we can precompute something?
        # Or just iterate bits of current_mask.
        # current_mask is sparse? Maybe not.
        # N=128. Iterating bits of a 128-bit int is fast in Python if we use bit tricks or just loop 1..N.
        # Since N is small, looping 1..N to check if u is in current_mask is O(N) = 128 ops.
        # This is acceptable for depth 128.
        
        new_blocked = blocked_mask
        # Iterate through all u in current_mask
        temp_mask = current_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            new_blocked |= bad_z[v][u]
            temp_mask ^= lsb
            
        # We only care about blocking future elements
        # But keeping the whole mask is fine, just ensure we check v_bit correctly.
        # Actually, we don't need to mask with future_masks because we check 'blocked_mask & v_bit' 
        # for the specific v we are about to process. 
        # However, if we block an element that was already processed (excluded), it doesn't matter.
        # If we block an element that will be processed later, it prevents it from being included.
        # So we just pass new_blocked.
        
        new_current_mask = current_mask | v_bit
        dfs(p + 1, new_current_mask, new_blocked, current_count + 1)

    # Start search
    # Initial state: p=0, current_mask=0, blocked_mask=0, current_count=0
    dfs(0, 0, 0, 0)
    
    # Extract result
    if best_count == 0:
        return [1]
        
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    # Fallback if result is empty (should not happen)
    if not result:
        return [1]
        
    return result

# Quick sanity check for validity in case of bugs in B&B logic (should be fast for N=128 if result is small)
def _validate(result):
    from itertools import combinations
    res_set = set(result)
    for a, b, c in combinations(res_set, 3):
        if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
            return False
    return True

# We don't call _validate inside construct to save time, but the B&B logic is sound if bad_z is correct.
