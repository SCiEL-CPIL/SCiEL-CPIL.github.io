import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # 1. Precompute conflict masks
    # conflict_masks[x] is a list of bitmasks.
    # A bitmask m in conflict_masks[x] represents a pair {y, z} such that {x, y, z} is a forbidden triple.
    # If (mask_S & m) == m, then both y and z are in S, so x cannot be added.
    
    conflict_masks = [[] for _ in range(N + 1)]
    
    # Generate all triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # (b-a)(c-a) = a^2
    for a in range(1, N + 1):
        a2 = a * a
        # d1 = b-a, d2 = c-a
        # d1 * d2 = a^2
        # d1 < d2 to ensure b < c
        # d1 < a (since d1 < d2 and d1*d2=a^2, if d1 >= a then d2 <= a, so d1=d2=a, b=c=2a, not distinct)
        for d1 in range(1, a):
            if a2 % d1 == 0:
                d2 = a2 // d1
                if d1 == d2:
                    continue
                b = a + d1
                c = a + d2
                if b < c and c <= N:
                    # Triple (a, b, c)
                    # For a: conflict pair is (b, c)
                    # For b: conflict pair is (a, c)
                    # For c: conflict pair is (a, b)
                    
                    mask_bc = (1 << (b-1)) | (1 << (c-1))
                    mask_ac = (1 << (a-1)) | (1 << (c-1))
                    mask_ab = (1 << (a-1)) | (1 << (b-1))
                    
                    conflict_masks[a].append(mask_bc)
                    conflict_masks[b].append(mask_ac)
                    conflict_masks[c].append(mask_ab)
    
    def can_add(x, mask_S):
        for m in conflict_masks[x]:
            if (mask_S & m) == m:
                return False
        return True

    def get_blocking_pairs(x, mask_S):
        """Returns list of pairs (y, z) that block x"""
        blockers = []
        for m in conflict_masks[x]:
            if (mask_S & m) == m:
                # Extract indices
                # m has two bits set
                # We can precompute the pairs for each mask? 
                # Or just find the bits.
                # Since N=180, we can store the pair in a dict or parallel list.
                pass
        return blockers

    # To make get_blocking_pairs fast, let's store the pairs alongside the masks.
    # conflict_info[x] = list of (mask, y, z)
    conflict_info = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d1 in range(1, a):
            if a2 % d1 == 0:
                d2 = a2 // d1
                if d1 == d2:
                    continue
                b = a + d1
                c = a + d2
                if b < c and c <= N:
                    mask_bc = (1 << (b-1)) | (1 << (c-1))
                    mask_ac = (1 << (a-1)) | (1 << (c-1))
                    mask_ab = (1 << (a-1)) | (1 << (b-1))
                    
                    conflict_info[a].append((mask_bc, b, c))
                    conflict_info[b].append((mask_ac, a, c))
                    conflict_info[c].append((mask_ab, a, b))

    def can_add_fast(x, mask_S):
        for mask, y, z in conflict_info[x]:
            if (mask_S & mask) == mask:
                return False
        return True

    def get_blockers(x, mask_S):
        """Returns list of (y, z) pairs in S that block x"""
        blockers = []
        for mask, y, z in conflict_info[x]:
            if (mask_S & mask) == mask:
                blockers.append((y, z))
        return blockers

    # Search State
    best_mask = 0
    best_size = 0
    best_set = set()
    
    def update_best(mask, S):
        nonlocal best_mask, best_size, best_set
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            best_set = set(S)

    def run_local_search(initial_mask, initial_set, time_limit):
        mask = initial_mask
        S = set(initial_set)
        
        # Precompute list of all numbers
        all_nums = list(range(1, N + 1))
        
        start_time = time.time()
        stuck_count = 0
        
        while time.time() < start_time + time_limit:
            # Try to add
            added = False
            # Pick a random number not in S
            # Efficiently pick from complement
            # Since N is small, we can just pick random 1..N and check
            # Or maintain a list of excluded.
            
            # Let's try a few random adds
            for _ in range(10):
                x = random.choice(all_nums)
                if (mask >> (x-1)) & 1:
                    continue # Already in S
                
                if can_add_fast(x, mask):
                    mask |= (1 << (x-1))
                    S.add(x)
                    added = True
                    stuck_count = 0
                    update_best(mask, S)
                    break
            
            if added:
                continue
                
            # Try to swap
            # Pick a random x not in S that is blocked
            x = random.choice(all_nums)
            if (mask >> (x-1)) & 1:
                continue
                
            blockers = get_blockers(x, mask)
            if not blockers:
                # Should have been added
                mask |= (1 << (x-1))
                S.add(x)
                stuck_count = 0
                update_best(mask, S)
                continue
                
            # Collect all elements involved in blockers
            candidates_to_remove = set()
            for y, z in blockers:
                candidates_to_remove.add(y)
                candidates_to_remove.add(z)
                
            found_swap = False
            # Try removing one candidate
            cand_list = list(candidates_to_remove)
            random.shuffle(cand_list)
            
            for u in cand_list:
                # Temporarily remove u
                mask_temp = mask & ~(1 << (u-1))
                S_temp = S - {u}
                
                if can_add_fast(x, mask_temp):
                    # Perform swap
                    mask = mask_temp | (1 << (x-1))
                    S = S_temp
                    S.add(x)
                    found_swap = True
                    stuck_count = 0
                    # Size unchanged, so update_best might not trigger if size is same, 
                    # but we should still record if this specific configuration is "better" 
                    # in terms of potential? No, just keep it.
                    # However, if we find a swap that allows future adds, it's good.
                    break
            
            if found_swap:
                continue
                
            # Try kick: remove 2, add 1 (size -1)
            # This is expensive, do rarely
            if random.random() < 0.1:
                if len(candidates_to_remove) >= 2:
                    u, v = random.sample(cand_list, 2)
                    mask_temp = mask & ~(1 << (u-1)) & ~(1 << (v-1))
                    S_temp = S - {u, v}
                    
                    if can_add_fast(x, mask_temp):
                        mask = mask_temp | (1 << (x-1))
                        S = S_temp
                        S.add(x)
                        stuck_count = 0
                        # Size decreased, don't update best
                        continue
            
            # If nothing worked, increment stuck counter
            stuck_count += 1
            if stuck_count > 500:
                break # Restart from this state? Or just break inner loop?
                # Let's break to outer loop to restart with new seed

    # Multi-restart strategy
    random.seed(42)
    
    # Strategy 1: Upper half
    S_init = set(range(91, N + 1))
    mask_init = 0
    for x in S_init:
        mask_init |= (1 << (x-1))
    
    # Greedy fill
    for x in range(1, 91):
        if can_add_fast(x, mask_init):
            mask_init |= (1 << (x-1))
            S_init.add(x)
            
    update_best(mask_init, S_init)
    
    time_per_restart = 2.0
    restarts = 0
    
    while time.time() < deadline - 5:
        # Choose initial state
        if restarts % 3 == 0:
            # Use best found so far
            init_mask = best_mask
            init_set = set(best_set)
        elif restarts % 3 == 1:
            # Random subset
            init_set = set(random.sample(range(1, N + 1), random.randint(50, 100)))
            init_mask = 0
            for x in init_set:
                init_mask |= (1 << (x-1))
            # Greedy cleanup: remove conflicts?
            # It's better to start with a valid set.
            # Let's make it valid by removing conflicts.
            # This is O(N^2) roughly.
            # Actually, just start with empty and greedy add?
            # Let's stick to valid starts.
            # Generate a valid set by greedy adding from random permutation
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            init_set = set()
            init_mask = 0
            for x in perm:
                if can_add_fast(x, init_mask):
                    init_mask |= (1 << (x-1))
                    init_set.add(x)
        else:
            # Upper half + random
            init_set = set(range(91, N + 1))
            init_mask = 0
            for x in init_set:
                init_mask |= (1 << (x-1))
            # Add some random small numbers
            for _ in range(20):
                x = random.randint(1, 90)
                if x not in init_set and can_add_fast(x, init_mask):
                    init_mask |= (1 << (x-1))
                    init_set.add(x)
        
        run_local_search(init_mask, init_set, time_per_restart)
        restarts += 1
        
        # Check time
        if time.time() > deadline - 5:
            break

    return sorted(best_set)
