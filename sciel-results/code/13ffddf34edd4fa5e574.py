import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bits
    bits = [0] * (N + 1)
    for i in range(1, N + 1):
        bits[i] = 1 << (i - 1)
        
    # Precompute conflict checks for speed?
    # For each x, store list of (bit_y, bit_z)
    conflict_bits = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_bits[x].append((bits[y], bits[z]))

    def can_add(x, mask):
        for by, bz in conflict_bits[x]:
            if (mask & by) and (mask & bz):
                return False
        return True

    # Initial Solution: Upper half
    # mask represents the set S
    mask = 0
    for i in range(91, N + 1):
        mask |= bits[i]
        
    # Greedy addition
    for x in range(1, 91):
        if can_add(x, mask):
            mask |= bits[x]
            
    best_mask = mask
    best_size = bin(mask).count("1")
    
    random.seed(42)
    
    # Local Search
    stuck_count = 0
    max_stuck = 50
    
    # We will iterate many times
    # To avoid tuple(excluded) cost, we just pick random x and check if it's in mask
    # If we want to add, x must not be in mask.
    # If we want to swap, u must be in mask, x not in mask.
    
    for iteration in range(10000000):
        if time.time() > deadline:
            break
            
        # Pick a random element x to try to add
        # We bias towards smaller numbers? Or random?
        x = random.randint(1, N)
        
        if mask & bits[x]:
            # x is already in S. 
            # Maybe try to remove it? No, we want to maximize size.
            # Skip or try a swap?
            continue
            
        # Try to add x
        if can_add(x, mask):
            mask |= bits[x]
            size = bin(mask).count("1")
            if size > best_size:
                best_mask = mask
                best_size = size
            stuck_count = 0
            continue
            
        # x is blocked. Find conflicts
        # We need to find which pairs (y,z) are active.
        # conflicts_active = [(y,z) for y,z in conflict_pairs[x] if (mask & bits[y]) and (mask & bits[z])]
        
        # Instead of building a list, we can just iterate and pick a candidate to remove.
        # Strategy: Pick a random conflict, pick a random element from it to remove.
        # If that doesn't work, try another.
        
        found_move = False
        
        # Collect candidates for removal
        # To be efficient, we can just iterate through conflict_bits[x]
        # and if both bits are set, pick one randomly.
        
        # Let's try up to 10 random removal attempts
        for _ in range(10):
            # Pick a random conflict pair that is active
            # We need to pick from conflict_pairs[x]
            # If no active conflicts, can_add would have returned True.
            # So there is at least one.
            
            # Randomly select a pair from conflict_pairs[x]
            # This is not uniform over active pairs, but it's a heuristic.
            # Better: iterate through all, collect active ones, pick one.
            # Since number of conflicts is small, this is fine.
            
            active_conflicts = []
            for y, z in conflict_pairs[x]:
                if (mask & bits[y]) and (mask & bits[z]):
                    active_conflicts.append((y, z))
                    
            if not active_conflicts:
                # Should not happen if can_add returned False
                mask |= bits[x]
                found_move = True
                break
                
            # Pick a random active conflict
            y, z = random.choice(active_conflicts)
            
            # Try removing y
            mask ^= bits[y] # Remove y
            if can_add(x, mask):
                mask |= bits[x]
                found_move = True
                stuck_count = 0
                break
            else:
                mask ^= bits[y] # Restore y
                
            # Try removing z
            mask ^= bits[z]
            if can_add(x, mask):
                mask |= bits[x]
                found_move = True
                stuck_count = 0
                break
            else:
                mask ^= bits[z] # Restore z
        
        if found_move:
            size = bin(mask).count("1")
            if size > best_size:
                best_mask = mask
                best_size = size
            continue
        
        # If no single removal worked, try a kick
        if stuck_count >= max_stuck or random.random() < 0.05:
            # Remove 1-2 random elements from S
            kick_size = random.randint(1, 2)
            # Pick random elements from S
            # We can iterate through bits or just pick random indices and check
            removed = []
            for _ in range(kick_size):
                # Pick random element in S
                # Simple rejection sampling
                while True:
                    u = random.randint(1, N)
                    if mask & bits[u]:
                        removed.append(u)
                        mask ^= bits[u]
                        break
                        
            # Try to add x and maybe others
            if can_add(x, mask):
                mask |= bits[x]
            
            # Try to add some random others to recover size
            for _ in range(5):
                w = random.randint(1, N)
                if not (mask & bits[w]):
                    if can_add(w, mask):
                        mask |= bits[w]
                        
            stuck_count = 0
            size = bin(mask).count("1")
            if size > best_size:
                best_mask = mask
                best_size = size
            continue
            
        stuck_count += 1
        
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & bits[i]:
            result.append(i)
            
    return result
