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
        
    # Precompute conflict checks for speed
    conflict_bits = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_bits[x].append((bits[y], bits[z]))

    def can_add(x, mask):
        for by, bz in conflict_bits[x]:
            if (mask & by) and (mask & bz):
                return False
        return True

    # Initial Solution: Greedy from 1 to N
    mask = 0
    for i in range(1, N + 1):
        if can_add(i, mask):
            mask |= bits[i]
            
    best_mask = mask
    best_size = bin(mask).count("1")
    
    random.seed(42)
    
    # Local Search
    stuck_count = 0
    max_stuck = 50
    
    for iteration in range(10000000):
        if time.time() > deadline:
            break
            
        # Pick a random element x to try to add
        x = random.randint(1, N)
        
        if mask & bits[x]:
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
            
        # x is blocked. Try to find a swap or kick.
        # First, try 1-swap: Remove one element u that conflicts with x, then add x.
        found_swap = False
        active_pairs = []
        for y, z in conflict_pairs[x]:
            if (mask & bits[y]) and (mask & bits[z]):
                active_pairs.append((y, z))
                
        if active_pairs:
            # Try removing y or z from a random active pair
            y, z = random.choice(active_pairs)
            
            # Try removing y
            mask ^= bits[y]
            if can_add(x, mask):
                mask |= bits[x]
                found_swap = True
            else:
                mask ^= bits[y] # Restore
                
            if not found_swap:
                # Try removing z
                mask ^= bits[z]
                if can_add(x, mask):
                    mask |= bits[x]
                    found_swap = True
                else:
                    mask ^= bits[z] # Restore
        
        if found_swap:
            # We performed a swap. Check if size improved (it shouldn't, but maybe we can add more now?)
            # No, swap is 1-for-1. Size is same.
            # But we might be closer to a state where we can add.
            # Let's try to add some random elements to see if we can grow
            for _ in range(3):
                w = random.randint(1, N)
                if not (mask & bits[w]):
                    if can_add(w, mask):
                        mask |= bits[w]
                        size = bin(mask).count("1")
                        if size > best_size:
                            best_mask = mask
                            best_size = size
                            stuck_count = 0
                            break
            continue
        
        # If no swap worked, try a Kick
        if stuck_count >= max_stuck or random.random() < 0.05:
            # Remove 2-3 random elements
            kick_size = random.randint(2, 3)
            removed = []
            for _ in range(kick_size):
                while True:
                    u = random.randint(1, N)
                    if mask & bits[u]:
                        removed.append(u)
                        mask ^= bits[u]
                        break
                        
            # Try to add x
            if can_add(x, mask):
                mask |= bits[x]
                
            # Greedily add back as many as possible, prioritizing removed ones
            # First try to add removed ones back
            random.shuffle(removed)
            for u in removed:
                if can_add(u, mask):
                    mask |= bits[u]
                    
            # Then try random others
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
