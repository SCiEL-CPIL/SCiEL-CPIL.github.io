import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Precompute 4-AP conflicts
    # conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            # For a: conflicts are (b, c, e)
            conflicts[a].append((b, c, e))
            # For b: conflicts are (a, c, e)
            conflicts[b].append((a, c, e))
            # For c: conflicts are (a, b, e)
            conflicts[c].append((a, b, e))
            # For e: conflicts are (a, b, c)
            conflicts[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        # Iterate over all possible (start, diff)
        # A 3-AP is (i, i+d, i+2d)
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # This is exactly the same as delta_3aps_add for the current mask
        # because removing x breaks exactly the APs that adding it would create
        return delta_3aps_add(mask, x)

    # 2. Greedy Initialization
    def greedy_init():
        mask = 0
        # Sort elements randomly to break symmetry
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                # Check if adding it improves 3-AP count? 
                # For init, we just want a valid set. 
                # Maybe prioritize high delta?
                # Let's just add if valid to get a large set quickly.
                # Or add if delta > 0?
                # Let's try adding if delta > 0 or random to ensure diversity.
                if delta_3aps_add(mask, x) > 0 or random.random() < 0.3:
                    mask |= (1 << x)
        return mask

    # 3. Simulated Annealing
    best_mask = 0
    best_score = -1
    current_mask = greedy_init()
    current_score = count_3aps(current_mask)
    
    if current_score > best_score:
        best_score = current_score
        best_mask = current_mask

    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of bits in mask for removal/swap sampling? 
    # For N=256, iterating is fine.
    
    steps = 0
    while time.time() < deadline:
        # Perform a move
        move_type = random.random()
        
        new_mask = current_mask
        new_score = current_score
        valid_move = False
        
        if move_type < 0.5:
            # Add
            # Pick random x not in mask
            x = random.randint(0, N - 1)
            if not ((current_mask >> x) & 1):
                if is_valid_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    new_score = current_score + delta_3aps_add(current_mask, x)
                    valid_move = True
        elif move_type < 0.8:
            # Remove
            # Pick random x in mask
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                new_mask = current_mask & ~(1 << x)
                # Removing always results in a valid set (subset of valid set)
                new_score = current_score - delta_3aps_remove(current_mask, x)
                valid_move = True
        else:
            # Swap
            # Pick x in mask, y not in mask
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                y = random.randint(0, N - 1)
                if not ((current_mask >> y) & 1):
                    # Check if removing x and adding y is valid
                    # 1. Check if adding y to (mask \ {x}) is valid
                    temp_mask = current_mask & ~(1 << x)
                    if is_valid_add(temp_mask, y):
                        # Calculate score change
                        # Remove x: - delta_remove(mask, x)
                        # Add y to (mask \ {x}): + delta_add(temp_mask, y)
                        # Note: delta_remove uses the mask containing x.
                        # delta_add uses the mask NOT containing y.
                        # The interaction between x and y in 3-APs is handled correctly?
                        # Let's verify.
                        # Score(new) = Score(old) - DeltaRem(x, old) + DeltaAdd(y, old\{x})
                        # DeltaRem(x, old) counts APs containing x in old.
                        # DeltaAdd(y, old\{x}) counts APs containing y in old\{x}.
                        # If an AP contains both x and y, it is counted in DeltaRem (as it contains x) 
                        # and NOT in DeltaAdd (because x is not in old\{x}, so the AP is not "newly formed" by y? 
                        # Wait. DeltaAdd(y, M) counts APs {y, a, b} where a,b in M.
                        # If an AP is {x, y, z}, and we remove x first, M' = M\{x}.
                        # The AP {x, y, z} is broken. It is counted in DeltaRem(x, M).
                        # Then we add y to M'. The AP {x, y, z} is NOT formed because x is not in M'.
                        # So we don't double count or miss it.
                        # What if AP is {y, x, z}? Same logic.
                        # What if AP is {y, a, b} where a,b in M\{x}? Counted in DeltaAdd.
                        # What if AP is {x, a, b} where a,b in M\{x}? Counted in DeltaRem.
                        # So the formula holds.
                        
                        d_rem = delta_3aps_remove(current_mask, x)
                        d_add = delta_3aps_add(temp_mask, y)
                        new_score = current_score - d_rem + d_add
                        new_mask = temp_mask | (1 << y)
                        valid_move = True

        if valid_move:
            # Acceptance
            if new_score >= current_score:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                # Metropolis criterion
                delta = current_score - new_score
                if delta < 1e-9:
                    current_mask = new_mask
                    current_score = new_score
                else:
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        current_mask = new_mask
                        current_score = new_score
            
            steps += 1
            T *= cooling
            if T < T_min:
                T = T_min
                # Optional: restart or just keep going with min T
                # For now, just keep going.
        
        # Periodic restart if stuck?
        # If no improvement for a long time, maybe perturb?
        # Simple approach: if T is low, we are fine.
        
        # Check time
        if time.time() > deadline - 2.0:
            break

    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
