"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    N = 256
    
    def can_add(mask, x):
        """Check if adding x (1-indexed) to mask creates a 3-term AP.
        x is 1..256, bit i in mask represents number i (1-indexed, so bit 1 = number 1).
        We use bit position = value, so bit x represents number x.
        """
        # x as middle: need a and c such that a + c = 2x, a < x < c
        # a = x - d, c = x + d for d = 1, 2, ...
        for d in range(1, x):
            a = x - d
            c = x + d
            if c > N:
                break
            if (mask & (1 << a)) and (mask & (1 << c)):
                return False
        # x as left end: need b, c such that x + c = 2b, i.e., b = (x+c)/2
        # Equivalent: for each c > x, check if (x+c) is even and b=(x+c)/2 is in mask
        # Better: x is left end means x, b, c with b-x = c-b = d
        # So b = x+d, c = x+2d
        for d in range(1, (N - x + 1) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if c > N:
                break
            if (mask & (1 << b)) and (mask & (1 << c)):
                return False
        # x as right end: need a, b such that a + x = 2b, i.e., b = (a+x)/2
        # a = x - 2d, b = x - d
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            if (mask & (1 << a)) and (mask & (1 << b)):
                return False
        return True
    
    # Greedy: add numbers from 1 to 256 if they don't create a 3-AP
    mask = 0
    for x in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        if can_add(mask, x):
            mask |= (1 << x)
    
    # Convert to set for easier manipulation
    selected = set()
    for x in range(1, N + 1):
        if mask & (1 << x):
            selected.add(x)
    
    best_mask = mask
    best_score = sum(1.0 / x for x in selected)
    
    # Hill climbing: try add, remove, swap moves
    # Add move: try adding an unselected number
    # Remove move: try removing a selected number (only if it helps via subsequent adds)
    # Swap move: remove one, add another
    
    def reciprocal_sum_from_mask(m):
        s = 0.0
        for x in range(1, N + 1):
            if m & (1 << x):
                s += 1.0 / x
        return s
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try add moves: for each unselected x, try adding it
        for x in range(1, N + 1):
            if time.time() > deadline - 10:
                break
            if mask & (1 << x):
                continue
            if can_add(mask, x):
                new_mask = mask | (1 << x)
                new_score = reciprocal_sum_from_mask(new_mask)
                if new_score > best_score + 1e-15:
                    mask = new_mask
                    best_score = new_score
                    best_mask = mask
                    improved = True
        
        # Try swap moves: for each selected a and unselected b, try removing a and adding b
        # This is O(n^2) which is 256^2 = 65536, should be fast
        if time.time() > deadline - 10:
            break
        sel_list = [x for x in range(1, N + 1) if mask & (1 << x)]
        unsel_list = [x for x in range(1, N + 1) if not (mask & (1 << x))]
        
        # For efficiency, only try swaps where 1/b > 1/a (i.e., b < a) since we want to increase sum
        # Actually, we might remove a small number to add two medium numbers? No, swap is 1-for-1.
        # So only try if b < a (1/b > 1/a)
        for a in sel_list:
            if time.time() > deadline - 10:
                break
            for b in unsel_list:
                if b >= a:
                    continue
                # Try removing a and adding b
                new_mask = mask & ~(1 << a)
                if can_add(new_mask, b):
                    new_mask |= (1 << b)
                    new_score = reciprocal_sum_from_mask(new_mask)
                    if new_score > best_score + 1e-15:
                        mask = new_mask
                        best_score = new_score
                        best_mask = mask
                        improved = True
                        # Update lists
                        sel_list = [x for x in range(1, N + 1) if mask & (1 << x)]
                        unsel_list = [x for x in range(1, N + 1) if not (mask & (1 << x))]
                        break
    
    # Try multiple random restarts with different greedy orderings if time permits
    if time.time() < deadline - 20:
        rng = random.Random(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Randomized greedy: shuffle and pick
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            rmask = 0
            for x in order:
                if time.time() > deadline - 10:
                    break
                if can_add(rmask, x):
                    rmask |= (1 << x)
            
            rscore = reciprocal_sum_from_mask(rmask)
            if rscore > best_score:
                mask = rmask
                best_score = rscore
                best_mask = mask
            
            # Quick local search on this restart
            improved2 = True
            while improved2 and time.time() < deadline - 10:
                improved2 = False
                for x in range(1, N + 1):
                    if time.time() > deadline - 10:
                        break
                    if mask & (1 << x):
                        continue
                    if can_add(mask, x):
                        new_mask = mask | (1 << x)
                        new_score = reciprocal_sum_from_mask(new_mask)
                        if new_score > best_score + 1e-15:
                            mask = new_mask
                            best_score = new_score
                            best_mask = mask
                            improved2 = True
    
    # Final result
    result = [x for x in range(1, N + 1) if best_mask & (1 << x)]
    return sorted(result)
# EVOLVE-BLOCK-END
